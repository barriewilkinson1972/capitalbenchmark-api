"""Stress model utilities for the Capital Benchmark Flask API.

The API-facing function is ``run_stress``.  It returns a Bubble-friendly JSON
object containing:

- scenario inputs
- portfolio summary metrics
- unconditional vs conditional loss-distribution bins
- top stressed industries
- download metadata
- model metadata

The model has two layers:

1. Closed-form two-factor Vasicek PD stress using oil and AI factor loadings.
2. Monte Carlo loss distributions using an industry systematic factor plus
   obligor idiosyncratic epsilon.  The unconditional distribution uses base PDs;
   the conditional distribution uses stressed PDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


PORTFOLIO_PATH = "market_data/portfolio_metadata.parquet"
LOADINGS_PATH = "data/industry_factor_loadings.parquet"
INDUSTRY_CORR_PATH = "data/industry_corr_clean.parquet"

MODEL_VERSION = "0.2.0"
MODEL_NAME = "two_factor_vasicek_with_industry_mc"


@dataclass(frozen=True)
class StressConfig:
    """Runtime configuration for the stress simulation."""

    n_sims: int = 1000
    n_bins: int = 40
    asset_rho: float = 0.20
    random_seed: int = 42
    top_n: int = 10


def _to_float(value: Any, default: float) -> float:
    """Safely coerce API query parameters to floats."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    """Safely coerce API query parameters to ints."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_inputs(
    portfolio_path: str = PORTFOLIO_PATH,
    loadings_path: str = LOADINGS_PATH,
    industry_corr_path: str = INDUSTRY_CORR_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load portfolio, industry factor loadings, and optional industry correlation."""
    portfolio = pd.read_parquet(portfolio_path)
    loadings = pd.read_parquet(loadings_path)

    if "Industry" not in loadings.columns:
        loadings = loadings.reset_index()

    industry_corr: Optional[pd.DataFrame]
    try:
        industry_corr = pd.read_parquet(industry_corr_path)
    except FileNotFoundError:
        industry_corr = None

    return portfolio, loadings, industry_corr


def _prepare_obligors(
    portfolio: pd.DataFrame,
    loadings: pd.DataFrame,
    oil: float,
    ai: float,
    base_pd: float,
    lgd: float,
    ead: float,
    obligors: Optional[int] = None,
) -> pd.DataFrame:
    """Create obligor-level base and stressed PDs."""
    if "Industry" not in portfolio.columns:
        raise ValueError("portfolio data must contain an 'Industry' column")

    df = portfolio.copy()

    # Optional MVP input: let a user scale the demo portfolio size while keeping
    # the industry mix approximately unchanged.
    if obligors is not None and obligors > 0 and obligors != len(df):
        replace = obligors > len(df)
        df = df.sample(n=obligors, replace=replace, random_state=42).reset_index(drop=True)

    loadings = loadings.copy()
    required_cols = {"Industry", "oil_corr", "ai_corr"}
    missing = required_cols.difference(loadings.columns)
    if missing:
        raise ValueError(f"loadings data missing required columns: {sorted(missing)}")

    loadings = loadings.set_index("Industry")

    df["base_pd"] = float(base_pd)
    df["lgd"] = float(lgd)
    df["ead"] = float(ead)

    df["oil_corr"] = df["Industry"].map(loadings["oil_corr"]).fillna(0.0).astype(float)
    df["ai_corr"] = df["Industry"].map(loadings["ai_corr"]).fillna(0.0).astype(float)

    # Sign convention: positive oil means a positive oil-price shock; negative AI
    # means AI optimism reversal. Negative factor impact is bad for credit.
    factor_impact = df["oil_corr"] * float(oil) + df["ai_corr"] * float(ai)

    r2 = (df["oil_corr"] ** 2 + df["ai_corr"] ** 2).clip(0.0, 0.999)
    residual_vol = np.sqrt(1.0 - r2)
    threshold = norm.ppf(df["base_pd"].clip(1e-8, 1.0 - 1e-8))

    df["stressed_pd"] = norm.cdf((threshold - factor_impact) / residual_vol)
    df["pd_change"] = df["stressed_pd"] - df["base_pd"]
    df["pd_multiple"] = np.where(df["base_pd"] > 0, df["stressed_pd"] / df["base_pd"], np.nan)

    df["base_expected_loss"] = df["base_pd"] * df["lgd"] * df["ead"]
    df["stressed_expected_loss"] = df["stressed_pd"] * df["lgd"] * df["ead"]

    return df


def _industry_correlation_matrix(
    industries: List[str], industry_corr: Optional[pd.DataFrame]
) -> np.ndarray:
    """Return a PSD-ish industry correlation matrix aligned to the portfolio."""
    n = len(industries)
    if industry_corr is None:
        return np.eye(n)

    corr = industry_corr.copy()
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)

    aligned = corr.reindex(index=industries, columns=industries).fillna(0.0)

    matrix = aligned.to_numpy(copy=True).astype(float)
    np.fill_diagonal(matrix, 1.0)

    matrix = (matrix + matrix.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-8, None)
    matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    std = np.sqrt(np.diag(matrix))
    matrix = matrix / np.outer(std, std)
    np.fill_diagonal(matrix, 1.0)

    return matrix


def _simulate_losses(
    df: pd.DataFrame,
    industry_corr: Optional[pd.DataFrame],
    config: StressConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate unconditional and conditional portfolio loss distributions.

    The unconditional distribution uses base PDs.  The conditional distribution
    uses the oil/AI stressed PDs, but both distributions retain the same residual
    stochastic structure: an industry systematic factor plus obligor epsilon.
    """
    rng = np.random.default_rng(config.random_seed)

    industries = sorted(df["Industry"].astype(str).unique().tolist())
    industry_to_idx = {industry: i for i, industry in enumerate(industries)}
    industry_idx = df["Industry"].astype(str).map(industry_to_idx).to_numpy()

    corr_matrix = _industry_correlation_matrix(industries, industry_corr)

    systematic = rng.multivariate_normal(
        mean=np.zeros(len(industries)),
        cov=corr_matrix,
        size=config.n_sims,
        method="svd",
    )

    eps = rng.standard_normal(size=(config.n_sims, len(df)))

    rho = float(np.clip(config.asset_rho, 0.0, 0.999))
    latent = np.sqrt(rho) * systematic[:, industry_idx] + np.sqrt(1.0 - rho) * eps

    base_threshold = norm.ppf(df["base_pd"].to_numpy().clip(1e-8, 1.0 - 1e-8))
    stressed_threshold = norm.ppf(df["stressed_pd"].to_numpy().clip(1e-8, 1.0 - 1e-8))

    loss_given_default = (df["lgd"] * df["ead"]).to_numpy()

    unconditional_losses = (latent < base_threshold) @ loss_given_default
    conditional_losses = (latent < stressed_threshold) @ loss_given_default

    return unconditional_losses, conditional_losses


def _bin_loss_distributions(
    unconditional_losses: np.ndarray,
    conditional_losses: np.ndarray,
    n_bins: int,
) -> List[Dict[str, float]]:
    """Return Bubble-friendly probability bins for two loss distributions."""
    max_loss = float(max(unconditional_losses.max(), conditional_losses.max(), 1.0))
    bins = np.linspace(0.0, max_loss, int(n_bins) + 1)

    uncond_counts, _ = np.histogram(unconditional_losses, bins=bins)
    cond_counts, _ = np.histogram(conditional_losses, bins=bins)

    uncond_probs = uncond_counts / max(1, len(unconditional_losses))
    cond_probs = cond_counts / max(1, len(conditional_losses))

    rows: List[Dict[str, float]] = []
    for i in range(len(bins) - 1):
        rows.append(
            {
                "loss_bin": float((bins[i] + bins[i + 1]) / 2.0),
                "loss_bin_min": float(bins[i]),
                "loss_bin_max": float(bins[i + 1]),
                "unconditional": float(uncond_probs[i]),
                "conditional": float(cond_probs[i]),
            }
        )

    return rows


def _loss_percentiles(losses: np.ndarray) -> Dict[str, float]:
    """Common loss-distribution summary metrics."""
    return {
        "mean": float(np.mean(losses)),
        "p50": float(np.percentile(losses, 50)),
        "p95": float(np.percentile(losses, 95)),
        "p99": float(np.percentile(losses, 99)),
        "max": float(np.max(losses)),
    }


def _top_industries(df: pd.DataFrame, top_n: int) -> List[Dict[str, Any]]:
    """Aggregate industry-level stress results."""
    industry = (
        df.groupby("Industry")
        .agg(
            obligors=("Industry", "size"),
            base_pd=("base_pd", "mean"),
            stressed_pd=("stressed_pd", "mean"),
            base_expected_loss=("base_expected_loss", "sum"),
            expected_loss=("stressed_expected_loss", "sum"),
            oil_corr=("oil_corr", "mean"),
            ai_corr=("ai_corr", "mean"),
        )
        .reset_index()
        .rename(columns={"Industry": "industry"})
    )

    industry["pd_change"] = industry["stressed_pd"] - industry["base_pd"]
    industry["pd_multiple"] = np.where(
        industry["base_pd"] > 0,
        industry["stressed_pd"] / industry["base_pd"],
        np.nan,
    )
    industry["expected_loss_change"] = industry["expected_loss"] - industry["base_expected_loss"]

    cols = [
        "industry",
        "obligors",
        "base_pd",
        "stressed_pd",
        "pd_change",
        "pd_multiple",
        "base_expected_loss",
        "expected_loss",
        "expected_loss_change",
        "oil_corr",
        "ai_corr",
    ]

    return (
        industry[cols]
        .sort_values("expected_loss_change", ascending=False)
        .head(int(top_n))
        .replace({np.nan: None})
        .to_dict(orient="records")
    )


def run_stress(
    oil: float = 0.0,
    ai: float = 0.0,
    pd_base: float = 0.01,
    lgd: float = 0.45,
    ead: float = 1.0,
    obligors: Optional[int] = None,
    n_sims: int = 10_000,
    n_bins: int = 40,
    asset_rho: float = 0.20,
    random_seed: int = 42,
    include_legacy_fields: bool = True,
) -> Dict[str, Any]:
    """Run the stress scenario and return the API JSON response.

    Parameters are intentionally primitive because they come directly from
    Bubble/API query-string inputs.
    """
    oil = _to_float(oil, 0.0)
    ai = _to_float(ai, 0.0)
    pd_base = _to_float(pd_base, 0.01)
    lgd = _to_float(lgd, 0.45)
    ead = _to_float(ead, 1.0)
    n_sims = _to_int(n_sims, 10_000)
    n_bins = _to_int(n_bins, 40)
    random_seed = _to_int(random_seed, 42)
    asset_rho = _to_float(asset_rho, 0.20)

    pd_base = float(np.clip(pd_base, 1e-8, 1.0 - 1e-8))
    lgd = float(np.clip(lgd, 0.0, 1.0))
    ead = float(max(ead, 0.0))
    n_sims = int(np.clip(n_sims, 100, 100_000))
    n_bins = int(np.clip(n_bins, 5, 100))

    portfolio, loadings, industry_corr = _load_inputs()
    df = _prepare_obligors(
        portfolio=portfolio,
        loadings=loadings,
        oil=oil,
        ai=ai,
        base_pd=pd_base,
        lgd=lgd,
        ead=ead,
        obligors=obligors,
    )

    config = StressConfig(
        n_sims=n_sims,
        n_bins=n_bins,
        asset_rho=asset_rho,
        random_seed=random_seed,
    )

    unconditional_losses, conditional_losses = _simulate_losses(df, industry_corr, config)
    loss_distribution = _bin_loss_distributions(unconditional_losses, conditional_losses, n_bins=n_bins)

    base_portfolio_pd = float(df["base_pd"].mean())
    stressed_portfolio_pd = float(df["stressed_pd"].mean())
    base_expected_loss = float(df["base_expected_loss"].sum())
    stressed_expected_loss = float(df["stressed_expected_loss"].sum())
    expected_loss_change = stressed_expected_loss - base_expected_loss
    expected_loss_multiple = (
        stressed_expected_loss / base_expected_loss if base_expected_loss > 0 else None
    )

    response: Dict[str, Any] = {
        "scenario": {
            "oil": float(oil),
            "ai": float(ai),
            "base_pd": float(pd_base),
            "lgd": float(lgd),
            "ead": float(ead),
            "obligors": int(len(df)),
            "requested_obligors": int(obligors) if obligors is not None else None,
            "n_sims": int(n_sims),
            "n_bins": int(n_bins),
            "asset_rho": float(asset_rho),
            "random_seed": int(random_seed),
        },
        "summary": {
            "base_portfolio_pd": base_portfolio_pd,
            "base_portfolio_pd_percent": base_portfolio_pd * 100.0,
            "stressed_portfolio_pd": stressed_portfolio_pd,
            "stressed_portfolio_pd_percent": stressed_portfolio_pd * 100.0,
            "base_expected_loss": base_expected_loss,
            "stressed_expected_loss": stressed_expected_loss,
            "expected_loss_change": expected_loss_change,
            "expected_loss_multiple": expected_loss_multiple,
            "unconditional_loss": _loss_percentiles(unconditional_losses),
            "conditional_loss": _loss_percentiles(conditional_losses),
        },
        "loss_distribution": loss_distribution,
        "top_industries": _top_industries(df, top_n=config.top_n),
        "download": {
            "available": False,
            "requires_registration": True,
            "endpoint": "/download_scenario",
        },
        "model_info": {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "factors": ["oil", "ai"],
            "portfolio_path": PORTFOLIO_PATH,
            "loadings_path": LOADINGS_PATH,
            "industry_corr_path": INDUSTRY_CORR_PATH,
        },
    }

    # Temporary backward compatibility for existing Bubble bindings.
    if include_legacy_fields:
        response.update(
            {
                "oil": float(oil),
                "ai": float(ai),
                "portfolio_pd": stressed_portfolio_pd,
                "portfolio_pd_percent": stressed_portfolio_pd * 100.0,
                "expected_loss": stressed_expected_loss,
            }
        )

    return response
