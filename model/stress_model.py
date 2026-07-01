from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from functools import lru_cache


# -----------------------------------------------------------------------------
# File locations
# -----------------------------------------------------------------------------

PORTFOLIO_PATH = "market_data/obligor_data.csv"
LOADINGS_PATH = "data/industry_factor_loadings.parquet"

MODEL_NAME = "obligor_pd_ead_three_factor_deterministic_vasicek"
MODEL_VERSION = "2.0.0"

# Explicit obligor-level model inputs from the new obligor dataset.
PD_COLUMN = "CB pd"
EAD_COLUMN = "totalDebt_usd"
INDUSTRY_COLUMN = "industry"

FACTOR_COLUMNS = ["rho_Market", "rho_Technology", "rho_Commodity"]
FACTOR_NAMES = ["market", "technology", "commodity"]

@lru_cache(maxsize=1)
def _load_inputs_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load obligors and industry factor loadings once per server process."""
    portfolio = pd.read_csv(PORTFOLIO_PATH)

    loadings = pd.read_parquet(LOADINGS_PATH)

    if "industry" not in loadings.columns:
        loadings = loadings.reset_index()

    if "Industry" in loadings.columns and "industry" not in loadings.columns:
        loadings = loadings.rename(columns={"Industry": "industry"})

    loadings["industry"] = loadings["industry"].astype(str).str.strip()

    return portfolio, loadings


@dataclass(frozen=True)
class StressConfig:
    """Runtime configuration for the Monte Carlo loss simulation."""

    n_sims: int = 10_000
    n_bins: int = 40
    asset_rho: float = 0.20
    random_seed: int = 42
    top_n: int = 100


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


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



# def _clean_text_series(s: pd.Series) -> pd.Series:
#     """Normalize text keys used for joins."""
#     return s.astype(str).str.strip()



def _required_columns(df: pd.DataFrame, columns: List[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")




def _one_tailed_probability(z: float) -> float:
    """
    Probability of a one-sided shock at least as extreme as z.

    z = 0 is treated as no constraint, so probability = 1.
    """
    z = float(z)

    if z > 0:

        z = 0


    return float(2 * norm.sf(abs(z)))


def _two_tailed_probability(z: float) -> float:
    """
    Probability of a two-sided shock at least as extreme as |z|.

    z = 0 naturally gives probability = 1.
    """
    z = float(z)

    return float(2.0 * norm.sf(abs(z)))


def _scenario_tail_probability(
    market: float,
    technology: float,
    commodity: float,
) -> dict:
    """
    Calculate scenario tail probability assuming independent standard normal factors.

    Market is one-tailed.
    Technology and commodity are two-tailed.
    """

    market_prob = _one_tailed_probability(market)
    technology_prob = _two_tailed_probability(technology)
    commodity_prob = _two_tailed_probability(commodity)

    joint_prob = np.round(market_prob * technology_prob * commodity_prob, 4)

    return {
        "market_tail_probability": market_prob,
        "technology_tail_probability": technology_prob,
        "commodity_tail_probability": commodity_prob,
        "scenario_tail_probability": 1 - joint_prob,
        "scenario_tail_odds": (
            int(np.round(1.0 / joint_prob, 0)) if joint_prob > 0 else None
        ),
    }

# -----------------------------------------------------------------------------
# Input loading and schema normalisation
# -----------------------------------------------------------------------------


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or parquet based on file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type for {path}")



def _normalise_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    """Return industry loadings with canonical column names and one row per industry."""
    loadings = loadings.copy()

    # Some saved parquet files may have industry as the index rather than a column.
    if INDUSTRY_COLUMN not in loadings.columns:
        if loadings.index.name == INDUSTRY_COLUMN or loadings.index.name is not None:
            loadings = loadings.reset_index()
        else:
            loadings = loadings.reset_index().rename(columns={"index": INDUSTRY_COLUMN})

    rename_map = {
        "Industry": INDUSTRY_COLUMN,
        "rho_market": "rho_Market",
        "rho_technology": "rho_Technology",
        "rho_tech": "rho_Technology",
        "rho_commodity": "rho_Commodity",
    }
    loadings = loadings.rename(columns=rename_map)

    _required_columns(loadings, [INDUSTRY_COLUMN, *FACTOR_COLUMNS], "industry factor loadings")

    for col in FACTOR_COLUMNS:
        loadings[col] = pd.to_numeric(loadings[col], errors="coerce").fillna(0.0)

    if "residual_rho" not in loadings.columns:
        rho_sq = (loadings[FACTOR_COLUMNS] ** 2).sum(axis=1).clip(0.0, 0.999999)
        loadings["residual_rho"] = np.sqrt(1.0 - rho_sq)
    else:
        loadings["residual_rho"] = pd.to_numeric(
            loadings["residual_rho"], errors="coerce"
        )
        missing_residual = loadings["residual_rho"].isna()
        if missing_residual.any():
            rho_sq = (loadings.loc[missing_residual, FACTOR_COLUMNS] ** 2).sum(axis=1)
            loadings.loc[missing_residual, "residual_rho"] = np.sqrt(
                1.0 - rho_sq.clip(0.0, 0.999999)
            )

    # Numerical safety for the conditional PD denominator.
    loadings["residual_rho"] = loadings["residual_rho"].clip(lower=1e-6)

    # If duplicate industries exist, keep the first. This mirrors the old lookup behaviour.
    loadings = loadings.drop_duplicates(subset=[INDUSTRY_COLUMN], keep="first")

    return loadings[[INDUSTRY_COLUMN, *FACTOR_COLUMNS, "residual_rho"]]



# def _normalise_industry_corr(industry_corr: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
#     """Normalize optional industry correlation matrix labels."""
#     if industry_corr is None:
#         return None

#     corr = industry_corr.copy()

#     # Parquet can sometimes preserve a named index cleanly; CSV-style tables may
#     # include an unnamed first column. Handle both cases defensively.
#     unnamed_cols = [c for c in corr.columns if str(c).startswith("Unnamed")]
#     if unnamed_cols:
#         corr = corr.set_index(unnamed_cols[0])

#     corr.index = corr.index.astype(str).str.strip()
#     corr.columns = corr.columns.astype(str).str.strip()
#     corr = corr.apply(pd.to_numeric, errors="coerce")

#     return corr



# def _load_inputs(
#     portfolio_path: str = PORTFOLIO_PATH,
#     loadings_path: str = LOADINGS_PATH,
#     industry_corr_path: str = INDUSTRY_CORR_PATH,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
#     """Load obligors, three-factor industry loadings, and optional industry correlation."""
#     portfolio = pd.read_csv(portfolio_path)
#     loadings = _normalise_loadings(_read_table(loadings_path))

#     try:
#         industry_corr = _normalise_industry_corr(_read_table(industry_corr_path))
#     except FileNotFoundError:
#         industry_corr = None

#     return portfolio, loadings, industry_corr


# -----------------------------------------------------------------------------
# Obligor-level PD stress transformation
# -----------------------------------------------------------------------------


def _prepare_obligors(
    portfolio: pd.DataFrame,
    loadings: pd.DataFrame,
    market: float,
    technology: float,
    commodity: float,
    lgd: float,
    obligors: Optional[int] = None,
) -> pd.DataFrame:
    """Create obligor-level base and stressed PDs using supplied PD and EAD."""
    df = portfolio.copy()

    _required_columns(
        df,
        ["symbol", "company_name", INDUSTRY_COLUMN, PD_COLUMN, EAD_COLUMN],
        "obligor portfolio",
    )

    df["base_pd"] = pd.to_numeric(df[PD_COLUMN], errors="coerce")
    df["ead"] = pd.to_numeric(df[EAD_COLUMN], errors="coerce")

    df = df.dropna(subset=[INDUSTRY_COLUMN, "base_pd", "ead"])
    df = df[df["ead"] > 0].copy()

    # Optional portfolio-size control for demo/API speed. Use largest EAD names
    # rather than arbitrary first rows.
    if obligors is not None:
        obligors = int(obligors)
        if obligors > 0 and obligors < len(df):
            df = df.sort_values("ead", ascending=False).head(obligors).copy()

    df["base_pd"] = df["base_pd"].clip(1e-8, 1.0 - 1e-8)
    df["ead"] = df["ead"].clip(lower=0.0)
    df["lgd"] = float(lgd)

    loadings_idx = loadings.set_index(INDUSTRY_COLUMN)

    for col in FACTOR_COLUMNS:
        df[col] = df[INDUSTRY_COLUMN].map(loadings_idx[col]).fillna(0.0).astype(float)

    df["residual_rho"] = (
        df[INDUSTRY_COLUMN]
        .map(loadings_idx["residual_rho"])
        .fillna(1.0)
        .astype(float)
        .clip(lower=1e-6)
    )

    # Conditional Vasicek PD transformation.
    # Positive factor shock is good for industries with positive loading and bad
    # for industries with negative loading. Negative market shock therefore raises
    # PDs for industries with positive market exposure.
    factor_impact = (
        df["rho_Market"] * float(market)
        + df["rho_Technology"] * float(technology)
        + df["rho_Commodity"] * float(commodity)
    )

    threshold = norm.ppf(df["base_pd"].to_numpy())
    residual = df["residual_rho"].to_numpy()

    df["stressed_pd"] = norm.cdf((threshold - factor_impact.to_numpy()) / residual)
    df["stressed_pd"] = df["stressed_pd"].clip(1e-8, 1.0 - 1e-8)

    df["pd_change"] = df["stressed_pd"] - df["base_pd"]
    df["pd_multiple"] = np.where(
        df["base_pd"] > 0,
        df["stressed_pd"] / df["base_pd"],
        np.nan,
    )

    df["base_expected_loss"] = df["base_pd"] * df["lgd"] * df["ead"]
    df["stressed_expected_loss"] = df["stressed_pd"] * df["lgd"] * df["ead"]
    df["expected_loss_change"] = df["stressed_expected_loss"] - df["base_expected_loss"]

    df.attrs["pd_column"] = PD_COLUMN
    df.attrs["ead_column"] = EAD_COLUMN

    return df


# -----------------------------------------------------------------------------
# Monte Carlo loss distribution
# -----------------------------------------------------------------------------


# def _industry_correlation_matrix(
#     industries: List[str], industry_corr: Optional[pd.DataFrame]
# ) -> np.ndarray:
#     """Return a PSD industry correlation matrix aligned to the current portfolio."""
#     n = len(industries)
#     if n == 0:
#         return np.empty((0, 0))

#     if industry_corr is None:
#         return np.eye(n)

#     corr = industry_corr.copy()
#     corr.index = corr.index.astype(str).str.strip()
#     corr.columns = corr.columns.astype(str).str.strip()

#     aligned = corr.reindex(index=industries, columns=industries).fillna(0.0)

#     matrix = aligned.to_numpy(copy=True).astype(float)
#     np.fill_diagonal(matrix, 1.0)

#     # Symmetrise and repair to positive semi-definite.
#     matrix = (matrix + matrix.T) / 2.0

#     eigvals, eigvecs = np.linalg.eigh(matrix)
#     eigvals = np.clip(eigvals, 1e-8, None)
#     matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

#     std = np.sqrt(np.diag(matrix))
#     matrix = matrix / np.outer(std, std)
#     np.fill_diagonal(matrix, 1.0)

#     return matrix



# def _simulate_losses(
#     df: pd.DataFrame,
#     industry_corr: Optional[pd.DataFrame],
#     config: StressConfig,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Simulate unconditional and conditional portfolio loss distributions.

#     The unconditional distribution uses obligor-level base PDs. The conditional
#     distribution uses the three-factor stressed PDs. Both retain the same residual
#     stochastic structure: an industry systematic factor plus obligor idiosyncratic
#     epsilon.
#     """
#     if df.empty:
#         return np.array([0.0]), np.array([0.0])

#     rng = np.random.default_rng(config.random_seed)

#     industries = sorted(df[INDUSTRY_COLUMN].astype(str).unique().tolist())
#     industry_to_idx = {industry: i for i, industry in enumerate(industries)}
#     industry_idx = df[INDUSTRY_COLUMN].astype(str).map(industry_to_idx).to_numpy()

#     corr_matrix = _industry_correlation_matrix(industries, industry_corr)

#     systematic = rng.multivariate_normal(
#         mean=np.zeros(len(industries)),
#         cov=corr_matrix,
#         size=config.n_sims,
#         method="svd",
#     )

#     eps = rng.standard_normal(size=(config.n_sims, len(df)))

#     rho = float(np.clip(config.asset_rho, 0.0, 0.999))
#     latent = np.sqrt(rho) * systematic[:, industry_idx] + np.sqrt(1.0 - rho) * eps

#     base_threshold = norm.ppf(df["base_pd"].to_numpy().clip(1e-8, 1.0 - 1e-8))
#     stressed_threshold = norm.ppf(df["stressed_pd"].to_numpy().clip(1e-8, 1.0 - 1e-8))

#     loss_given_default = (df["lgd"] * df["ead"]).to_numpy()

#     unconditional_losses = (latent < base_threshold) @ loss_given_default
#     conditional_losses = (latent < stressed_threshold) @ loss_given_default

#     return unconditional_losses, conditional_losses



# def _bin_loss_distributions(
#     unconditional_losses: np.ndarray,
#     conditional_losses: np.ndarray,
#     n_bins: int,
# ) -> List[Dict[str, float]]:
#     """Return Bubble-friendly probability bins for two loss distributions."""
#     max_loss = float(max(unconditional_losses.max(), conditional_losses.max(), 1.0))
#     bins = np.linspace(0.0, max_loss, int(n_bins) + 1)

#     uncond_counts, _ = np.histogram(unconditional_losses, bins=bins)
#     cond_counts, _ = np.histogram(conditional_losses, bins=bins)

#     uncond_probs = uncond_counts / max(1, len(unconditional_losses))
#     cond_probs = cond_counts / max(1, len(conditional_losses))

#     rows: List[Dict[str, float]] = []
#     for i in range(len(bins) - 1):
#         rows.append(
#             {
#                 "loss_bin": float((bins[i] + bins[i + 1]) / 2.0),
#                 "loss_bin_min": float(bins[i]),
#                 "loss_bin_max": float(bins[i + 1]),
#                 "unconditional": float(uncond_probs[i]),
#                 "conditional": float(cond_probs[i]),
#             }
#         )

#     return rows



# def _loss_percentiles(losses: np.ndarray) -> Dict[str, float]:
#     """Common loss-distribution summary metrics."""
#     return {
#         "mean": float(np.mean(losses)),
#         "p50": float(np.percentile(losses, 50)),
#         "p95": float(np.percentile(losses, 95)),
#         "p99": float(np.percentile(losses, 99)),
#         "max": float(np.max(losses)),
#     }


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Weighted average with safe fallback to simple mean."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").clip(lower=0.0)
    valid = v.notna() & w.notna()
    if not valid.any() or w[valid].sum() <= 0:
        return float(v.mean())
    return float(np.average(v[valid], weights=w[valid]))

def _pd_to_radius(
    pd_values,
    pd_floor: float = 0.001,
    pd_cap: float = 0.20,
    max_radius: float = 1.0,
) -> np.ndarray:
    """Map PD to radial distance using log scaling."""
    p = np.clip(np.asarray(pd_values, dtype=float), pd_floor, pd_cap)

    log_p = np.log10(p)
    log_lo = np.log10(pd_floor)
    log_hi = np.log10(pd_cap)

    radius = (log_p - log_lo) / (log_hi - log_lo)
    radius = np.clip(radius, 0.0, 1.0)

    return radius * max_radius

def colors_ind(pd_values):

    colors = []

    for p in pd_values:

        if p > 0.05:

            colors.append("red")

        
        elif p<0.005:

            colors.append("green")

        else:

            colors.append("amber")

    return colors

def _top_industries(
    df: pd.DataFrame,
    market: float = 0.0,
    technology: float = 0.0,
    commodity: float = 0.0,
    pd_floor: float = 0.001,
    pd_cap: float = 0.20,
    max_radius: float = 1.0,
) -> List[Dict[str, Any]]:
    """Aggregate industry-level stress results, including XY plot coordinates."""

    grouped = []

    for industry, g in df.groupby(INDUSTRY_COLUMN):
        grouped.append(
            {
                "industry": industry,
                "obligors": int(len(g)),
                "ead": float(g["ead"].sum()),

                "base_pd": _weighted_average(g["base_pd"], g["ead"]),
                "stressed_pd": _weighted_average(g["stressed_pd"], g["ead"]),

                "base_expected_loss": float(g["base_expected_loss"].sum()),
                "stressed_expected_loss": float(g["stressed_expected_loss"].sum()),
                "expected_loss_change": float(g["expected_loss_change"].sum()),

                # These should be identical within an industry, but mean is safe.
                "rho_Market": float(g["rho_Market"].mean()),
                "rho_Technology": float(g["rho_Technology"].mean()),
                "rho_Commodity": float(g["rho_Commodity"].mean()),
                "residual_rho": float(g["residual_rho"].mean()),
            }
        )

    industry_df = pd.DataFrame(grouped)

    if industry_df.empty:
        return []

    industry_df["pd_change"] = (
        industry_df["stressed_pd"] - industry_df["base_pd"]
    )

    industry_df["pd_multiple"] = np.where(
        industry_df["base_pd"] > 0,
        industry_df["stressed_pd"] / industry_df["base_pd"],
        np.nan,
    )

    # ------------------------------------------------------------
    # XY risk-map coordinates
    # ------------------------------------------------------------

    # Scenario-weighted structural angle.
    # This keeps commodity-positive industries on the right and
    # commodity-negative industries on the left, regardless of whether
    # the commodity slider is + or -.
    x_driver = industry_df["rho_Commodity"] * abs(float(commodity))
    y_driver = industry_df["rho_Technology"] * abs(float(technology))

    driver_strength = np.sqrt(x_driver**2 + y_driver**2)

    structural_angle = np.arctan2(
        industry_df["rho_Technology"],
        industry_df["rho_Commodity"],
    )

    scenario_angle = np.arctan2(
        y_driver,
        x_driver,
    )

    industry_df["risk_map_angle"] = np.where(
        driver_strength > 1e-8,
        scenario_angle,
        structural_angle,
    )

    # Radius comes from Vasicek stressed PD.
    industry_df["risk_map_radius"] = _pd_to_radius(
        industry_df["stressed_pd"],
        pd_floor=pd_floor,
        pd_cap=pd_cap,
        max_radius=max_radius,
    )

    industry_df["color"] = colors_ind(industry_df["stressed_pd"])

    industry_df["x_plot"] = np.round((
        industry_df["risk_map_radius"]
        * np.cos(industry_df["risk_map_angle"])
    ),4)

    industry_df["y_plot"] = np.round((
        industry_df["risk_map_radius"]
        * np.sin(industry_df["risk_map_angle"])
    ),4)

    # Optional extras useful for Bubble tooltips / debugging
    industry_df["commodity_driver"] = x_driver
    industry_df["technology_driver"] = y_driver
    industry_df["circle_size"]=np.sqrt(industry_df["stressed_expected_loss"]/50000000)

    cols = [
        "industry",
        "obligors",
        "ead",
        "circle_size",
        "color",
        "base_pd",
        "stressed_pd",
        "pd_change",
        "pd_multiple",

        "base_expected_loss",
        "stressed_expected_loss",
        "expected_loss_change",

        "rho_Market",
        "rho_Technology",
        "rho_Commodity",
        "residual_rho",

        "risk_map_radius",
        "risk_map_angle",
        "x_plot",
        "y_plot",
        "commodity_driver",
        "technology_driver",
    ]

    return (
        industry_df[cols]
        .sort_values("stressed_pd", ascending=False)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )



def _top_obligors(df: pd.DataFrame, top_n: int) -> List[Dict[str, Any]]:
    """Return top obligors by stressed expected loss for debugging/download UI."""
    display_cols = [
        "symbol",
        "company_name",
        "Agency Rating",
        "cb_rating",
        "is_rated_obligor",
        INDUSTRY_COLUMN,
        "sector",
        "country",
        "ead",
        "base_pd",
        "stressed_pd",
        "pd_change",
        "pd_multiple",
        "base_expected_loss",
        "stressed_expected_loss",
        "expected_loss_change",
        "rho_Market",
        "rho_Technology",
        "rho_Commodity",
        "residual_rho",
    ]
    cols = [c for c in display_cols if c in df.columns]

    return (
        df[cols]
        .sort_values("stressed_expected_loss", ascending=False)
        .head(int(top_n))
        .replace({np.nan: None})
        .to_dict(orient="records")
    )


# -----------------------------------------------------------------------------
# Public API entrypoint
# -----------------------------------------------------------------------------


def run_stress(
    market: float = 0.0,
    technology: float = 0.0,
    commodity: float = 0.0,
    lgd: float = 0.45,
    obligors: Optional[int] = None,
    top_n: int = 10,
    include_empty_loss_distribution: bool = True,
) -> Dict[str, Any]:
    """Run deterministic three-factor Vasicek stressed PD calculation.

    No Monte Carlo simulation is performed. The endpoint returns obligor-level
    and industry-level Vasicek stressed PDs and expected losses.
    """

    market = _to_float(market, 0.0)
    technology = _to_float(technology, 0.0)
    commodity = _to_float(commodity, 0.0)
    lgd = float(np.clip(_to_float(lgd, 0.45), 0.0, 1.0))
    top_n = int(np.clip(_to_int(top_n, 10), 1, 100))

    scenario_likelihood = _scenario_tail_probability(
    market=market,
    technology=technology,
    commodity=commodity)

    portfolio, loadings = _load_inputs_cached()

    df = _prepare_obligors(
        portfolio=portfolio,
        loadings=loadings,
        market=market,
        technology=technology,
        commodity=commodity,
        lgd=lgd,
        obligors=obligors,
    )

    total_ead = float(df["ead"].sum())

    base_portfolio_pd = _weighted_average(df["base_pd"], df["ead"])
    stressed_portfolio_pd = _weighted_average(df["stressed_pd"], df["ead"])

    simple_base_portfolio_pd = float(df["base_pd"].mean())
    simple_stressed_portfolio_pd = float(df["stressed_pd"].mean())

    base_expected_loss = float(df["base_expected_loss"].sum())
    stressed_expected_loss = float(df["stressed_expected_loss"].sum())

    expected_loss_change = stressed_expected_loss - base_expected_loss
    expected_loss_multiple = (
        stressed_expected_loss / base_expected_loss
        if base_expected_loss > 0
        else None
    )

    response: Dict[str, Any] = {
        "scenario": {
            "market": float(market),
            "technology": float(technology),
            "commodity": float(commodity),
            "lgd": float(lgd),
            "obligors": int(len(df)),
            "requested_obligors": int(obligors) if obligors is not None else None,
            "top_n": int(top_n),
        },
        "summary": {
            "total_ead": total_ead,

            "base_portfolio_pd": base_portfolio_pd,
            "base_portfolio_pd_percent": base_portfolio_pd * 100.0,

            "stressed_portfolio_pd": stressed_portfolio_pd,
            "stressed_portfolio_pd_percent": stressed_portfolio_pd * 100.0,

            "simple_base_portfolio_pd": simple_base_portfolio_pd,
            "simple_base_portfolio_pd_percent": simple_base_portfolio_pd * 100.0,

            "simple_stressed_portfolio_pd": simple_stressed_portfolio_pd,
            "simple_stressed_portfolio_pd_percent": simple_stressed_portfolio_pd * 100.0,

            "base_expected_loss": base_expected_loss,
            "stressed_expected_loss": stressed_expected_loss,
            "expected_loss_change": expected_loss_change,
            "expected_loss_multiple": expected_loss_multiple,

            "scenario_likelihood": scenario_likelihood
        },
        "top_industries": _top_industries(df),
        "top_obligors": _top_obligors(df, top_n=100),
        "download": {
            "available": False,
            "requires_registration": True,
            "endpoint": "/download_scenario",
        },
        "model_info": {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "calculation_mode": "deterministic_vasicek",
            "factors": FACTOR_NAMES,
            "portfolio_path": PORTFOLIO_PATH,
            "loadings_path": LOADINGS_PATH,
            "pd_column": df.attrs.get("pd_column"),
            "ead_column": df.attrs.get("ead_column"),
            "pd_is_obligor_level": True,
            "ead_is_obligor_level": True,
            "loading_columns": FACTOR_COLUMNS,
            "simulation_enabled": False,
        },

        # Flat aliases for simpler Bubble/API bindings
        "market": float(market),
        "technology": float(technology),
        "commodity": float(commodity),
        "portfolio_pd": stressed_portfolio_pd,
        "portfolio_pd_percent": stressed_portfolio_pd * 100.0,
        "expected_loss": stressed_expected_loss,
    }


    return response
