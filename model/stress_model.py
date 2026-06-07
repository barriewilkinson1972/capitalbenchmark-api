import pandas as pd
import numpy as np
from scipy.stats import norm


PORTFOLIO_PATH = "market_data/portfolio_metadata.parquet"
LOADINGS_PATH = "data/industry_factor_loadings.parquet"


def run_stress(oil=0.0, ai=0.0, pd_base=0.01, lgd=0.45, ead=1.0):
    portfolio = pd.read_parquet(PORTFOLIO_PATH)
    loadings = pd.read_parquet(LOADINGS_PATH)

    df = portfolio.copy()

    df["base_pd"] = pd_base
    df["lgd"] = lgd
    df["ead"] = ead

    df["oil_corr"] = df["Industry"].map(loadings["oil_corr"]).fillna(0.0)
    df["ai_corr"] = df["Industry"].map(loadings["ai_corr"]).fillna(0.0)

    shift = df["oil_corr"] * oil + df["ai_corr"] * ai

    r2 = (df["oil_corr"] ** 2 + df["ai_corr"] ** 2).clip(0.0, 0.999)
    residual_vol = np.sqrt(1.0 - r2)

    threshold = norm.ppf(df["base_pd"])

    df["stressed_pd"] = norm.cdf(
        (threshold - shift) / residual_vol
    )

    df["expected_loss"] = df["stressed_pd"] * df["lgd"] * df["ead"]

    portfolio_pd = df["stressed_pd"].mean()
    expected_loss = df["expected_loss"].sum()

    industry = (
        df.groupby("Industry")
        .agg(
            base_pd=("base_pd", "mean"),
            stressed_pd=("stressed_pd", "mean"),
            expected_loss=("expected_loss", "sum"),
            obligors=("Symbol", "count"),
        )
        .reset_index()
    )

    industry["change"] = industry["stressed_pd"] - industry["base_pd"]

    top_industries = (
        industry.sort_values("change", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return {
        "oil": oil,
        "ai": ai,
        "portfolio_pd": float(portfolio_pd),
        "portfolio_pd_percent": float(portfolio_pd * 100),
        "expected_loss": float(expected_loss),
        "top_industries": top_industries,
    }