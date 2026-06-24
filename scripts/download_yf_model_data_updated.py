"""Download Yahoo Finance fundamentals and 5-year equity history for rating models.

Outputs
-------
market_data/yf_fundamentals_raw.parquet
market_data/yf_fundamentals_clean.parquet
market_data/yf_fundamentals_failed.csv
market_data/adj_close_5y.parquet
market_data/log_returns_5y.parquet
market_data/equity_vol_metrics_5y.parquet
market_data/equity_price_failed.csv

Notes
-----
- Treat Yahoo Finance as an offline ingestion source, not a production dependency.
- Script is resume-safe for fundamentals and saves price data in chunks.
- Company descriptions are captured from Yahoo's longBusinessSummary field.
"""

import time
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
INPUT_PATH = "market_data/yf_company_info.parquet"
MARKET_DATA_DIR = Path("market_data")

OUT_RAW = MARKET_DATA_DIR / "yf_fundamentals_raw.parquet"
OUT_CLEAN = MARKET_DATA_DIR / "yf_fundamentals_clean.parquet"
OUT_FAILED = MARKET_DATA_DIR / "yf_fundamentals_failed.csv"

OUT_ADJ_CLOSE = MARKET_DATA_DIR / "adj_close_5y.parquet"
OUT_LOG_RETURNS = MARKET_DATA_DIR / "log_returns_5y.parquet"
OUT_VOL_METRICS = MARKET_DATA_DIR / "equity_vol_metrics_5y.parquet"
OUT_PRICE_FAILED = MARKET_DATA_DIR / "equity_price_failed.csv"


# -----------------------------------------------------------------------------
# Yahoo fundamental fields
# -----------------------------------------------------------------------------
FIELDS = [
    "symbol",
    "shortName",
    "longName",
    "longBusinessSummary",  # Yahoo company description
    "sector",
    "industry",
    "country",
    "exchange",
    "quoteType",
    "currency",
    "financialCurrency",

    "marketCap",
    "enterpriseValue",
    "totalDebt",
    "totalCash",
    "ebitda",
    "totalRevenue",
    "netIncomeToCommon",

    "debtToEquity",
    "currentRatio",
    "quickRatio",

    "grossMargins",
    "ebitdaMargins",
    "operatingMargins",
    "profitMargins",
    "returnOnAssets",
    "returnOnEquity",

    "beta",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToRevenue",
    "enterpriseToEbitda",

    "fullTimeEmployees",
]

NUMERIC_FIELDS = [
    "marketCap",
    "enterpriseValue",
    "totalDebt",
    "totalCash",
    "ebitda",
    "totalRevenue",
    "netIncomeToCommon",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "grossMargins",
    "ebitdaMargins",
    "operatingMargins",
    "profitMargins",
    "returnOnAssets",
    "returnOnEquity",
    "beta",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToRevenue",
    "enterpriseToEbitda",
    "fullTimeEmployees",
]

TEXT_FIELDS = [
    "symbol",
    "shortName",
    "longName",
    "longBusinessSummary",
    "sector",
    "industry",
    "country",
    "exchange",
    "quoteType",
    "currency",
    "financialCurrency",
    "download_status",
    "error",
]

BAD_VALUES = {
    "Infinity": np.nan,
    "-Infinity": np.nan,
    "inf": np.nan,
    "-inf": np.nan,
    "NaN": np.nan,
    "nan": np.nan,
    "N/A": np.nan,
    "None": np.nan,
    "": np.nan,
}


def normalise_symbols(symbols: Iterable[str]) -> List[str]:
    """Uppercase, trim, de-duplicate and drop null ticker symbols."""
    clean_symbols = [str(s).strip().upper() for s in symbols if pd.notna(s)]
    return list(dict.fromkeys(clean_symbols))


def chunked(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield items in fixed-size chunks."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


# -----------------------------------------------------------------------------
# Fundamentals
# -----------------------------------------------------------------------------
def clean_yahoo_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(BAD_VALUES)

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TEXT_FIELDS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def fetch_one_fundamental(symbol: str) -> dict:
    symbol = str(symbol).strip().upper()

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        row = {field: info.get(field) for field in FIELDS}
        row["symbol"] = symbol
        row["download_status"] = "ok"
        row["error"] = None
        return row

    except Exception as e:
        return {
            "symbol": symbol,
            "download_status": "failed",
            "error": str(e),
        }


def fetch_fundamentals(
    symbols: Iterable[str],
    sleep_seconds: float = 2.0,
    save_every: int = 10,
    max_retries: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch ticker.info fundamentals with resume-safe checkpoints."""
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_RAW.exists():
        existing = pd.read_parquet(OUT_RAW)
        existing_symbols = set(existing["symbol"].astype(str).str.upper())
        rows = existing.to_dict("records")
        print(f"Resuming fundamentals from {len(existing_symbols)} existing symbols")
    else:
        existing_symbols = set()
        rows = []

    symbols = normalise_symbols(symbols)
    remaining = [s for s in symbols if s not in existing_symbols]
    print(f"Remaining fundamental symbols: {len(remaining)}")

    for i, symbol in enumerate(remaining, start=1):
        print(f"{i}/{len(remaining)} fundamentals {symbol}")

        row = None
        for attempt in range(max_retries + 1):
            row = fetch_one_fundamental(symbol)
            if row.get("download_status") == "ok":
                break

            wait = sleep_seconds * (attempt + 1) + random.uniform(0, 1)
            print(f"  failed attempt {attempt + 1}; sleeping {wait:.1f}s")
            time.sleep(wait)

        rows.append(row)

        if i % save_every == 0:
            partial = clean_yahoo_df(pd.DataFrame(rows))
            partial.to_parquet(OUT_RAW, index=False)
            print(f"  saved fundamentals checkpoint: {len(partial)} rows")
            time.sleep(sleep_seconds + random.uniform(0, 1))

    raw = clean_yahoo_df(pd.DataFrame(rows))
    raw.to_parquet(OUT_RAW, index=False)

    clean = raw[raw["download_status"] == "ok"].copy()
    clean.to_parquet(OUT_CLEAN, index=False)

    failed = raw[raw["download_status"] != "ok"].copy()
    failed.to_csv(OUT_FAILED, index=False)

    print("Fundamentals done.")
    print(f"Raw rows:    {len(raw)}")
    print(f"Clean rows:  {len(clean)}")
    print(f"Failed rows: {len(failed)}")

    return clean, failed


# -----------------------------------------------------------------------------
# Equity history
# -----------------------------------------------------------------------------
def _extract_adjusted_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Return an adjusted-close style price matrix from yfinance.download output."""
    if downloaded.empty:
        return pd.DataFrame()

    # yfinance with multiple tickers usually returns a MultiIndex column structure.
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        if "Close" in level0:
            close = downloaded["Close"].copy()
        elif "Adj Close" in level0:
            close = downloaded["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output")
    else:
        # Single ticker case.
        if "Close" in downloaded.columns:
            close = downloaded[["Close"]].copy()
        elif "Adj Close" in downloaded.columns:
            close = downloaded[["Adj Close"]].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output")

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.dropna(axis=1, how="all")
    return close


def download_price_chunk(
    symbols: List[str],
    period: str = "5y",
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download one chunk of price data."""
    if not symbols:
        return pd.DataFrame()

    data = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    close = _extract_adjusted_close(data)

    # In the single ticker case, rename Close to ticker.
    if len(symbols) == 1 and list(close.columns) == ["Close"]:
        close.columns = symbols

    # yfinance sometimes returns original case; normalise to uppercase strings.
    close.columns = [str(c).strip().upper() for c in close.columns]

    time.sleep(sleep_seconds + random.uniform(0, 1))
    return close


def fetch_equity_history(
    symbols: Iterable[str],
    period: str = "5y",
    chunk_size: int = 50,
    sleep_seconds: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download 5-year adjusted close and log returns in chunks."""
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    symbols = normalise_symbols(symbols)
    all_closes = []

    print(f"Downloading equity history for {len(symbols)} symbols")

    for i, batch in enumerate(chunked(symbols, chunk_size), start=1):
        print(f"Price chunk {i}: {len(batch)} symbols")
        try:
            close = download_price_chunk(
                batch,
                period=period,
                sleep_seconds=sleep_seconds,
            )
            if not close.empty:
                all_closes.append(close)

                interim = pd.concat(all_closes, axis=1)
                interim = interim.loc[:, ~interim.columns.duplicated()]
                interim.to_parquet(OUT_ADJ_CLOSE)
                print(f"  saved price checkpoint: {interim.shape[1]} columns")
        except Exception as e:
            print(f"  price chunk failed: {e}")

    if all_closes:
        adj_close = pd.concat(all_closes, axis=1)
        adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]
        adj_close = adj_close.sort_index()
    else:
        adj_close = pd.DataFrame()

    # Track tickers with no usable price history.
    downloaded_symbols = set(adj_close.columns.astype(str).str.upper()) if not adj_close.empty else set()
    failed_symbols = [s for s in symbols if s not in downloaded_symbols]
    failed = pd.DataFrame({"symbol": failed_symbols})
    failed.to_csv(OUT_PRICE_FAILED, index=False)

    adj_close.to_parquet(OUT_ADJ_CLOSE)

    log_returns = np.log(adj_close / adj_close.shift(1)).replace([np.inf, -np.inf], np.nan)
    log_returns.to_parquet(OUT_LOG_RETURNS)

    vol_metrics = make_vol_metrics(log_returns)
    vol_metrics.to_parquet(OUT_VOL_METRICS, index=False)

    print("Equity history done.")
    print(f"Adj close shape:   {adj_close.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Failed prices:     {len(failed)}")

    return adj_close, log_returns, vol_metrics


def make_vol_metrics(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Create basic annualised volatility metrics from daily log returns."""
    if log_returns.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "daily_obs",
            "annual_vol_5y",
            "annual_vol_3y",
            "annual_vol_1y",
        ])

    ann = np.sqrt(252)

    metrics = pd.DataFrame({
        "symbol": log_returns.columns.astype(str),
        "daily_obs": log_returns.notna().sum().values,
        "annual_vol_5y": (log_returns.std(skipna=True) * ann).values,
        "annual_vol_3y": (log_returns.tail(252 * 3).std(skipna=True) * ann).values,
        "annual_vol_1y": (log_returns.tail(252).std(skipna=True) * ann).values,
    })

    return metrics


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def main():
    universe = pd.read_parquet(INPUT_PATH)

    # Change this if your ticker column has a different name.
    symbols = universe["symbol"].dropna().unique().tolist()

    clean, failed = fetch_fundamentals(
        symbols,
        sleep_seconds=2.0,
        save_every=10,
        max_retries=2,
    )

    # Use all requested symbols, not only clean fundamentals, because a ticker can
    # fail fundamentals but still have useful price history.
    adj_close, log_returns, vol_metrics = fetch_equity_history(
        symbols,
        period="5y",
        chunk_size=50,
        sleep_seconds=2.0,
    )

    print("All downloads complete.")


if __name__ == "__main__":
    main()
