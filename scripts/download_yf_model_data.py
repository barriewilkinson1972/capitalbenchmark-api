import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


INPUT_PATH = "market_data/model_universe.csv"
OUT_RAW = Path("market_data/yf_fundamentals_raw.parquet")
OUT_CLEAN = Path("market_data/yf_fundamentals_clean.parquet")
OUT_FAILED = Path("market_data/yf_fundamentals_failed.csv")


FIELDS = [
    "symbol",
    "shortName",
    "longName",
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


def clean_yahoo_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace({
        "Infinity": np.nan,
        "-Infinity": np.nan,
        "inf": np.nan,
        "-inf": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "N/A": np.nan,
        "None": np.nan,
        "": np.nan,
    })

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TEXT_FIELDS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def fetch_one(symbol: str) -> dict:
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
    symbols,
    sleep_seconds: float = 2.0,
    save_every: int = 10,
    max_retries: int = 2,
):
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    if OUT_RAW.exists():
        existing = pd.read_parquet(OUT_RAW)
        existing_symbols = set(existing["symbol"].astype(str).str.upper())
        rows = existing.to_dict("records")
        print(f"Resuming from {len(existing_symbols)} existing symbols")
    else:
        existing_symbols = set()
        rows = []

    symbols = [str(s).strip().upper() for s in symbols if pd.notna(s)]
    symbols = list(dict.fromkeys(symbols))

    remaining = [s for s in symbols if s not in existing_symbols]
    print(f"Remaining symbols: {len(remaining)}")

    for i, symbol in enumerate(remaining, start=1):
        print(f"{i}/{len(remaining)} downloading {symbol}")

        row = None

        for attempt in range(max_retries + 1):
            row = fetch_one(symbol)

            if row.get("download_status") == "ok":
                break

            wait = sleep_seconds * (attempt + 1) + random.uniform(0, 1)
            print(f"  failed attempt {attempt + 1}; sleeping {wait:.1f}s")
            time.sleep(wait)

        rows.append(row)

        if i % save_every == 0:
            partial = clean_yahoo_df(pd.DataFrame(rows))
            partial.to_parquet(OUT_RAW, index=False)
            print(f"  saved checkpoint: {len(partial)} rows")
            time.sleep(sleep_seconds + random.uniform(0, 1))

    raw = clean_yahoo_df(pd.DataFrame(rows))
    raw.to_parquet(OUT_RAW, index=False)

    clean = raw[raw["download_status"] == "ok"].copy()
    clean.to_parquet(OUT_CLEAN, index=False)

    failed = raw[raw["download_status"] != "ok"].copy()
    failed.to_csv(OUT_FAILED, index=False)

    print("Done.")
    print(f"Raw rows:    {len(raw)}")
    print(f"Clean rows:  {len(clean)}")
    print(f"Failed rows: {len(failed)}")

    return clean, failed


if __name__ == "__main__":
    universe = pd.read_csv(INPUT_PATH, encoding = "latin1")

    # Change this if your ticker column has a different name.
    symbols = universe["symbol"].dropna().unique().tolist()

    clean, failed = fetch_fundamentals(
        symbols,
        sleep_seconds=2.0,
        save_every=10,
        max_retries=2,
    )