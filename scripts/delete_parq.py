from pathlib import Path

for f in [
    "market_data/yf_fundamentals_raw.parquet",
    "market_data/yf_fundamentals_clean.parquet",
    "market_data/yf_fundamentals_failed.csv",
]:
    p = Path(f)
    if p.exists():
        p.unlink()
        print("Deleted", f)