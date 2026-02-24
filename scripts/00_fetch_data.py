from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass
class DataProvenance:
    ticker: str
    provider: str
    access_method: str
    yfinance_version: str
    retrieved_at_utc: str
    start_date: str
    end_date: str
    interval: str
    fields_expected: list[str]


def fetch_spy_08_crash_dataset(
    out_dir: str | Path = "data",
    start: str = "2002-01-01",  # includes pre-2007 training history
    end: str = "2010-12-31",    # includes crisis + aftermath
    ticker: str = "SPY",
    interval: str = "1d",
) -> tuple[pd.DataFrame, DataProvenance]:
    """
    Fetch SPY daily data from Yahoo Finance using yfinance and save it locally
    with metadata for academic reproducibility.
    """
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Download daily bars. Keep both Close and Adj Close; use Adj Close for returns in most finance workflows.
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # keeps "Adj Close" column
        actions=True,        # adds Dividends and Stock Splits columns
        progress=False,
        group_by="column",
        threads=True,
    )

    # yfinance may return MultiIndex columns, especially when it internally treats the request
    # as a multi-ticker download. This function is single-ticker by design, so normalize to a
    # plain 1-level column index for downstream code.
    if isinstance(df.columns, pd.MultiIndex):
        # Common shape: level 0 = OHLCV field, level 1 = ticker
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, level=-1, axis=1)
        else:
            # Fallback: drop the last level if it is length-1 or otherwise unhelpful.
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = [" ".join(map(str, col)).strip() for col in df.columns]

    if df.empty:
        raise RuntimeError("No data returned. Try again later or check ticker/date range.")

    # Standardize columns & index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Expected columns (yfinance typically returns these for ETFs/stocks)
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        # Not fatal, but good to know for your paper/pipeline
        print(f"Warning: missing columns: {missing}")

    # Add analysis-friendly series (optional but commonly needed)
    # Use adjusted close for log-price and log-returns
    import numpy as np

    adj_close = pd.to_numeric(df.get("Adj Close"), errors="coerce")
    if adj_close is None:
        raise RuntimeError("Expected 'Adj Close' column not found; cannot compute log-returns.")

    # Avoid log(0) -> -inf; treat non-positive prices as missing for log-based analysis.
    adj_close = adj_close.where(adj_close > 0)
    df["LogAdjClose"] = np.log(adj_close)
    df["LogReturn"] = df["LogAdjClose"].diff()

    # Save frozen dataset (so your results are reproducible)
    stamp = retrieved_at_utc.replace(":", "").replace("-", "")
    csv_path = raw_dir / f"{ticker}_yahoo_{start}_to_{end}_{stamp}.csv"
    meta_path = raw_dir / f"{ticker}_yahoo_{start}_to_{end}_{stamp}_metadata.json"

    df.reset_index(names="Date").to_csv(csv_path, index=False)

    prov = DataProvenance(
        ticker=ticker,
        provider="Yahoo Finance",
        access_method="Downloaded via yfinance (Python)",
        yfinance_version=getattr(yf, "__version__", "unknown"),
        retrieved_at_utc=retrieved_at_utc,
        start_date=start,
        end_date=end,
        interval=interval,
        fields_expected=expected,
    )
    meta_path.write_text(json.dumps(asdict(prov), indent=2), encoding="utf-8")

    return df, prov


if __name__ == "__main__":
    df, prov = fetch_spy_08_crash_dataset()
    print(df.head())
    print("\nSaved with provenance:\n", prov)