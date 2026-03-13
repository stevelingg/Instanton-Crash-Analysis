from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    end_date_inclusive_requested: str
    provider_end_exclusive_used: str
    interval: str
    fields_expected: list[str]


def fetch_spy_08_crash_dataset(
    out_dir: str | Path = "data",
    start: str = "2002-01-01",
    end: str = "2010-12-31",
    ticker: str = "SPY",
    interval: str = "1d",
) -> tuple[pd.DataFrame, DataProvenance]:
    """
    Fetch SPY daily data from Yahoo Finance using yfinance and save it locally
    with metadata for academic reproducibility.

    Important:
      yfinance/Yahoo daily downloads interpret `end` as exclusive.
      This function treats the user-facing `end` as INCLUSIVE and converts it
      to the provider's exclusive end date internally.
    """
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end={end!r} must be >= start={start!r}")

    # Yahoo daily API uses exclusive end; convert requested inclusive end -> exclusive.
    provider_end_exclusive = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        tickers=ticker,
        start=start_ts.strftime("%Y-%m-%d"),
        end=provider_end_exclusive,
        interval=interval,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, level=-1, axis=1)
        else:
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = [" ".join(map(str, col)).strip() for col in df.columns]

    if df.empty:
        raise RuntimeError("No data returned. Try again later or check ticker/date range.")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Hard trim to the user-requested inclusive window.
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    if df.empty:
        raise RuntimeError(
            "Downloaded data became empty after trimming to the requested inclusive date window."
        )

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"Warning: missing columns: {missing}")

    if "Adj Close" not in df.columns:
        raise RuntimeError(
            "Expected 'Adj Close' column not found in downloaded data. "
            "This pipeline defines log-prices and returns from adjusted close for reproducibility."
        )

    adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
    if adj_close.isna().all():
        raise RuntimeError("'Adj Close' exists but could not be converted to numeric values.")

    adj_close = adj_close.where(adj_close > 0)
    if adj_close.dropna().empty:
        raise RuntimeError("Adjusted close has no positive numeric observations.")

    df["LogAdjClose"] = np.log(adj_close)
    df["LogReturn"] = df["LogAdjClose"].diff()

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
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date_inclusive_requested=end_ts.strftime("%Y-%m-%d"),
        provider_end_exclusive_used=provider_end_exclusive,
        interval=interval,
        fields_expected=expected,
    )
    meta_path.write_text(json.dumps(asdict(prov), indent=2), encoding="utf-8")

    return df, prov


if __name__ == "__main__":
    df, prov = fetch_spy_08_crash_dataset()
    print(df.head())
    print("\nSaved with provenance:\n", prov)