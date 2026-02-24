from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io import ensure_dir, find_latest_file
from src.data_state import StateBuildSpec, build_state_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--pattern", type=str, default=None, help="Override raw filename glob.")
    ap.add_argument("--price_col", type=str, default="Adj Close")
    ap.add_argument("--rv_window_days", type=int, default=20)
    ap.add_argument("--min_periods", type=int, default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = ensure_dir(args.processed_dir)

    pattern = args.pattern or f"{args.ticker}_yahoo_*.csv"
    raw_path = find_latest_file(raw_dir, pattern)

    raw = pd.read_csv(raw_path)
    spec = StateBuildSpec(
        price_col=args.price_col,
        rv_window_days=args.rv_window_days,
        min_periods=args.min_periods,
    )
    state_df, meta = build_state_frame(raw, spec=spec)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "").replace("-", "")
    out_csv = processed_dir / f"state_{args.ticker}_{stamp}.csv"
    out_meta = processed_dir / f"state_{args.ticker}_{stamp}_metadata.json"

    state_df.reset_index(names="Date").to_csv(out_csv, index=False)

    meta_full = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_source_csv": str(raw_path.resolve()),
        "rows": int(state_df.shape[0]),
        "cols": list(state_df.columns),
        "build_spec": {
            "price_col": spec.price_col,
            "rv_window_days": spec.rv_window_days,
            "annualisation": spec.annualisation,
            "min_periods": spec.min_periods if spec.min_periods is not None else spec.rv_window_days,
        },
        "meta": meta,
    }
    out_meta.write_text(json.dumps(meta_full, indent=2), encoding="utf-8")

    print(f"Wrote:\n  {out_csv}\n  {out_meta}")


if __name__ == "__main__":
    main()