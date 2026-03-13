from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.fit import fit_sv_params_qmle_from_state, params_to_jsonable
from src.io import ensure_dir, resolve_input_file, file_fingerprint


def read_state_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise KeyError(f"'Date' column missing in {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--state_csv", type=str, default=None, help="Explicit state CSV path. Preferred for reproducibility.")
    ap.add_argument("--allow_latest_state", action="store_true", help="Allow newest matching state file when pattern is ambiguous.")
    ap.add_argument("--outputs_dir", type=str, default="outputs/params")
    ap.add_argument("--pattern", type=str, default="state_*.csv")
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--train_start", type=str, default=None)
    ap.add_argument("--train_end", type=str, default="2006-12-31")
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument(
        "--allow_failed_fit",
        action="store_true",
        help="Write parameters even if optimisation reports failure. Not recommended for publishable runs.",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = ensure_dir(args.outputs_dir)

    state_path = resolve_input_file(
        explicit_path=args.state_csv,
        directory=processed_dir,
        pattern=args.pattern,
        allow_latest=bool(args.allow_latest_state),
        purpose="state input",
    )
    state_df = read_state_csv(state_path)

    params, diag = fit_sv_params_qmle_from_state(
        state_df=state_df,
        train_start=args.train_start,
        train_end=args.train_end,
        price_ticker=args.ticker,
        maxiter=args.maxiter,
        require_success=not bool(args.allow_failed_fit),
    )

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "").replace("-", "")
    out_path = out_dir / f"sv_params_{args.ticker}_{stamp}.json"

    payload = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state_source_csv": str(state_path.resolve()),
        "state_source_fingerprint": file_fingerprint(state_path),
        **params_to_jsonable(params, diag),
        "cli_args": vars(args),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote:\n  {out_path}")
    print("\nParams:\n", payload["params"])
    print("\nDiagnostics:\n", payload["diagnostics"]["moments"])
    print("\nOptimiser:\n", payload["diagnostics"]["optimiser"])


if __name__ == "__main__":
    main()
