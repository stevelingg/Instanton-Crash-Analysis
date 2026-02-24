from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_state import StateBuildSpec, build_state_frame
from src.io import find_latest_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def _read_csv_date_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise KeyError(f"'Date' column not found in {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df


def _load_latest_state_and_meta() -> tuple[Path, pd.DataFrame, dict]:
    state_path = find_latest_file(DATA_PROCESSED, "state_*.csv")
    state_df = _read_csv_date_index(state_path)

    meta_path = state_path.with_name(state_path.stem + "_metadata.json")
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return state_path, state_df, meta


def _resolve_raw_path(meta: dict) -> Path:
    # Prefer the raw_source_csv recorded by the build script, if it exists on this machine.
    raw_source = meta.get("raw_source_csv")
    if raw_source:
        p = Path(raw_source)
        if p.exists():
            return p

    # Fallback: latest SPY raw CSV in data/raw
    return find_latest_file(DATA_RAW, "SPY_yahoo_*.csv")


def _build_spec_from_meta(meta: dict) -> StateBuildSpec:
    # Use metadata if available; otherwise fall back to defaults.
    bs = meta.get("build_spec", {}) if meta else {}
    return StateBuildSpec(
        price_col=bs.get("price_col", "Adj Close"),
        rv_window_days=int(bs.get("rv_window_days", 20)),
        annualisation=int(bs.get("annualisation", 252)),
        min_periods=bs.get("min_periods", None),
    )


def _assert_series_close(a: pd.Series, b: pd.Series, name: str, atol: float = 1e-12, rtol: float = 1e-12) -> None:
    if not a.index.equals(b.index):
        raise AssertionError(f"Index mismatch for {name}")
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    if not np.allclose(av, bv, atol=atol, rtol=rtol, equal_nan=True):
        diff = np.nanmax(np.abs(av - bv))
        raise AssertionError(f"{name} mismatch: max_abs_diff={diff}")


def test_state_matches_recompute_from_raw() -> None:
    _, state_df, meta = _load_latest_state_and_meta()
    raw_path = _resolve_raw_path(meta)
    raw_df = _read_csv_date_index(raw_path)

    spec = _build_spec_from_meta(meta)
    recomputed, _ = build_state_frame(raw_df.reset_index(), spec=spec)
    # build_state_frame returns index=Date already if Date column present;
    # but ensure matching index:
    recomputed.index = pd.to_datetime(recomputed.index)

    # Compare on common date range (just in case of minor mismatched endpoints)
    common = state_df.index.intersection(recomputed.index)
    assert len(common) > 0, "No overlapping dates between processed state and recomputed state."

    for col in ["P", "X", "r", "M", "D", "delta", "V_proxy"]:
        assert col in state_df.columns, f"Missing {col} in processed state CSV"
        assert col in recomputed.columns, f"Missing {col} in recomputed state"
        _assert_series_close(state_df.loc[common, col], recomputed.loc[common, col], name=col)


def test_drawdown_identities_hold() -> None:
    _, state_df, _ = _load_latest_state_and_meta()

    # Identity 1: D = M - X
    _assert_series_close(state_df["D"], state_df["M"] - state_df["X"], name="D == M - X")

    # Identity 2: delta = 1 - exp(-D)
    delta_from_D = 1.0 - np.exp(-state_df["D"].to_numpy(dtype=float))
    delta_from_D = pd.Series(delta_from_D, index=state_df.index, name="delta_from_D")
    _assert_series_close(state_df["delta"], delta_from_D, name="delta == 1 - exp(-D)")

    # Identity 3: delta = 1 - P / running_max(P)
    P = state_df["P"].to_numpy(dtype=float)
    Pmax = np.maximum.accumulate(P)
    delta_from_prices = 1.0 - (P / Pmax)
    delta_from_prices = pd.Series(delta_from_prices, index=state_df.index, name="delta_from_prices")
    _assert_series_close(state_df["delta"], delta_from_prices, name="delta == 1 - P/Pmax", atol=1e-10, rtol=1e-10)


def test_return_definition_holds() -> None:
    _, state_df, _ = _load_latest_state_and_meta()
    r_from_X = state_df["X"].diff()
    _assert_series_close(state_df["r"], r_from_X, name="r == diff(X)")


def test_variance_proxy_definition_holds() -> None:
    _, state_df, meta = _load_latest_state_and_meta()
    spec = _build_spec_from_meta(meta)

    r = state_df["r"].astype(float)
    rv = spec.annualisation * (r.pow(2)).rolling(window=spec.rv_window_days, min_periods=spec.min_periods or spec.rv_window_days).mean()
    _assert_series_close(state_df["V_proxy"], rv, name="V_proxy rolling RV")