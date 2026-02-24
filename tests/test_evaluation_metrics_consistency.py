from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluation


def _ref_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2


def _ref_log_score(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _ref_score_table(df: pd.DataFrame, *, p_col: str = "p_hat", y_col: str = "y") -> dict:
    d = df.copy()
    y = d[y_col].to_numpy(dtype=float)
    d = d[np.isfinite(y)]
    if d.empty:
        return {"n": 0, "mean_log_score": np.nan, "mean_brier": np.nan}
    p = d[p_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    return {
        "n": int(len(d)),
        "mean_log_score": float(np.mean(_ref_log_score(p, y))),
        "mean_brier": float(np.mean(_ref_brier(p, y))),
    }


def _ref_reliability_by_group(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
    p_col: str = "p_hat",
    y_col: str = "y",
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")

    d = df.copy()
    y = d[y_col].to_numpy(dtype=float)
    d = d[np.isfinite(y)]

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

    rows: list[dict] = []
    for keys, sub in d.groupby(list(group_cols), sort=True):
        if isinstance(keys, tuple):
            key_tuple = keys
        else:
            key_tuple = (keys,)
        key_map = {col: val for col, val in zip(group_cols, key_tuple)}

        p = sub[p_col].to_numpy(dtype=float)
        yy = sub[y_col].to_numpy(dtype=float)
        for b in range(int(n_bins)):
            lo = float(edges[b])
            hi = float(edges[b + 1])
            if b < int(n_bins) - 1:
                mask = (p >= lo) & (p < hi)
            else:
                mask = (p >= lo) & (p <= hi)  # include p==1.0 in the last bin
            if not np.any(mask):
                rows.append({**key_map, "bin_lo": lo, "bin_hi": hi, "count": 0, "p_mean": np.nan, "y_mean": np.nan})
            else:
                rows.append(
                    {
                        **key_map,
                        "bin_lo": lo,
                        "bin_hi": hi,
                        "count": int(np.sum(mask)),
                        "p_mean": float(np.mean(p[mask])),
                        "y_mean": float(np.mean(yy[mask])),
                    }
                )

    out = pd.DataFrame(rows)
    sort_cols = list(group_cols) + ["bin_lo", "bin_hi"]
    if not out.empty:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def test_scoring_functions_match_reference() -> None:
    p = np.array([0.0, 0.2, 0.8, 1.0, 0.3333333333], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=float)

    got_brier = evaluation.brier_score(p, y)
    got_log = evaluation.log_score(p, y, eps=1e-12)

    exp_brier = _ref_brier(p, y)
    exp_log = _ref_log_score(p, y, eps=1e-12)

    assert np.allclose(got_brier, exp_brier, rtol=0.0, atol=0.0)
    assert np.allclose(got_log, exp_log, rtol=0.0, atol=0.0)


def test_run_evaluation_script_outputs_are_numerically_consistent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # State series: D is the realised drawdown on each date.
    # For Tdays=2, labels are defined for the first 3 dates only; last 2 are NaN.
    state = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2008-09-09", "2008-09-10", "2008-09-11", "2008-09-12", "2008-09-15"]),
            "D": [0.10, 0.25, 0.20, 0.40, 0.05],
        }
    )
    state_csv = tmp_path / "state.csv"
    state.to_csv(state_csv, index=False)

    # Forecast table: include edge-case probabilities (0 and 1) to exercise eps-clipping in log score.
    forecasts = pd.DataFrame(
        [
            {"Date": "2008-09-09", "Tdays": 2, "delta": 0.20, "p_hat": 0.0},
            {"Date": "2008-09-10", "Tdays": 2, "delta": 0.20, "p_hat": 1.0},
            {"Date": "2008-09-11", "Tdays": 2, "delta": 0.20, "p_hat": 0.5},
            {"Date": "2008-09-12", "Tdays": 2, "delta": 0.20, "p_hat": 0.9},
            {"Date": "2008-09-15", "Tdays": 2, "delta": 0.20, "p_hat": 0.1},
            {"Date": "2008-09-09", "Tdays": 2, "delta": 0.30, "p_hat": 0.2},
            {"Date": "2008-09-10", "Tdays": 2, "delta": 0.30, "p_hat": 0.3},
            {"Date": "2008-09-11", "Tdays": 2, "delta": 0.30, "p_hat": 0.4},
            {"Date": "2008-09-12", "Tdays": 2, "delta": 0.30, "p_hat": 0.5},
            {"Date": "2008-09-15", "Tdays": 2, "delta": 0.30, "p_hat": 0.6},
        ]
    )
    forecasts_csv = tmp_path / "forecasts.csv"
    forecasts.to_csv(forecasts_csv, index=False)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "04_run_evaluation.py"),
        "--forecasts_csv",
        str(forecasts_csv),
        "--state_csv",
        str(state_csv),
        "--out_dir",
        str(out_dir),
        "--reliability_bins",
        "4",
    ]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    assert res.returncode == 0, f"script failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    meta_files = list(out_dir.glob("eval_*.json"))
    joined_files = list(out_dir.glob("eval_joined_*.csv"))
    rel_files = list(out_dir.glob("reliability_*.csv"))
    assert len(meta_files) == 1
    assert len(joined_files) == 1
    assert len(rel_files) == 1

    meta_path = meta_files[0]
    joined_path = joined_files[0]
    rel_path = rel_files[0]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    joined = pd.read_csv(joined_path)
    rel = pd.read_csv(rel_path)

    # --- Check meta['scores'] matches a fresh recomputation from eval_joined CSV (reference math) ---
    ref_scores = _ref_score_table(joined, p_col="p_hat", y_col="y")
    got_scores = meta["scores"]
    assert got_scores["n"] == ref_scores["n"]
    assert np.isclose(got_scores["mean_log_score"], ref_scores["mean_log_score"], rtol=0.0, atol=1e-12)
    assert np.isclose(got_scores["mean_brier"], ref_scores["mean_brier"], rtol=0.0, atol=1e-12)

    # --- Check group scores are consistent ---
    scores_by = pd.DataFrame(meta["scores_by_query"]).sort_values(["Tdays", "delta"]).reset_index(drop=True)
    rows: list[dict] = []
    for (T, delta), sub in joined.groupby(["Tdays", "delta"], sort=True):
        s = _ref_score_table(sub, p_col="p_hat", y_col="y")
        rows.append({"Tdays": int(T), "delta": float(delta), **s})
    ref_by = pd.DataFrame(rows).sort_values(["Tdays", "delta"]).reset_index(drop=True)

    assert scores_by["n"].to_list() == ref_by["n"].to_list()
    assert np.allclose(scores_by["mean_log_score"].to_numpy(float), ref_by["mean_log_score"].to_numpy(float), rtol=0.0, atol=1e-12)
    assert np.allclose(scores_by["mean_brier"].to_numpy(float), ref_by["mean_brier"].to_numpy(float), rtol=0.0, atol=1e-12)

    # --- Check reliability bin table matches reference computation ---
    ref_rel = _ref_reliability_by_group(joined, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y", n_bins=4)
    rel_sorted = rel.sort_values(["Tdays", "delta", "bin_lo", "bin_hi"]).reset_index(drop=True)

    # Ensure same shape and bins.
    assert len(rel_sorted) == len(ref_rel)
    assert np.allclose(rel_sorted["bin_lo"].to_numpy(float), ref_rel["bin_lo"].to_numpy(float), rtol=0.0, atol=0.0)
    assert np.allclose(rel_sorted["bin_hi"].to_numpy(float), ref_rel["bin_hi"].to_numpy(float), rtol=0.0, atol=0.0)
    assert rel_sorted[["Tdays", "delta"]].to_numpy().tolist() == ref_rel[["Tdays", "delta"]].to_numpy().tolist()

    assert rel_sorted["count"].to_list() == ref_rel["count"].to_list()

    # p_mean / y_mean can be NaN for empty bins; compare with NaN-safe logic.
    for col in ("p_mean", "y_mean"):
        got = rel_sorted[col].to_numpy(dtype=float)
        exp = ref_rel[col].to_numpy(dtype=float)
        nan_mask = np.isnan(got) & np.isnan(exp)
        assert np.allclose(got[~nan_mask], exp[~nan_mask], rtol=0.0, atol=1e-12)
