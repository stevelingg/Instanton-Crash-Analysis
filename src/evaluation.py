from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .constants import log_drawdown_threshold
from .fit import SVParams
from .importance_sampling import simulate_is_paths, estimate_probability_is, ISResult


@dataclass(frozen=True)
class NaiveMCResult:
    """
    Naive Monte Carlo estimate under the baseline law P (no IS drift shift).
    This is used as a sanity baseline (PDF Section 6): where feasible, check that
    the IS estimator is unbiased / consistent and variance-reducing.
    """
    p_hat: float
    se: float
    n_paths: int


def compute_hit_by_horizon_labels_from_state(
    state_df: pd.DataFrame,
    tdays: int,
    d: float,
    *,
    d_col: str = "D",
) -> pd.Series:
    """
    PDF label:
        y_k = 1{ max_{0<=i<=Tdays} D_{k+i} >= d }.

    We return NaN for the final tdays rows where the future window is incomplete.
    """
    if tdays <= 0:
        raise ValueError("tdays must be positive.")
    if d <= 0.0:
        raise ValueError("d must be positive.")

    df = state_df.copy()
    if d_col not in df.columns:
        raise KeyError(f"state_df missing required column {d_col!r}")
    df = df.sort_index()
    D = df[d_col].to_numpy(dtype=float)
    n = int(len(D))

    y = np.full(n, np.nan, dtype=float)
    for k in range(n):
        if not np.isfinite(D[k]):
            continue
        end = k + int(tdays)
        if end >= n:
            continue
        mx = float(np.nanmax(D[k : (end + 1)]))
        y[k] = 1.0 if mx >= float(d) else 0.0

    return pd.Series(y, index=df.index, name=f"hit_T{tdays}_d{d:.6g}")


def add_realised_labels_to_forecasts(
    forecasts_df: pd.DataFrame,
    state_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    tdays_col: str = "Tdays",
    delta_col: str = "delta",
) -> pd.DataFrame:
    f = forecasts_df.copy()
    f[date_col] = pd.to_datetime(f[date_col])
    f = f.sort_values([date_col, tdays_col, delta_col]).reset_index(drop=True)

    s = state_df.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    y_out = pd.Series(np.nan, index=f.index, dtype=float, name="y")
    for (T, delta), sub in f.groupby([tdays_col, delta_col], sort=False):
        d = float(log_drawdown_threshold(float(delta)))
        y = compute_hit_by_horizon_labels_from_state(s, int(T), d)
        dates = pd.to_datetime(sub[date_col]).to_numpy()
        y_out.loc[sub.index] = y.reindex(dates).to_numpy(dtype=float)

    f["y"] = y_out.to_numpy(dtype=float)
    return f


def brier_score(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2


def log_score(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def score_forecasts(
    df: pd.DataFrame,
    *,
    p_col: str = "p_hat",
    y_col: str = "y",
) -> dict:
    d = df.copy()
    d = d[np.isfinite(d[y_col].to_numpy(dtype=float))]
    if d.empty:
        return {"n": 0, "mean_log_score": np.nan, "mean_brier": np.nan}

    p = d[p_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    return {
        "n": int(len(d)),
        "mean_log_score": float(np.mean(log_score(p, y))),
        "mean_brier": float(np.mean(brier_score(p, y))),
    }


def empirical_rate_baseline_scores(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
) -> dict:
    """
    Constant-probability baseline using the empirical event rate in the supplied sample.
    """
    d = df.copy()
    y = pd.to_numeric(d[y_col], errors="coerce")
    d = d[np.isfinite(y.to_numpy(dtype=float))]
    if d.empty:
        return {"n": 0, "p_const": np.nan, "mean_log_score": np.nan, "mean_brier": np.nan}

    yv = d[y_col].to_numpy(dtype=float)
    p_const = float(np.mean(yv))
    p = np.full(len(yv), p_const, dtype=float)
    return {
        "n": int(len(yv)),
        "p_const": p_const,
        "mean_log_score": float(np.mean(log_score(p, yv))),
        "mean_brier": float(np.mean(brier_score(p, yv))),
    }


def empirical_rate_baseline_scores_by_group(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
    y_col: str = "y",
) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame([empirical_rate_baseline_scores(df, y_col=y_col)])

    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        s = empirical_rate_baseline_scores(sub, y_col=y_col)
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        row = {col: val for col, val in zip(group_cols, keys_t)}
        row.update(s)
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(
    df: pd.DataFrame,
    *,
    p_col: str = "p_hat",
    y_col: str = "y",
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins <= 1:
        raise ValueError("n_bins must be >= 2")

    d = df.copy()
    d = d[np.isfinite(d[y_col].to_numpy(dtype=float))]
    if d.empty:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "count", "p_mean", "y_mean"])

    p = d[p_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(p, edges, right=False) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            rows.append(
                {"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "count": 0, "p_mean": np.nan, "y_mean": np.nan}
            )
            continue
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": int(np.sum(mask)),
                "p_mean": float(np.mean(p[mask])),
                "y_mean": float(np.mean(y[mask])),
            }
        )
    return pd.DataFrame(rows)


def reliability_bins_by_group(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
    p_col: str = "p_hat",
    y_col: str = "y",
    n_bins: int = 10,
) -> pd.DataFrame:
    if not group_cols:
        return reliability_bins(df, p_col=p_col, y_col=y_col, n_bins=n_bins)

    parts: list[pd.DataFrame] = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        r = reliability_bins(sub, p_col=p_col, y_col=y_col, n_bins=n_bins)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(group_cols, key_tuple):
            r[col] = val
        parts.append(r)

    if not parts:
        cols = list(group_cols) + ["bin_lo", "bin_hi", "count", "p_mean", "y_mean"]
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def score_forecasts_by_group(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
    p_col: str = "p_hat",
    y_col: str = "y",
) -> pd.DataFrame:
    if not group_cols:
        s = score_forecasts(df, p_col=p_col, y_col=y_col)
        return pd.DataFrame([{**s}])

    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_cols), sort=True):
        s = score_forecasts(sub, p_col=p_col, y_col=y_col)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {col: val for col, val in zip(group_cols, key_tuple)}
        row.update(s)
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_naive_mc(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d_thresh: float,
    n_steps: int,
    n_paths: int,
    *,
    seed: int = 123,
) -> NaiveMCResult:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    u0 = np.zeros((int(n_steps), 2), dtype=float)
    hit, L = simulate_is_paths(
        params=params,
        x0=float(x0),
        v0=float(v0),
        d0=float(d0),
        d_thresh=float(d_thresh),
        n_steps=int(n_steps),
        u=u0,
        n_paths=int(n_paths),
        seed=int(seed),
    )

    if not np.allclose(L, 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("Naive MC expected all weights L==1 when u≡0, but got deviation.")

    hit = np.asarray(hit, dtype=float)
    p_hat = float(np.mean(hit))
    se = float(np.sqrt(np.var(hit, ddof=1) / len(hit))) if len(hit) > 1 else 0.0
    return NaiveMCResult(p_hat=p_hat, se=se, n_paths=int(n_paths))


def compare_estimates_zscore(
    p1: float,
    se1: float,
    p2: float,
    se2: float,
    *,
    floor: float = 1e-12,
) -> float:
    denom = float(np.sqrt(max(floor, se1 * se1 + se2 * se2)))
    return float((p1 - p2) / denom)


def naive_mc_sanity_check_against_is(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d_thresh: float,
    n_steps: int,
    u: np.ndarray,
    *,
    n_is: int = 2000,
    n_mc: int = 5000,
    seed_is: int = 11,
    seed_mc: int = 22,
) -> Tuple[ISResult, NaiveMCResult, float]:
    hit_is, L_is = simulate_is_paths(
        params=params,
        x0=float(x0),
        v0=float(v0),
        d0=float(d0),
        d_thresh=float(d_thresh),
        n_steps=int(n_steps),
        u=np.asarray(u, dtype=float),
        n_paths=int(n_is),
        seed=int(seed_is),
    )
    is_res = estimate_probability_is(hit_is, L_is)

    mc_res = simulate_naive_mc(
        params=params,
        x0=float(x0),
        v0=float(v0),
        d0=float(d0),
        d_thresh=float(d_thresh),
        n_steps=int(n_steps),
        n_paths=int(n_mc),
        seed=int(seed_mc),
    )

    z = compare_estimates_zscore(is_res.p_hat, is_res.se, mc_res.p_hat, mc_res.se)
    return is_res, mc_res, z
