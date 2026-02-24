from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.io import find_latest_file, load_json
from src.fit import SVParams, fit_sv_params_qmle_from_state
from src.instanton import select_instanton_tau_star, MAMSettings
from src.importance_sampling import simulate_is_paths, estimate_probability_is, cap_control
from src.evaluation import simulate_naive_mc, compare_estimates_zscore
from src.constants import log_drawdown_threshold, LAMBDA_IS, UMAX


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
PARAMS_DIR = REPO_ROOT / "outputs" / "params"

_CACHED_PARAMS: SVParams | None = None


def _read_latest_state() -> pd.DataFrame:
    p = find_latest_file(DATA_PROCESSED, "state_*.csv")
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")


def _get_params(state_df: pd.DataFrame) -> SVParams:
    """
    Prefer latest saved params JSON (if present). If none exists, fit QMLE quickly.
    """
    global _CACHED_PARAMS
    if _CACHED_PARAMS is not None:
        return _CACHED_PARAMS

    try:
        p = find_latest_file(PARAMS_DIR, "sv_params_*.json")
        js = load_json(p)
        _CACHED_PARAMS = SVParams(**js["params"])
        return _CACHED_PARAMS
    except FileNotFoundError:
        params, _ = fit_sv_params_qmle_from_state(
            state_df=state_df,
            train_end="2006-12-31",
            maxiter=40,
            price_ticker="SPY",
        )
        _CACHED_PARAMS = params
        return _CACHED_PARAMS


def _pick_row_below_threshold(state: pd.DataFrame, d: float, margin: float = 1e-3) -> pd.Series:
    """
    Pick a row with D_k < d - margin so that τ_d^{(k)} is not 0.
    This avoids the τ=0 edge case when testing first-hit constraints.
    """
    sub = state[(state["D"] < d - margin)].copy()
    if sub.empty:
        pytest.skip("No rows with D < d - margin; cannot test first-hit constraints safely.")
    return sub.iloc[0]


def _pick_reasonable_row_for_mc(state: pd.DataFrame, d: float, margin: float = 1e-3) -> pd.Series:
    """
    For naive-vs-IS agreement tests, try to pick a row that is likely to give a non-degenerate
    probability (not ~0, not ~1). We prefer a crisis-ish date if present, else fall back.
    """
    # Prefer a known volatile date in your dataset if available (stable and deterministic)
    prefer = pd.Timestamp("2008-09-02")
    if prefer in state.index:
        row = state.loc[prefer]
        if pd.notna(row["X"]) and pd.notna(row["V_proxy"]) and pd.notna(row["D"]) and float(row["D"]) < d - margin:
            return row

    # Otherwise: pick among high-variance rows below threshold
    sub = state[(state["D"] < d - margin)].dropna(subset=["X", "V_proxy", "D"]).copy()
    if sub.empty:
        return _pick_row_below_threshold(state, d=d, margin=margin)

    q = float(sub["V_proxy"].quantile(0.95))
    hi = sub[sub["V_proxy"] >= q]
    if hi.empty:
        return sub.iloc[0]
    return hi.iloc[0]


def test_mam_constraints_and_postcheck_hold() -> None:
    state = _read_latest_state().dropna(subset=["X", "V_proxy", "D"])
    params = _get_params(state)

    tdays = 20
    d = log_drawdown_threshold(0.20)

    row = _pick_row_below_threshold(state, d=d, margin=1e-3)
    x0, v0, d0 = float(row["X"]), float(row["V_proxy"]), float(row["D"])

    settings = MAMSettings(
        max_iter=60,
        tol_grad=1e-6,
        exact_tol=1e-2,
    )

    sol = select_instanton_tau_star(
        params=params,
        x0=x0,
        v0=v0,
        d0=d0,
        d=d,
        n_steps=tdays,
        refine_radius=1,
        settings=settings,
        warm_start=None,
    )

    assert abs(sol.hit_err) <= 5e-3
    assert sol.prehit_max_violation <= 5e-3
    assert sol.dexact_tau >= d - 1e-2


def test_one_query_is_runs() -> None:
    state = _read_latest_state().dropna(subset=["X", "V_proxy", "D"])
    params = _get_params(state)

    tdays = 20
    d = log_drawdown_threshold(0.20)

    row = _pick_row_below_threshold(state, d=d, margin=1e-3)
    x0, v0, d0 = float(row["X"]), float(row["V_proxy"]), float(row["D"])

    sol = select_instanton_tau_star(
        params=params,
        x0=x0, v0=v0, d0=d0,
        d=d,
        n_steps=tdays,
        refine_radius=1,
        settings=MAMSettings(max_iter=40, exact_tol=1e-2),
    )

    u_star_full = np.zeros((tdays, 2))
    if sol.tau > 0:
        u_star_full[: sol.tau] = sol.u_star

    u = cap_control(LAMBDA_IS * u_star_full, umax=UMAX)

    hit, L = simulate_is_paths(params, x0, v0, d0, d, tdays, u, n_paths=300, seed=11)
    res = estimate_probability_is(hit, L)

    assert 0.0 <= res.p_hat <= 1.0
    assert res.se >= 0.0
    assert res.ess > 0.0


def test_naive_mc_runs_and_bounds() -> None:
    state = _read_latest_state().dropna(subset=["X", "V_proxy", "D"])
    params = _get_params(state)

    tdays = 20
    d = log_drawdown_threshold(0.20)

    row = _pick_row_below_threshold(state, d=d, margin=1e-3)
    x0, v0, d0 = float(row["X"]), float(row["V_proxy"]), float(row["D"])

    mc = simulate_naive_mc(params, x0, v0, d0, d, tdays, n_paths=1500, seed=101)

    assert 0.0 <= mc.p_hat <= 1.0
    assert mc.se >= 0.0
    assert mc.n_paths == 1500


def test_is_agrees_with_naive_mc_on_moderate_event() -> None:
    """
    PDF baseline sanity check: where feasible, naive MC under P should be consistent with
    the IS estimate (unbiasedness / correctness of LR). We keep this test in a moderate
    regime so naive MC has reasonable precision.
    """
    state = _read_latest_state().dropna(subset=["X", "V_proxy", "D"])
    params = _get_params(state)

    tdays = 20
    d = log_drawdown_threshold(0.20)

    row = _pick_reasonable_row_for_mc(state, d=d, margin=1e-3)
    x0, v0, d0 = float(row["X"]), float(row["V_proxy"]), float(row["D"])

    sol = select_instanton_tau_star(
        params=params,
        x0=x0, v0=v0, d0=d0,
        d=d,
        n_steps=tdays,
        refine_radius=1,
        settings=MAMSettings(max_iter=40, exact_tol=1e-2),
    )

    u_star_full = np.zeros((tdays, 2))
    if sol.tau > 0:
        u_star_full[: sol.tau] = sol.u_star
    u = cap_control(LAMBDA_IS * u_star_full, umax=UMAX)

    # IS estimate
    hit_is, L_is = simulate_is_paths(params, x0, v0, d0, d, tdays, u, n_paths=2000, seed=7)
    is_res = estimate_probability_is(hit_is, L_is)

    # Naive MC baseline (u=0)
    mc_res = simulate_naive_mc(params, x0, v0, d0, d, tdays, n_paths=5000, seed=9)

    # If the event is too close to 0 or 1 under the model, this check can become numerically
    # uninformative (tiny SE). In that case, skip rather than be flaky.
    if mc_res.p_hat < 0.02 or mc_res.p_hat > 0.98:
        pytest.skip("Naive MC probability is too extreme for a stable agreement test.")

    z = abs(compare_estimates_zscore(is_res.p_hat, is_res.se, mc_res.p_hat, mc_res.se))
    assert z <= 4.5, f"IS vs naive MC mismatch too large: z={z:.3f}, is={is_res.p_hat:.4g}, mc={mc_res.p_hat:.4g}"