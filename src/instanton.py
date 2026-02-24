from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .constants import ALPHA_M, ALPHA_V, HIT_EPS, V_MIN
from .fit import SVParams
from .model_sv import effective_variance


# ----------------------------
# Smooth max + smooth drawdown
# ----------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic / expit."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def smax_alpha(a: np.ndarray, b: np.ndarray, alpha_m: float) -> np.ndarray:
    r"""
    smax_α(a,b) := b + (1/α) log(1 + exp(α(a-b))).

    PDF-committed smooth-max surrogate with uniform bound:
      max(a,b) <= smax_α(a,b) <= max(a,b) + log(2)/α.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = alpha_m * (a - b)
    return b + (1.0 / alpha_m) * (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def smooth_drawdown_path(x: np.ndarray, m0: float, alpha_m: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Optimiser-side smoothed running peak + drawdown:
      \tilde m_0 = m0
      \tilde m_{i+1} = smax_α(\tilde m_i, x_{i+1})
      \tilde D_i = \tilde m_i - x_i
    """
    x = np.asarray(x, dtype=float)
    m = np.empty_like(x)
    d = np.empty_like(x)
    m[0] = float(m0)
    d[0] = m[0] - x[0]
    for i in range(len(x) - 1):
        m[i + 1] = smax_alpha(m[i], x[i + 1], alpha_m=alpha_m)
        d[i + 1] = m[i + 1] - x[i + 1]
    return m, d


def exact_drawdown_at_tau(x: np.ndarray, m0: float, d0: float, tau: int) -> float:
    """
    Exact recursion (used for simulation/evaluation post-check):
      m_{i+1} = max(m_i, x_{i+1})
      D_{i+1} = m_{i+1} - x_{i+1}
    """
    m = float(m0)
    D = float(d0)
    for i in range(tau):
        m = max(m, float(x[i + 1]))
        D = m - float(x[i + 1])
    return float(D)


# ----------------------------
# Model coefficients (matched)
# ----------------------------
def drift_b_eff(params: SVParams, v: float) -> np.ndarray:
    v_eff = float(effective_variance(np.array([v]))[0])
    return np.array([params.mu, params.kappa * (params.vbar - v_eff)], dtype=float)


def sigma_eff_inv(params: SVParams, v: float) -> np.ndarray:
    """
    Independent-driver representation (ΔB1, ΔB2) with single effective variance:
      Δx = sqrt(v_eff) ΔB1
      Δv = xi sqrt(v_eff) (rho ΔB1 + sqrt(1-rho^2) ΔB2)
    """
    v_eff = float(effective_variance(np.array([v]))[0])
    sv = np.sqrt(max(1e-16, v_eff))
    rho = float(params.rho)
    s = np.sqrt(max(1e-12, 1.0 - rho * rho))
    Ainv = np.array([[1.0, 0.0], [-rho / s, 1.0 / (params.xi * s)]], dtype=float)
    return (1.0 / sv) * Ainv


def u_from_path(params: SVParams, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Stepwise control extracted from a discrete path on the daily grid:
      u_i = σ_eff(z_i)^{-1} ( (z_{i+1}-z_i)/Δt - b_eff(z_i) )
    """
    dt = params.dt_years
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    m = len(x) - 1
    u = np.zeros((m, 2), dtype=float)
    for i in range(m):
        z_i = np.array([x[i], v[i]], dtype=float)
        z_ip1 = np.array([x[i + 1], v[i + 1]], dtype=float)
        dz_dt = (z_ip1 - z_i) / dt
        b = drift_b_eff(params, v[i])
        u[i] = sigma_eff_inv(params, v[i]) @ (dz_dt - b)
    return u


def matched_action(params: SVParams, x: np.ndarray, v: np.ndarray) -> float:
    """
    Matched discretised (daily-grid) Freidlin–Wentzell action:
      S = 1/2 Σ Δt ||u_i||^2
    """
    u = u_from_path(params, x, v)
    return 0.5 * float(np.sum(params.dt_years * np.sum(u * u, axis=1)))


# ----------------------------
# Fixed-time MAM (standard solver)
# ----------------------------
@dataclass(frozen=True)
class MAMSettings:
    """
    Fixed-time MAM implementation aligned with the PDF and reference [E, Ren, Vanden-Eijnden 2004]:
      discretise the action and optimise it using (L-)BFGS with supplied gradients.

    We enforce:
      - smooth drawdown constraint in the optimiser,
      - exact recursion only for evaluation / post-check.
    """
    max_iter: int = 200
    tol_grad: float = 1e-6
    maxcor: int = 10  # L-BFGS memory
    gamma_pre: float = 5e4
    pre_eps: float = HIT_EPS

    # kept for CLI/backward compatibility (unused by L-BFGS directly)
    step0: float = 0.5
    grad_eps: float = 1e-4

    # continuation post-check
    exact_tol: float = 2e-3


@dataclass(frozen=True)
class InstantonSolution:
    tau: int
    alpha_m: float
    action: float
    u_star: np.ndarray
    x_path: np.ndarray
    v_path: np.ndarray
    dtilde_tau: float
    dexact_tau: float
    prehit_max_violation: float
    hit_err: float


# ----------------------------
# Internals: derivatives needed by MAM
# ----------------------------
def _dveff_dv(v: float, vmin: float = V_MIN, alpha_v: float = ALPHA_V) -> float:
    """
    For v_eff = vmin + softplus_{alpha_v}(v - vmin),
    derivative is sigmoid(alpha_v * (v - vmin)).
    """
    x = alpha_v * (float(v) - vmin)
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _inv_softplus_alpha(y: float, alpha: float) -> float:
    """
    Invert softplus_{alpha}(x) = (1/alpha) log(1+exp(alpha x)) for y>0:
      x = (1/alpha) log(exp(alpha y) - 1).
    """
    y = float(y)
    if y <= 0.0:
        raise ValueError("inv_softplus_alpha requires y>0.")
    return float((1.0 / alpha) * np.log(np.expm1(alpha * y)))


def _x_tau_from_mprev(mprev: float, d: float, alpha_m: float) -> float:
    """
    Enforce the optimiser equality constraint \tilde D_tau = d exactly by solving:
      smax_alpha(mprev, x_tau) - x_tau = d
    which yields:
      x_tau = mprev - inv_softplus_alpha(d, alpha_m).
    """
    return float(mprev - _inv_softplus_alpha(d, alpha_m))


def _smooth_prefix_m_d(
    x0: float, x_interior: np.ndarray, m0: float, alpha_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (\tilde m_i, \tilde D_i) for i=0..tau-1 for a prefix path
    [x0, x1, ..., x_{tau-1}] (i.e., without x_tau), and cache logistic weights for reverse-mode.

    Returns:
      mtil: shape (tau,) for indices 0..tau-1
      Dtil: shape (tau,) for indices 0..tau-1
      sig : shape (tau-1,) where sig[i] = d/da smax(a,b) at (a=m_i, b=x_{i+1})
    """
    x_interior = np.asarray(x_interior, dtype=float)
    tau = int(x_interior.shape[0] + 1)

    x_prefix = np.empty(tau, dtype=float)
    x_prefix[0] = float(x0)
    if tau > 1:
        x_prefix[1:] = x_interior

    mtil = np.empty(tau, dtype=float)
    Dtil = np.empty(tau, dtype=float)
    sig = np.empty(max(0, tau - 1), dtype=float)

    mtil[0] = float(m0)
    Dtil[0] = mtil[0] - x_prefix[0]

    for i in range(tau - 1):
        s = _sigmoid(alpha_m * (mtil[i] - x_prefix[i + 1]))
        sig[i] = s
        mtil[i + 1] = smax_alpha(mtil[i], x_prefix[i + 1], alpha_m=alpha_m)
        Dtil[i + 1] = mtil[i + 1] - x_prefix[i + 1]

    return mtil, Dtil, sig


def _mprev_sensitivities(sig: np.ndarray) -> np.ndarray:
    """
    Return sensitivities dm_{tau-1}/dx_j (j=1..tau-1) for the smooth recursion.
    """
    sig = np.asarray(sig, dtype=float)
    tau_minus_1 = int(sig.shape[0])  # tau-1
    if tau_minus_1 <= 0:
        return np.zeros(0, dtype=float)

    adj_m = np.zeros(tau_minus_1 + 1, dtype=float)
    adj_m[-1] = 1.0
    grad_x = np.zeros(tau_minus_1 + 1, dtype=float)  # includes x0; drop later

    for i in range(tau_minus_1 - 1, -1, -1):
        adj_m[i] += adj_m[i + 1] * sig[i]
        grad_x[i + 1] += adj_m[i + 1] * (1.0 - sig[i])

    return grad_x[1:]


def _action_and_grad(params: SVParams, x: np.ndarray, v: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Matched discretised action and gradients w.r.t. all nodes x_i, v_i.
    Vectorized over time steps (no change in math).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = float(params.dt_years)
    n = int(len(x))
    m = n - 1
    if m <= 0:
        return 0.0, np.zeros_like(x), np.zeros_like(v)

    rho = float(params.rho)
    s = np.sqrt(max(1e-12, 1.0 - rho * rho))

    # C is constant across i
    C = np.array([[1.0, 0.0], [-rho / s, 1.0 / (params.xi * s)]], dtype=float)
    G = C.T @ C  # constant

    invdt = 1.0 / dt

    v_i = v[:-1]
    # v_eff_i for i=0..m-1
    v_eff = effective_variance(v_i)
    v_eff = np.maximum(v_eff, 1e-16)
    sv = np.sqrt(v_eff)
    inv_sv = 1.0 / sv
    inv_v_eff = 1.0 / v_eff

    # dveff/dv in vector form (same as scalar _dveff_dv)
    xsig = ALPHA_V * (v_i - V_MIN)
    xsig = np.clip(xsig, -60.0, 60.0)
    dve = 1.0 / (1.0 + np.exp(-xsig))

    dx = (x[1:] - x[:-1]) * invdt
    dv = (v[1:] - v[:-1]) * invdt

    b1 = params.mu
    b2 = params.kappa * (params.vbar - v_eff)

    w1 = dx - b1
    w2 = dv - b2
    w = np.stack([w1, w2], axis=1)  # (m,2)

    # u_i = (1/sqrt(v_eff_i)) * C @ w_i
    u = (w @ C.T) * inv_sv[:, None]
    uu = np.sum(u * u, axis=1)
    action = 0.5 * dt * float(np.sum(uu))

    # p_i = A^T (dt u_i) = dt * (A^T A) w_i = dt*(1/v_eff_i)*(G @ w_i)
    p = (w @ G.T) * (dt * inv_v_eff)[:, None]
    p1 = p[:, 0]
    p2 = p[:, 1]

    grad_x = np.zeros_like(x)
    grad_v = np.zeros_like(v)

    # x gradients (finite-difference structure)
    grad_x[1:] += invdt * p1
    grad_x[:-1] -= invdt * p1

    # v gradients (finite-difference structure)
    grad_v[1:] += invdt * p2
    grad_v[:-1] -= invdt * p2

    # extra v_i terms:
    # + p2 * kappa * dve  (from b2 dependence)
    grad_v[:-1] += p2 * params.kappa * dve

    # + scale * dt * ||u||^2, scale = -(1/2) * dve / v_eff (from A dependence)
    scale = -(0.5) * dve * inv_v_eff
    grad_v[:-1] += scale * dt * uu

    return float(action), grad_x, grad_v


def _prehit_penalty_and_grad_x_from_prefix(
    Dtil: np.ndarray,
    sig: np.ndarray,
    *,
    d: float,
    settings: MAMSettings,
) -> Tuple[float, np.ndarray, float]:
    """
    Same penalty/gradient as _prehit_penalty_and_grad_x, but reuses the already-computed
    smooth prefix drawdown recursion outputs:
      - Dtil: shape (tau,) for i=0..tau-1
      - sig : shape (tau-1,) logistic weights used in reverse-mode

    No change in math: this just avoids recomputing _smooth_prefix_m_d().
    """
    Dtil = np.asarray(Dtil, dtype=float)
    sig = np.asarray(sig, dtype=float)
    tau = int(Dtil.shape[0])

    if tau <= 0:
        return 0.0, np.zeros(0, dtype=float), 0.0

    thr = float(d - settings.pre_eps)
    viol = np.maximum(Dtil - thr, 0.0)  # includes i=0 exactly as your current implementation

    pre_max = float(np.max(viol)) if tau > 0 else 0.0
    pen = float(settings.gamma_pre * np.sum(viol * viol))

    # Reverse-mode to compute gradient wrt x_prefix = (x0, x1..x_{tau-1})
    # Then return grad wrt x_interior = (x1..x_{tau-1}) only.
    gD = 2.0 * settings.gamma_pre * viol
    adj_m = gD.copy()
    grad_x_prefix = -gD.copy()

    for i in range(tau - 2, -1, -1):
        # sig[i] exists for i=0..tau-2
        adj_m[i] += adj_m[i + 1] * sig[i]
        grad_x_prefix[i + 1] += adj_m[i + 1] * (1.0 - sig[i])

    return pen, grad_x_prefix[1:], pre_max


# ----------------------------
# Public solver API
# ----------------------------
def _initial_path_guess(x0: float, v0: float, d0: float, d: float, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline init (PDF): linear downmove in x, hold v constant."""
    m0 = x0 + d0
    xT = m0 - d
    x = np.linspace(x0, xT, tau + 1)
    v = np.full(tau + 1, float(v0))
    return x, v


def solve_instanton_tau_fixed_time_mam(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d: float,
    tau: int,
    alpha_m: float,
    settings: MAMSettings,
    warm_start: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> InstantonSolution:
    """
    Fixed-horizon MAM on the daily grid for candidate discrete hitting day tau.

    - Minimises the matched discretised action via L-BFGS with analytic gradients (standard MAM practice).
    - Enforces \tilde D_tau = d EXACTLY by solving for x_tau in closed form.
    - Enforces pre-hit \tilde D_i < d for all i < tau (i = 0..tau-1) via squared hinge penalties.
    """
    if tau <= 0:
        raise ValueError("tau must be a positive integer (tau>=1).")
    if d <= 0.0:
        raise ValueError("d must be positive.")

    m0 = float(x0 + d0)

    if warm_start is None:
        x_init, v_init = _initial_path_guess(x0, v0, d0, d, tau)
    else:
        x_init, v_init = warm_start
        x_init = np.asarray(x_init, dtype=float).copy()
        v_init = np.asarray(v_init, dtype=float).copy()
        if len(x_init) != tau + 1 or len(v_init) != tau + 1:
            x_init, v_init = _initial_path_guess(x0, v0, d0, d, tau)

    x_init[0] = float(x0)
    v_init[0] = float(v0)

    x_interior0 = x_init[1:tau] if tau > 1 else np.zeros(0, dtype=float)  # x1..x_{tau-1}
    v_vars0 = v_init[1:].copy()  # v1..v_tau
    free0 = np.concatenate([x_interior0, v_vars0], axis=0)

    def unpack_free(free: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        free = np.asarray(free, dtype=float)
        x_interior = free[: max(0, tau - 1)]
        v_vars = free[max(0, tau - 1):]
        if v_vars.shape[0] != tau:
            raise ValueError(f"Expected tau={tau} variance variables; got {v_vars.shape[0]}.")

        mtil, Dtil, sig = _smooth_prefix_m_d(x0=float(x0), x_interior=x_interior, m0=m0, alpha_m=float(alpha_m))
        mprev = float(mtil[tau - 1])
        x_tau = _x_tau_from_mprev(mprev, d=float(d), alpha_m=float(alpha_m))

        x = np.empty(tau + 1, dtype=float)
        x[0] = float(x0)
        if tau > 1:
            x[1:tau] = x_interior
        x[tau] = x_tau

        v = np.empty(tau + 1, dtype=float)
        v[0] = float(v0)
        v[1:] = v_vars

        dmprev_dx = _mprev_sensitivities(sig)  # dm_{tau-1}/dx_interior

        return x, v, Dtil, x_tau, dmprev_dx, sig

    def fun_and_jac(free: np.ndarray) -> Tuple[float, np.ndarray]:
        x, v, Dtil_prefix, _x_tau, dmprev_dx, sig = unpack_free(free)

        act, gx, gv = _action_and_grad(params, x, v)

        pen, gpre_x_interior, _pre_max = _prehit_penalty_and_grad_x_from_prefix(
            Dtil_prefix,
            sig,
            d=float(d),
            settings=settings,
        )

        # Chain: x_tau = m_{tau-1} - const, so dx_tau/dx_interior = dm_{tau-1}/dx_interior
        if tau > 1:
            gx_interior = gx[1:tau] + gpre_x_interior + gx[tau] * dmprev_dx
        else:
            gx_interior = np.zeros(0, dtype=float)

        gv_vars = gv[1:]  # v1..v_tau

        obj = float(act + pen)
        grad = np.concatenate([gx_interior, gv_vars], axis=0)
        return obj, grad

    res = minimize(
        fun=lambda z: fun_and_jac(z)[0],
        x0=free0,
        jac=lambda z: fun_and_jac(z)[1],
        method="L-BFGS-B",
        options={"maxiter": int(settings.max_iter), "gtol": float(settings.tol_grad), "maxcor": int(settings.maxcor)},
    )

    x, v, _Dtil_prefix, _x_tau, _dmprev_dx, _sig = unpack_free(res.x)

    mtil_full, Dtil_full = smooth_drawdown_path(x, m0=m0, alpha_m=float(alpha_m))

    # Literal pre-hit check includes i=0..tau-1
    thr = float(d - settings.pre_eps)
    pre_max = float(np.max(np.maximum(Dtil_full[:tau] - thr, 0.0))) if tau > 0 else 0.0

    hit_err = float(Dtil_full[tau] - d)  # should be ~0 (equality enforced)

    dexact = exact_drawdown_at_tau(x, m0=m0, d0=float(d0), tau=tau)

    u_star = u_from_path(params, x, v)
    act = matched_action(params, x, v)

    return InstantonSolution(
        tau=int(tau),
        alpha_m=float(alpha_m),
        action=float(act),
        u_star=u_star,
        x_path=x,
        v_path=v,
        dtilde_tau=float(Dtil_full[tau]),
        dexact_tau=float(dexact),
        prehit_max_violation=float(pre_max),
        hit_err=float(hit_err),
    )


def _resample_warm_start(sol: InstantonSolution, new_tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear resample in index space (used for warm-starting across τ)."""
    old_tau = int(sol.tau)
    if old_tau == new_tau:
        return sol.x_path, sol.v_path
    t_old = np.linspace(0.0, 1.0, old_tau + 1)
    t_new = np.linspace(0.0, 1.0, new_tau + 1)
    x_new = np.interp(t_new, t_old, sol.x_path)
    v_new = np.interp(t_new, t_old, sol.v_path)
    return x_new, v_new


def solve_tau_with_alpha_continuation(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d: float,
    tau: int,
    alpha_seq: Sequence[float] = (50.0, 100.0, 200.0),
    settings: Optional[MAMSettings] = None,
    warm_start: Optional[InstantonSolution] = None,
) -> InstantonSolution:
    """
    Continuation in alpha_m:
      start from alpha_m=50 and increase if the exact drawdown at tau materially undershoots d.
    """
    if settings is None:
        settings = MAMSettings()

    ws_xy = None
    if warm_start is not None:
        ws_xy = _resample_warm_start(warm_start, tau)

    sol: Optional[InstantonSolution] = None
    for alpha_m in alpha_seq:
        sol = solve_instanton_tau_fixed_time_mam(
            params=params,
            x0=x0,
            v0=v0,
            d0=d0,
            d=d,
            tau=tau,
            alpha_m=float(alpha_m),
            settings=settings,
            warm_start=ws_xy,
        )
        ws_xy = (sol.x_path, sol.v_path)

        if sol.dexact_tau >= d - settings.exact_tol:
            return sol

    assert sol is not None
    return sol


def tau_coarse_grid(n_steps: int) -> List[int]:
    """Coarse τ grids (PDF suggestion)."""
    if n_steps <= 20:
        return [5, 10, 15, n_steps]
    return [10, 20, 30, 40, 50, n_steps]


def select_instanton_tau_star(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d: float,
    n_steps: int,
    refine_radius: int = 5,
    settings: Optional[MAMSettings] = None,
    warm_start: Optional[InstantonSolution] = None,
) -> InstantonSolution:
    """
    Minimise over discrete hitting day τ ∈ {1,...,n} using coarse-to-fine search + warm-starts.

    Change (per PDF operational guidance):
      - If warm_start is provided (e.g., yesterday's best solution), use it for ALL candidate τ values.
        solve_tau_with_alpha_continuation will resample it to the requested τ via _resample_warm_start.
    """
    if settings is None:
        settings = MAMSettings()

    # PDF regime check: if d0 >= d, first-hitting instanton constraint is infeasible as stated.
    if float(d0) >= float(d):
        return InstantonSolution(
            tau=0,
            alpha_m=float("nan"),
            action=0.0,
            u_star=np.zeros((0, 2), dtype=float),
            x_path=np.array([float(x0)], dtype=float),
            v_path=np.array([float(v0)], dtype=float),
            dtilde_tau=float(d0),
            dexact_tau=float(d0),
            prehit_max_violation=0.0,
            hit_err=0.0,
        )

    sols: Dict[int, InstantonSolution] = {}
    coarse = [t for t in tau_coarse_grid(n_steps) if 1 <= t <= n_steps]

    for tau in coarse:
        sols[tau] = solve_tau_with_alpha_continuation(
            params=params,
            x0=x0,
            v0=v0,
            d0=d0,
            d=d,
            tau=tau,
            settings=settings,
            warm_start=warm_start,
        )

    best_tau = min(sols.keys(), key=lambda t: sols[t].action)
    lo = max(1, best_tau - refine_radius)
    hi = min(n_steps, best_tau + refine_radius)

    for tau in range(lo, hi + 1):
        if tau in sols:
            continue
        sols[tau] = solve_tau_with_alpha_continuation(
            params=params,
            x0=x0,
            v0=v0,
            d0=d0,
            d=d,
            tau=tau,
            settings=settings,
            warm_start=warm_start,
        )

    return sols[min(sols.keys(), key=lambda t: sols[t].action)]