from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .fit import SVParams
from .model_sv import effective_variance


@dataclass(frozen=True)
class ISResult:
    p_hat: float
    se: float
    ess: float
    mean_weight: float
    max_weight: float
    hit_rate_under_Q: float


def cap_control(u: np.ndarray, umax: float) -> np.ndarray:
    """
    Norm cap (PDF-allowed convention):
      cap(u) = u * min(1, umax / ||u||).
    """
    u = np.asarray(u, dtype=float)
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    scale = np.ones_like(norms)
    mask = norms > umax
    scale[mask] = umax / norms[mask]
    return u * scale


def simulate_is_paths(
    params: SVParams,
    x0: float,
    v0: float,
    d0: float,
    d_thresh: float,
    n_steps: int,
    u: np.ndarray,
    n_paths: int,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate under Q on the matched daily grid.

    Returns:
      hit indicator array (N,),
      likelihood weights L (N,).

    This function is intentionally strict about non-finite weights: if the chosen control causes
    numerical overflow in the Radon–Nikodym weights, we fail loudly instead of returning artefacts.
    """
    dt = float(params.dt_years)
    rng = np.random.default_rng(seed)

    u = np.asarray(u, dtype=float)
    if u.shape != (n_steps, 2):
        raise ValueError(f"u must have shape {(n_steps, 2)}; got {u.shape}")

    x = np.full(n_paths, float(x0))
    v = np.full(n_paths, float(v0))

    m = np.full(n_paths, float(x0 + d0))
    D = np.full(n_paths, float(d0))

    hit = (D >= float(d_thresh))
    logL = np.zeros(n_paths, dtype=float)

    sqrt_dt = np.sqrt(dt)
    rho = float(params.rho)
    sqrt_1mr2 = np.sqrt(max(1e-12, 1.0 - rho * rho))

    for i in range(n_steps):
        ui = u[i]
        eps = rng.standard_normal(size=(n_paths, 2))
        dB = ui * dt + sqrt_dt * eps

        u_dot_dB = ui[0] * dB[:, 0] + ui[1] * dB[:, 1]
        u_norm2 = float(ui[0] ** 2 + ui[1] ** 2)
        logL += (-u_dot_dB + 0.5 * u_norm2 * dt)

        v_eff = effective_variance(v)
        sqrt_v = np.sqrt(v_eff)

        x = x + params.mu * dt + sqrt_v * dB[:, 0]
        v = v + params.kappa * (params.vbar - v_eff) * dt + params.xi * sqrt_v * (
            rho * dB[:, 0] + sqrt_1mr2 * dB[:, 1]
        )

        m = np.maximum(m, x)
        D = m - x
        hit |= (D >= float(d_thresh))

    max_safe_log = float(np.log(np.finfo(np.float64).max) - 2.0)
    if float(np.max(logL)) > max_safe_log:
        raise FloatingPointError(
            "Importance-sampling log weights overflow float64 range. "
            "Reduce the control strength/cap or inspect the instanton solution."
        )

    L = np.exp(logL)
    if not np.all(np.isfinite(L)):
        raise FloatingPointError("Importance-sampling weights contain non-finite values.")
    if np.any(L < 0.0):
        raise FloatingPointError("Importance-sampling weights contain negative values.")

    return hit.astype(float), L


def estimate_probability_is(hit: np.ndarray, L: np.ndarray) -> ISResult:
    """
    Unbiased IS estimator (PDF):
      p̂ = (1/N) Σ 1(hit) * L

    The implementation is intentionally strict: probabilities/weights that are non-finite or wildly
    out of range indicate a numerically unusable run and should not be published.
    """
    hit = np.asarray(hit, dtype=float)
    L = np.asarray(L, dtype=float)

    if hit.ndim != 1 or L.ndim != 1 or hit.shape != L.shape:
        raise ValueError("hit and L must be one-dimensional arrays with the same shape.")
    if len(hit) == 0:
        raise ValueError("estimate_probability_is requires at least one path.")
    if not np.all(np.isfinite(hit)):
        raise FloatingPointError("Hit array contains non-finite values.")
    if not np.all(np.isfinite(L)):
        raise FloatingPointError("Weight array contains non-finite values.")
    if np.any(L < 0.0):
        raise FloatingPointError("Weight array contains negative values.")

    contrib = hit * L
    if not np.all(np.isfinite(contrib)):
        raise FloatingPointError("Weighted hit contributions contain non-finite values.")

    p_hat_raw = float(np.mean(contrib))
    tol = 1e-10
    if p_hat_raw < -tol or p_hat_raw > 1.0 + tol:
        raise FloatingPointError(
            f"Estimated probability lies outside [0,1]: p_hat={p_hat_raw:.12g}. "
            "This indicates an unusable IS run with pathological weights."
        )
    p_hat = float(np.clip(p_hat_raw, 0.0, 1.0))

    se = float(np.sqrt(np.var(contrib, ddof=1) / len(contrib))) if len(contrib) > 1 else 0.0
    if not np.isfinite(se) or se < 0.0:
        raise FloatingPointError(f"Invalid Monte Carlo standard error: se={se}")

    sumw = float(np.sum(L))
    sumw2 = float(np.sum(L * L))
    if not np.isfinite(sumw) or not np.isfinite(sumw2):
        raise FloatingPointError("Weight moments are non-finite.")
    ess = float((sumw * sumw) / sumw2) if sumw2 > 0 else 0.0

    mean_weight = float(np.mean(L))
    max_weight = float(np.max(L))
    hit_rate_q = float(np.mean(hit))

    for name, val in (("ess", ess), ("mean_weight", mean_weight), ("max_weight", max_weight), ("hit_rate_under_Q", hit_rate_q)):
        if not np.isfinite(val):
            raise FloatingPointError(f"Non-finite IS diagnostic {name}={val}")

    return ISResult(
        p_hat=p_hat,
        se=se,
        ess=ess,
        mean_weight=mean_weight,
        max_weight=max_weight,
        hit_rate_under_Q=hit_rate_q,
    )
