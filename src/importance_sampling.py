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
    u: np.ndarray,  # shape (n_steps, 2), drift shift on independent drivers
    n_paths: int,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate under Q on the matched daily grid:

      ΔB_i ~ N(u_i Δt, Δt I2)
      x_{i+1} = x_i + μΔt + sqrt(v_eff_i) ΔB^{(1)}_i
      v_{i+1} = v_i + κ( vbar - v_eff_i )Δt + ξ sqrt(v_eff_i) ( ρ ΔB^{(1)}_i + sqrt(1-ρ^2) ΔB^{(2)}_i )

    Event evaluation is ALWAYS exact (forward-looking target):
      start from (x0, d0), update drawdown by exact running-max recursion after each step,
      and declare hit iff max_{0<=i<=n} D_i >= d_thresh (origin i=0 included, per PDF).

    Returns:
      hit indicator array (N,),
      likelihood weights L (N,).
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

    # PDF definition includes the origin i=0 in max_{0<=i<=n} D_i >= d.
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

    L = np.exp(logL)
    return hit.astype(float), L


def estimate_probability_is(hit: np.ndarray, L: np.ndarray) -> ISResult:
    """
    Unbiased IS estimator (PDF):
      p̂ = (1/N) Σ 1(hit) * L
    """
    hit = np.asarray(hit, dtype=float)
    L = np.asarray(L, dtype=float)

    contrib = hit * L
    p_hat = float(np.mean(contrib))
    se = float(np.sqrt(np.var(contrib, ddof=1) / len(contrib))) if len(contrib) > 1 else 0.0

    sumw = float(np.sum(L))
    sumw2 = float(np.sum(L * L))
    ess = float((sumw * sumw) / sumw2) if sumw2 > 0 else 0.0

    return ISResult(
        p_hat=p_hat,
        se=se,
        ess=ess,
        mean_weight=float(np.mean(L)),
        max_weight=float(np.max(L)),
        hit_rate_under_Q=float(np.mean(hit)),
    )