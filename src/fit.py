from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import DT_YEARS
from .model_sv import effective_variance


@dataclass(frozen=True)
class SVParams:
    mu: float
    kappa: float
    vbar: float
    xi: float
    rho: float
    dt_years: float = DT_YEARS


def _initial_guess_moments(v: np.ndarray, v1: np.ndarray, r: np.ndarray, dt: float) -> SVParams:
    """
    Cheap moment-based initialiser (not the final estimator).
    """
    v_eff = effective_variance(v)

    # mu init: weighted by 1/v_eff
    w = 1.0 / np.maximum(v_eff, 1e-16)
    mu0 = float(np.sum(w * r) / (dt * np.sum(w)))

    dv = v1 - v
    y = dv / dt
    x = v_eff
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = float(np.sum((x - xm) * (y - ym)) / sxx) if sxx > 0 else 0.0
    kappa0 = max(1e-3, -b)
    vbar0 = max(1e-6, a / kappa0) if (a := float(ym - b * xm)) > 0 else 1e-6

    # rough xi,rho init from residual correlation
    dB1 = (r - mu0 * dt) / np.sqrt(np.maximum(v_eff, 1e-16))
    res_v = dv - kappa0 * (vbar0 - v_eff) * dt
    yv = res_v / np.sqrt(np.maximum(v_eff, 1e-16))
    rho0 = float(np.clip(np.corrcoef(dB1, yv)[0, 1], -0.5, 0.5))
    xi0 = float(max(1e-3, np.sqrt(np.var(yv, ddof=1) / dt)))
    return SVParams(mu=mu0, kappa=kappa0, vbar=vbar0, xi=xi0, rho=rho0, dt_years=dt)


def _pack_unconstrained(p: SVParams) -> np.ndarray:
    # (mu, log kappa, log vbar, log xi, atanh rho)
    return np.array(
        [
            p.mu,
            np.log(p.kappa),
            np.log(p.vbar),
            np.log(p.xi),
            np.arctanh(np.clip(p.rho, -0.999, 0.999)),
        ],
        dtype=float,
    )


def _unpack_unconstrained(theta: np.ndarray, dt: float) -> SVParams:
    mu = float(theta[0])
    kappa = float(np.exp(theta[1]))
    vbar = float(np.exp(theta[2]))
    xi = float(np.exp(theta[3]))
    rho = float(np.tanh(theta[4]))
    rho = float(np.clip(rho, -0.999, 0.999))
    return SVParams(mu=mu, kappa=kappa, vbar=vbar, xi=xi, rho=rho, dt_years=dt)


def _qmle_negloglik(theta: np.ndarray, r: np.ndarray, dv: np.ndarray, v_eff: np.ndarray, dt: float) -> float:
    """
    QMLE negative log-likelihood under the matched daily increment scheme.
    Uses the conditional bivariate Gaussian of (r, dv) given v_eff.
    """
    p = _unpack_unconstrained(theta, dt=dt)
    mu, kappa, vbar, xi, rho = p.mu, p.kappa, p.vbar, p.xi, p.rho

    y1 = r - mu * dt
    y2 = dv - kappa * (vbar - v_eff) * dt

    one_m_r2 = max(1e-12, 1.0 - rho * rho)
    detC = (xi * xi) * one_m_r2

    inv11 = (xi * xi) / detC
    inv12 = (-xi * rho) / detC
    inv22 = 1.0 / detC

    scale = np.maximum(v_eff * dt, 1e-16)
    quad = (inv11 * y1 * y1 + 2.0 * inv12 * y1 * y2 + inv22 * y2 * y2) / scale
    logdet = 2.0 * np.log(scale) + np.log(detC)
    return float(0.5 * np.sum(logdet + quad))


def fit_sv_params_qmle_from_state(
    state_df: pd.DataFrame,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    price_ticker: str = "SPY",
    maxiter: int = 200,
    require_success: bool = True,
) -> Tuple[SVParams, Dict[str, Any]]:
    """
    Step 2 (PDF): QMLE under the matched daily discretisation using returns + variance proxy only.

    By default this is strict: unsuccessful optimisation raises, rather than silently emitting
    parameters that could contaminate downstream forecasts.
    """
    df = state_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if train_start is not None:
        df = df.loc[pd.to_datetime(train_start) :]
    if train_end is not None:
        df = df.loc[: pd.to_datetime(train_end)]

    if not {"r", "V_proxy"}.issubset(df.columns):
        raise KeyError("State must contain columns: r, V_proxy")

    dt = DT_YEARS

    v = df["V_proxy"].astype(float)
    v1 = v.shift(-1)
    r1 = df["r"].shift(-1).astype(float)

    mask = v.notna() & v1.notna() & r1.notna()
    if not bool(mask.any()):
        raise RuntimeError("No valid (v_i, v_{i+1}, r_{i+1}) observations available for QMLE fit.")

    v = v.loc[mask].to_numpy(dtype=float)
    v1 = v1.loc[mask].to_numpy(dtype=float)
    r = r1.loc[mask].to_numpy(dtype=float)

    dv = v1 - v
    v_eff = effective_variance(v)

    p0 = _initial_guess_moments(v=v, v1=v1, r=r, dt=dt)
    x0 = _pack_unconstrained(p0)

    res = minimize(
        fun=_qmle_negloglik,
        x0=x0,
        args=(r, dv, v_eff, dt),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    p_hat = _unpack_unconstrained(res.x, dt=dt)

    mu, kappa, vbar, xi, rho = p_hat.mu, p_hat.kappa, p_hat.vbar, p_hat.xi, p_hat.rho
    y1 = r - mu * dt
    y2 = dv - kappa * (vbar - v_eff) * dt

    eps1 = y1 / np.sqrt(np.maximum(v_eff * dt, 1e-16))
    eps2 = (y2 / (xi * np.sqrt(np.maximum(v_eff * dt, 1e-16))) - rho * eps1) / np.sqrt(max(1e-12, 1.0 - rho * rho))

    nll = _qmle_negloglik(res.x, r, dv, v_eff, dt)
    diag: Dict[str, Any] = {
        "ticker": price_ticker,
        "train_start": str(df.index[mask][0].date()),
        "train_end": str(df.index[mask][-1].date()),
        "n_steps": int(mask.sum()),
        "negloglik": float(nll),
        "optimiser": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "nit": int(getattr(res, "nit", -1)),
        },
        "moments": {
            "eps1_mean": float(np.mean(eps1)),
            "eps1_var": float(np.var(eps1, ddof=1)),
            "eps2_mean": float(np.mean(eps2)),
            "eps2_var": float(np.var(eps2, ddof=1)),
            "corr_eps1_raw_y2": float(np.corrcoef(eps1, y2)[0, 1]),
        },
    }

    if require_success and not res.success:
        raise RuntimeError(
            "SV QMLE optimisation failed; refusing to emit parameters for downstream forecasting. "
            f"status={res.status}, message={res.message!s}"
        )

    vals = np.array([p_hat.mu, p_hat.kappa, p_hat.vbar, p_hat.xi, p_hat.rho], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"SV QMLE produced non-finite parameter values: {vals}")

    return p_hat, diag


def params_to_jsonable(params: SVParams, diag: Dict[str, Any]) -> Dict[str, Any]:
    return {"params": asdict(params), "diagnostics": diag}
