from __future__ import annotations

import numpy as np

from .constants import V_MIN, ALPHA_V


def softplus_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    softplus_alpha(x) = (1/alpha) * log(1 + exp(alpha * x))
    Implemented with numerical stability.
    """
    x = np.asarray(x, dtype=float)
    ax = alpha * x
    # log(1+exp(ax)) stable form
    return (1.0 / alpha) * np.log1p(np.exp(-np.abs(ax))) + np.maximum(ax, 0.0) / alpha


def effective_variance(v: np.ndarray, vmin: float = V_MIN, alpha_v: float = ALPHA_V) -> np.ndarray:
    """
    v_eff := g(v) = vmin + softplus_alpha(v - vmin, alpha_v)
    Matches the PDF definition exactly.
    """
    v = np.asarray(v, dtype=float)
    return vmin + softplus_alpha(v - vmin, alpha_v)