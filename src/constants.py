from __future__ import annotations

import math
from typing import Final, Tuple

# ---- Global conventions (PDF: daily grid) ----
TRADING_DAYS_PER_YEAR: Final[int] = 252
DT_YEARS: Final[float] = 1.0 / TRADING_DAYS_PER_YEAR  # Δt := 1/252 years

# ---- Committed query grid (used later, but defined now) ----
DELTA_GRID: Final[Tuple[float, ...]] = (0.20, 0.30)    # δ ∈ {0.20, 0.30}
TDAYS_GRID: Final[Tuple[int, ...]] = (20, 60)          # Tdays ∈ {20, 60}


def log_drawdown_threshold(delta: float) -> float:
    """Convert price drawdown fraction δ to log threshold d = -log(1-δ)."""
    if not (0.0 <= delta < 1.0):
        raise ValueError("delta must be in [0, 1).")
    return -math.log(1.0 - delta)


D_GRID: Final[Tuple[float, ...]] = tuple(log_drawdown_threshold(d) for d in DELTA_GRID)

# ---- Pre-registered numerical params for later stages (not used in Step 1 yet) ----
V_MIN: Final[float] = 1e-6       # variance floor vmin = 10^-6
ALPHA_V: Final[float] = 50.0     # softplus sharpness αv = 50
ALPHA_M: Final[float] = 50.0     # running-max surrogate start αm = 50

# ---- Step 1 variance proxy default (PDF doesn't pin a window; we choose a sane default) ----
RV_WINDOW_DAYS: Final[int] = 20

# --- IS stability params (PDF starting values) ---
LAMBDA_IS = 0.5   # start with λ = 0.5
UMAX = 10.0       # umax in a practical starting range [5,15]

# --- Optimiser tolerance for "first-hit" constraint ---
HIT_EPS = 1e-4    # enforce D~_i <= d - eps for i<tau (operational note)