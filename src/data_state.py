from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import TRADING_DAYS_PER_YEAR, RV_WINDOW_DAYS


@dataclass(frozen=True)
class StateBuildSpec:
    price_col: str = "Adj Close"
    rv_window_days: int = RV_WINDOW_DAYS
    annualisation: int = TRADING_DAYS_PER_YEAR
    min_periods: int | None = None  # if None, use rv_window_days


def _to_numeric_series(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        raise ValueError(f"{name} could not be converted to numeric (all NaN).")
    return out


def compute_log_price(price: pd.Series) -> pd.Series:
    """X_k = log P_k (P_k is adjusted close by default)."""
    p = _to_numeric_series(price, "price")
    p = p.where(p > 0)  # avoid log(<=0)
    return np.log(p)


def compute_log_returns(log_price: pd.Series) -> pd.Series:
    """r_k = X_k - X_{k-1}."""
    return log_price.diff()


def compute_drawdown_exact(log_price: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Exact drawdown on the daily grid:
      M_k = max_{j<=k} X_j
      D_k = M_k - X_k
    This is the *exact* recursion used for simulation/evaluation. 
    """
    m = log_price.cummax()
    d = m - log_price
    return m, d


def compute_drawdown_recursive_from_returns(r: pd.Series, d0: float = 0.0) -> pd.Series:
    """
    Equivalent recursion:
      D_{i+1} = max(0, D_i - r_{i+1}),  where r_{i+1} = x_{i+1} - x_i.
    Useful as a cross-check.
    """
    r = _to_numeric_series(r, "returns")
    out = np.full(len(r), np.nan, dtype=float)
    d_prev = float(d0)
    for i, ri in enumerate(r.values):
        if np.isnan(ri):
            out[i] = np.nan
            continue
        d_prev = max(0.0, d_prev - float(ri))
        out[i] = d_prev
    return pd.Series(out, index=r.index, name="D_recursive")


def compute_drawdown_fraction(drawdown_log: pd.Series) -> pd.Series:
    """
    δ_k = 1 - exp(-D_k).
    """
    return 1.0 - np.exp(-drawdown_log)


def realised_variance_proxy(
    r: pd.Series,
    window_days: int,
    annualisation: int = TRADING_DAYS_PER_YEAR,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Variance proxy (PDF: "rolling realised variance, annualised consistently"). 

    We implement:
      RV_k = annualisation * mean_{window}( r^2 )
    where r are daily log-returns.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive.")
    if min_periods is None:
        min_periods = window_days
    r2 = _to_numeric_series(r, "returns").pow(2)
    rv = annualisation * r2.rolling(window=window_days, min_periods=min_periods).mean()
    rv.name = f"V_proxy_rv{window_days}"
    return rv


def build_state_frame(raw: pd.DataFrame, spec: StateBuildSpec = StateBuildSpec()) -> Tuple[pd.DataFrame, Dict]:
    """
    Build the Step-1 state table:
      X_k (log price), r_k (log return), D_k (drawdown), V_k proxy.

    Returns:
      (state_df, metadata_dict)
    """
    df = raw.copy()

    # Expect Date either as index or column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

    if spec.price_col not in df.columns:
        # fallback
        if "Close" in df.columns:
            price_col = "Close"
        else:
            raise KeyError(f"Missing price column {spec.price_col!r} (and no 'Close' fallback).")
    else:
        price_col = spec.price_col

    price = df[price_col]
    X = compute_log_price(price)
    r = compute_log_returns(X)
    M, D = compute_drawdown_exact(X)
    delta = compute_drawdown_fraction(D)

    V_proxy = realised_variance_proxy(
        r=r,
        window_days=spec.rv_window_days,
        annualisation=spec.annualisation,
        min_periods=spec.min_periods,
    )

    # Cross-check exact drawdown vs equivalent recursion (ignoring the first NaN return)
    D_rec = compute_drawdown_recursive_from_returns(r.fillna(0.0), d0=float(D.iloc[0]) if pd.notna(D.iloc[0]) else 0.0)
    # D_rec uses r[0]=0 in the fill; align to D for comparison
    mask = D.notna() & D_rec.notna()
    if mask.any():
        max_abs_err = float(np.max(np.abs((D[mask] - D_rec[mask]).values)))
        if max_abs_err > 1e-10:
            raise RuntimeError(f"Drawdown recursion mismatch: max_abs_err={max_abs_err:g}")

    out = pd.DataFrame(
        {
            "P": _to_numeric_series(price, "price"),
            "X": X,
            "r": r,
            "M": M,
            "D": D,
            "delta": delta,
            "V_proxy": V_proxy,
        },
        index=df.index,
    )

    meta = {
        "price_col_used": price_col,
        "rv_window_days": spec.rv_window_days,
        "annualisation": spec.annualisation,
        "min_periods": spec.min_periods if spec.min_periods is not None else spec.rv_window_days,
        "notes": {
            "X": "X_k = log P_k (daily adjusted close by default)",
            "r": "r_k = X_k - X_{k-1}",
            "M_D": "M_k = max_{j<=k} X_j; D_k = M_k - X_k (exact daily-grid drawdown)",
            "delta": "delta_k = 1 - exp(-D_k)",
            "V_proxy": "rolling realised variance proxy, annualised",
        },
    }
    return out, meta