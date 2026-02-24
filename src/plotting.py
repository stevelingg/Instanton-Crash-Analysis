from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl


from contextlib import contextmanager

@contextmanager
def _plot_context(spec: PlotSpec):
    with plt.style.context(spec.style):
        with mpl.rc_context(rc=_rcparams_for_spec(spec)):
            yield


def _contiguous_true_spans(dates: pd.Series, mask: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_span = False
    start_i = 0
    for i, m in enumerate(mask):
        if m and not in_span:
            in_span = True
            start_i = i
        if (not m) and in_span:
            in_span = False
            spans.append((pd.to_datetime(dates.iloc[start_i]), pd.to_datetime(dates.iloc[i - 1])))
    if in_span:
        spans.append((pd.to_datetime(dates.iloc[start_i]), pd.to_datetime(dates.iloc[-1])))
    return spans


def _shade_tau0(ax: plt.Axes, dates: pd.Series, tau0_mask: np.ndarray, alpha: float = 0.10) -> None:
    # Under the PDF event definition max_{0<=i<=n} D_i >= d, tau*=0 corresponds to d0>=d
    # so the event is already true at the origin and p_hat==1.
    spans = _contiguous_true_spans(dates.reset_index(drop=True), tau0_mask)
    first = True
    for a, b in spans:
        ax.axvspan(a, b, alpha=alpha, label=(r"$\tau^*=0$ (already hit at origin: $\hat p=1$)" if first else None))
        first = False


@dataclass(frozen=True)
class PlotSpec:
    """
    Plot styling spec.

    Defaults aim for clean, publication-style figures without introducing any
    additional dependencies beyond matplotlib.
    """
    dpi: int = 200
    figsize: Tuple[float, float] = (10.0, 4.0)
    style: str = "seaborn-v0_8-whitegrid"
    font_size: int = 11
    title_size: int = 12
    label_size: int = 11
    tick_size: int = 10
    line_width: float = 1.6
    ci_alpha: float = 0.20
    grid_alpha: float = 0.35
    savefig_kwargs: Dict[str, object] = None  # type: ignore[assignment]


def _rcparams_for_spec(spec: PlotSpec) -> dict:
    return {
        "font.size": spec.font_size,
        "axes.titlesize": spec.title_size,
        "axes.labelsize": spec.label_size,
        "xtick.labelsize": spec.tick_size,
        "ytick.labelsize": spec.tick_size,
        "lines.linewidth": spec.line_width,
        "axes.grid": True,
        "grid.alpha": spec.grid_alpha,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": spec.dpi,
    }


def _savefig(fig: plt.Figure, out_path: Path, spec: PlotSpec) -> None:
    kwargs = {"bbox_inches": "tight"}
    if spec.savefig_kwargs:
        kwargs.update(dict(spec.savefig_kwargs))
    fig.savefig(out_path, **kwargs)


def _format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_forecast_timeseries(
    df: pd.DataFrame,
    *,
    tdays: int,
    delta: float,
    out_path: str | Path,
    spec: PlotSpec = PlotSpec(),
) -> Path:
    """
    Plot p_hat over time for a single (Tdays, delta) query.
    If available, overlay realised label y in {0,1}.

    Expected columns in df:
      Date, Tdays, delta, p_hat, se (optional), y (optional)
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date")

    d = d[(d["Tdays"].astype(int) == int(tdays)) & (np.isclose(d["delta"].astype(float), float(delta)))]
    if d.empty:
        raise ValueError(f"No rows for Tdays={tdays}, delta={delta}")

    out_path = _ensure_parent(out_path)

    with _plot_context(spec):
        if "y" in d.columns:
            fig, (ax, ax_y) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(spec.figsize[0], max(4.8, spec.figsize[1] * 1.25)),
                dpi=spec.dpi,
                gridspec_kw={"height_ratios": [3.0, 1.0]},
            )
        else:
            fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)
            ax_y = None

        date = d["Date"].reset_index(drop=True)
        p = d["p_hat"].astype(float).to_numpy()
        
        if "tau_star" in d.columns:
            tau0 = (d["tau_star"].fillna(0).astype(int).to_numpy() == 0)
            _shade_tau0(ax, date, tau0, alpha=0.10)

        ax.plot(date, p, label=r"$\hat p$")

        ax.set_title(f"Forecast probability over time (T={int(tdays)} days, δ={float(delta):.2f})")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        # Prevent matplotlib from expanding the x-range beyond the available dates.
        ax.set_xlim(date.min(), date.max())
        ax.margins(x=0)
        ax.legend(loc="best")

        if ax_y is not None:
            y = d["y"].astype(float).to_numpy()
            mask = np.isfinite(y)
            if mask.any():
                ax_y.scatter(date.to_numpy()[mask], y[mask], s=10, alpha=0.6)
            ax_y.set_ylabel("Realised")
            ax_y.set_yticks([0.0, 1.0])
            ax_y.set_ylim(-0.10, 1.10)
            ax_y.set_xlabel("Date")
            ax_y.set_xlim(date.min(), date.max())
            ax_y.margins(x=0)
            _format_date_axis(ax_y)
        else:
            ax.set_xlabel("Date")
            _format_date_axis(ax)

        fig.tight_layout()
        _savefig(fig, out_path, spec)
        plt.close(fig)
    return out_path


def plot_reliability(
    rel: pd.DataFrame,
    *,
    out_path: str | Path,
    spec: PlotSpec = PlotSpec(figsize=(6.2, 6.2)),
    tdays: int | None = None,
    delta: float | None = None,
    show_binomial_se: bool = True,
) -> Path:
    """
    Reliability / calibration diagram from reliability_bins() output.

    Expected columns:
      bin_lo, bin_hi, count, p_mean, y_mean
    """
    out_path = _ensure_parent(out_path)

    r = rel.copy()

    # Optional filtering for grouped reliability outputs.
    if {"Tdays", "delta"}.issubset(r.columns) and (tdays is not None or delta is not None):
        if tdays is not None:
            r = r[r["Tdays"].astype(int) == int(tdays)]
        if delta is not None:
            r = r[np.isclose(r["delta"].astype(float), float(delta))]

    if r.empty:
        # create an empty plot for consistency
        fig = plt.figure(figsize=spec.figsize, dpi=spec.dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Reliability (empty)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    with _plot_context(spec):
        fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)

        ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="ideal")

        has_groups = {"Tdays", "delta"}.issubset(r.columns)
        if has_groups:
            groups = list(r.groupby(["Tdays", "delta"], sort=True))
        else:
            groups = [((None, None), r)]

        for (T, dlt), sub in groups:
            x = sub["p_mean"].to_numpy(dtype=float)
            y = sub["y_mean"].to_numpy(dtype=float)
            c = sub["count"].to_numpy(dtype=float)

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c) & (c > 0)
            if not np.any(mask):
                continue

            # Sane marker scaling: use sqrt(count) and cap sizes.
            c_eff = np.sqrt(c[mask])
            c_eff = c_eff / max(1e-12, float(np.max(c_eff)))
            sizes = 25.0 + 175.0 * c_eff

            label = "bins" if (T is None and dlt is None) else f"T={int(T)}d, δ={float(dlt):.2f}"
            ax.scatter(x[mask], y[mask], s=sizes, alpha=0.85, label=label)

            if show_binomial_se:
                # Approx SE for empirical rate: sqrt(y(1-y)/n)
                se = np.sqrt(np.clip(y[mask] * (1.0 - y[mask]) / c[mask], 0.0, np.inf))
                ax.errorbar(x[mask], y[mask], yerr=1.96 * se, fmt="none", elinewidth=1.0, alpha=0.55)

            # Optional: show bin extents as faint horizontal segments.
            if {"bin_lo", "bin_hi"}.issubset(sub.columns):
                lo = sub["bin_lo"].to_numpy(dtype=float)[mask]
                hi = sub["bin_hi"].to_numpy(dtype=float)[mask]
                for yv, lov, hiv in zip(y[mask], lo, hi):
                    ax.plot([lov, hiv], [yv, yv], color="0.7", linewidth=1.0, alpha=0.45)

        ax.set_title("Reliability diagram")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical hit rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        _savefig(fig, out_path, spec)
        plt.close(fig)
    return out_path


def plot_ess_timeseries(
    df: pd.DataFrame,
    *,
    tdays: int,
    delta: float,
    out_path: str | Path,
    spec: PlotSpec = PlotSpec(),
) -> Path:
    """
    Plot ESS over time for a single (Tdays, delta) query.

    Expected columns:
      Date, Tdays, delta, ess
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date")

    d = d[(d["Tdays"].astype(int) == int(tdays)) & (np.isclose(d["delta"].astype(float), float(delta)))]
    if d.empty:
        raise ValueError(f"No rows for Tdays={tdays}, delta={delta}")

    if "ess" not in d.columns:
        raise KeyError("df missing ess column")

    out_path = _ensure_parent(out_path)

    with _plot_context(spec):
        fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)

        if "tau_star" in d.columns:
            tau0 = (d["tau_star"].fillna(0).astype(int).to_numpy() == 0)
            _shade_tau0(ax, d["Date"].reset_index(drop=True), tau0, alpha=0.10)

        ax.plot(d["Date"], d["ess"].astype(float), label="ESS")
        ax.set_title(f"Effective sample size (T={int(tdays)} days, δ={float(delta):.2f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("ESS")
        ax.set_xlim(d["Date"].min(), d["Date"].max())
        ax.margins(x=0)
        _format_date_axis(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        _savefig(fig, out_path, spec)
        plt.close(fig)
    return out_path


def plot_weight_diagnostics_timeseries(
    df: pd.DataFrame,
    *,
    tdays: int,
    delta: float,
    out_path: str | Path,
    spec: PlotSpec = PlotSpec(),
) -> Path:
    """
    Plot max_weight and mean_weight over time for a single (Tdays, delta) query.

    Expected columns:
      Date, Tdays, delta, mean_weight, max_weight
    """
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date")

    d = d[(d["Tdays"].astype(int) == int(tdays)) & (np.isclose(d["delta"].astype(float), float(delta)))]
    if d.empty:
        raise ValueError(f"No rows for Tdays={tdays}, delta={delta}")

    for col in ("mean_weight", "max_weight"):
        if col not in d.columns:
            raise KeyError(f"df missing {col} column")

    out_path = _ensure_parent(out_path)

    with _plot_context(spec):
        fig, ax = plt.subplots(figsize=spec.figsize, dpi=spec.dpi)

        if "tau_star" in d.columns:
            tau0 = (d["tau_star"].fillna(0).astype(int).to_numpy() == 0)
            _shade_tau0(ax, d["Date"].reset_index(drop=True), tau0, alpha=0.10)

        ax.plot(d["Date"], d["mean_weight"].astype(float), label="mean weight")
        ax.plot(d["Date"], d["max_weight"].astype(float), label="max weight")
        ax.set_title(f"IS weight diagnostics (T={int(tdays)} days, δ={float(delta):.2f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.set_xlim(d["Date"].min(), d["Date"].max())
        ax.margins(x=0)
        # A log scale is typically the clearest way to show max vs mean.
        if np.all(d["mean_weight"].astype(float) > 0) and np.all(d["max_weight"].astype(float) > 0):
            ax.set_yscale("log")
        _format_date_axis(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        _savefig(fig, out_path, spec)
        plt.close(fig)
    return out_path


def plot_instanton_paths(
    npz_path: str | Path,
    *,
    out_path: str | Path,
    spec: PlotSpec = PlotSpec(figsize=(10.0, 6.0)),
) -> Path:
    """
    Plot an instanton path (x_path and v_path) from a cached .npz file saved by the forecast script.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=False)
    x = np.asarray(data["x_path"], dtype=float)
    v = np.asarray(data["v_path"], dtype=float)
    tau = int(data["tau"])
    action = float(data["action"])

    out_path = _ensure_parent(out_path)

    with _plot_context(spec):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=spec.figsize, dpi=spec.dpi, sharex=True)
        ax1.plot(np.arange(len(x)), x)
        ax1.set_title(f"Instanton path (τ={tau}, action={action:.6g})")
        ax1.set_ylabel("x")

        ax2.plot(np.arange(len(v)), v)
        ax2.set_ylabel("v")
        ax2.set_xlabel("step")

        fig.tight_layout()
        _savefig(fig, out_path, spec)
        plt.close(fig)
    return out_path