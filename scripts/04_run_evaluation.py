from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root on sys.path BEFORE importing src.*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from datetime import datetime, timezone
from typing import Optional

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import src.plotting as plotting_mod

from src.constants import log_drawdown_threshold
from src.evaluation import (
    add_realised_labels_to_forecasts,
    score_forecasts,
    score_forecasts_by_group,
    reliability_bins,
    reliability_bins_by_group,
    simulate_naive_mc,
    compare_estimates_zscore,
)
from src.importance_sampling import simulate_is_paths, estimate_probability_is
from src.io import (
    ensure_dir, 
    find_latest_file, 
    save_json,
    read_state_csv,
    read_forecasts_csv,
    load_params_json,
    load_u_from_cached_instanton,
    maybe_load_forecast_metadata,
)
from src.plotting import (
    plot_forecast_timeseries,
    plot_reliability,
    plot_ess_timeseries,
    plot_weight_diagnostics_timeseries,
)


# -----------------------------
# Input resolution / run layout
# -----------------------------
def _maybe_infer_inputs_from_run_dir(run_dir: Path) -> tuple[Path, Path, Path]:
    """
    Given a run directory produced by script 03, infer:
      forecasts_csv, instantons_dir, diagnostics_dir
    """
    forecasts_dir = run_dir / "forecasts"
    instantons_dir = run_dir / "instantons"
    diagnostics_dir = run_dir / "diagnostics"

    if not forecasts_dir.exists():
        raise FileNotFoundError(f"run_dir has no forecasts/: {forecasts_dir}")

    try:
        forecasts_csv = find_latest_file(forecasts_dir, "*forecasts_*.csv")
    except FileNotFoundError:
        forecasts_csv = find_latest_file(forecasts_dir, "*.csv")

    return forecasts_csv, instantons_dir, diagnostics_dir


def _infer_default_out_dir_from_forecasts(forecasts_csv: Path) -> Optional[Path]:
    # If forecasts path looks like .../<run_dir>/forecasts/<file>.csv, use sibling diagnostics
    p = forecasts_csv.resolve()
    if p.parent.name == "forecasts":
        return p.parent.parent / "diagnostics"
    return None


def _pick_latest_or_none(directory: Path, pattern: str) -> Optional[Path]:
    try:
        return find_latest_file(directory, pattern)
    except FileNotFoundError:
        return None


def _resolve_inputs(args: argparse.Namespace) -> dict:
    """
    Resolve forecasts/state/params/instantons/out_dir with preference order:
      explicit CLI > run_dir inference > forecasts metadata > latest-in-directory fallback
    """
    meta = None

    # Forecasts + instantons + out_dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
        forecasts_csv, instantons_dir_default, out_dir_default = _maybe_infer_inputs_from_run_dir(run_dir)
        out_dir = ensure_dir(out_dir_default if args.out_dir is None else Path(args.out_dir))
    else:
        run_dir = None
        if args.forecasts_csv:
            forecasts_csv = Path(args.forecasts_csv)
        else:
            forecasts_csv = find_latest_file(Path(args.forecasts_dir), args.forecasts_pattern)
        instantons_dir_default = Path(args.instantons_dir) if args.instantons_dir else Path("outputs/instantons")
        inferred_out = _infer_default_out_dir_from_forecasts(forecasts_csv)
        out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (inferred_out or Path("outputs/diagnostics")))

    meta = maybe_load_forecast_metadata(forecasts_csv)

    # State CSV
    if args.state_csv:
        state_csv = Path(args.state_csv)
    else:
        state_from_meta = None
        if isinstance(meta, dict):
            state_from_meta = meta.get("inputs", {}).get("state_source_csv", None)
        if state_from_meta:
            state_csv = Path(state_from_meta)
        else:
            state_csv = find_latest_file(Path(args.state_dir), args.state_pattern)

    # Params JSON (needed only for naive checks; still resolve now for consistency)
    if args.params_json:
        params_json = Path(args.params_json)
    else:
        params_from_meta = None
        if isinstance(meta, dict):
            params_from_meta = meta.get("inputs", {}).get("params_json", None)
        if params_from_meta:
            params_json = Path(params_from_meta)
        else:
            params_json = find_latest_file(Path(args.params_dir), args.params_pattern)

    # Instantons dir
    if args.instantons_dir:
        instantons_dir = Path(args.instantons_dir)
    else:
        inst_from_meta = None
        if isinstance(meta, dict):
            inst_from_meta = meta.get("outputs", {}).get("instantons_dir", None)
        instantons_dir = Path(inst_from_meta) if inst_from_meta else instantons_dir_default

    return {
        "run_dir": run_dir,
        "forecasts_csv": forecasts_csv,
        "state_csv": state_csv,
        "params_json": params_json,
        "instantons_dir": instantons_dir,
        "out_dir": out_dir,
        "forecast_meta": meta,
    }


# -----------------------------
# Calibration / uncertainty utils
# -----------------------------
def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n)) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def _enrich_reliability_bins(rel: pd.DataFrame) -> pd.DataFrame:
    """
    Add Wilson CI + calibration gaps to reliability output.
    Assumes columns include: count, p_mean, y_mean, and optionally group columns.
    """
    r = rel.copy()
    if r.empty:
        return r

    # Note: reliability tables include empty bins with count=0 and y_mean=NaN.
    # Casting NaNs to int triggers RuntimeWarning and creates bogus sentinel ints.
    # Keep as float and only convert to int after finiteness / non-empty checks.
    count_f = pd.to_numeric(r["count"], errors="coerce").to_numpy(dtype=float)
    y_mean_f = pd.to_numeric(r["y_mean"], errors="coerce").to_numpy(dtype=float)
    k_f = np.round(y_mean_f * count_f)

    ci_lo = np.full(len(r), np.nan, dtype=float)
    ci_hi = np.full(len(r), np.nan, dtype=float)
    for i in range(len(r)):
        n_i_f = float(count_f[i]) if i < len(count_f) else np.nan
        k_i_f = float(k_f[i]) if i < len(k_f) else np.nan
        if not (np.isfinite(n_i_f) and np.isfinite(k_i_f)):
            continue
        if n_i_f <= 0:
            continue
        lo, hi = _wilson_interval(int(k_i_f), int(n_i_f))
        ci_lo[i], ci_hi[i] = lo, hi

    r["y_wilson_lo"] = ci_lo
    r["y_wilson_hi"] = ci_hi
    r["abs_calib_gap"] = np.abs(r["p_mean"].astype(float) - r["y_mean"].astype(float))
    r["sq_calib_gap"] = (r["p_mean"].astype(float) - r["y_mean"].astype(float)) ** 2
    return r


def _calibration_summary_from_reliability(
    rel: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
) -> pd.DataFrame:
    """
    Summarise calibration by group using binned reliability table:
      ECE = sum_b w_b |p_mean - y_mean|
      MCE = max_b |p_mean - y_mean|
    """
    if rel.empty:
        cols = list(group_cols) + ["n_bins_nonempty", "n_obs", "ece", "mce", "rmsce"]
        return pd.DataFrame(columns=cols)

    r = rel.copy()
    req = {"count", "p_mean", "y_mean"}
    if not req.issubset(r.columns):
        raise KeyError(f"Reliability table missing columns: {req - set(r.columns)}")

    if not group_cols:
        groups = [((None,), r)]
    else:
        groups = list(r.groupby(list(group_cols), sort=True))

    rows: list[dict] = []
    for keys, sub in groups:
        s = sub.copy()
        s = s[s["count"].astype(float) > 0]
        if s.empty:
            row = {"n_bins_nonempty": 0, "n_obs": 0, "ece": np.nan, "mce": np.nan, "rmsce": np.nan}
        else:
            count = s["count"].astype(float).to_numpy()
            gap = np.abs(s["p_mean"].astype(float).to_numpy() - s["y_mean"].astype(float).to_numpy())
            w = count / np.sum(count)
            row = {
                "n_bins_nonempty": int(len(s)),
                "n_obs": int(np.sum(count)),
                "ece": float(np.sum(w * gap)),
                "mce": float(np.max(gap)),
                "rmsce": float(np.sqrt(np.sum(w * gap * gap))),
            }

        if group_cols:
            keys_t = keys if isinstance(keys, tuple) else (keys,)
            row = {**{c: v for c, v in zip(group_cols, keys_t)}, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _summarise_is_diagnostics(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
) -> pd.DataFrame:
    """
    Summaries of MC/IS quality using forecast outputs.
    """
    # In unit tests / lightweight runs, forecasts may only have (Date, Tdays, delta, p_hat).
    # Treat IS diagnostics as optional: if missing, populate with NaN and still produce a summary table.
    needed = {"p_hat", "se", "ess", "N", "mean_weight", "max_weight", "hit_rate_Q"}
    d = df.copy()
    for col in needed:
        if col not in d.columns:
            d[col] = np.nan

    d["N"] = pd.to_numeric(d["N"], errors="coerce")
    d["ess"] = pd.to_numeric(d["ess"], errors="coerce")
    d["se"] = pd.to_numeric(d["se"], errors="coerce")
    d["p_hat"] = pd.to_numeric(d["p_hat"], errors="coerce")
    d["mean_weight"] = pd.to_numeric(d["mean_weight"], errors="coerce")
    d["max_weight"] = pd.to_numeric(d["max_weight"], errors="coerce")
    d["hit_rate_Q"] = pd.to_numeric(d["hit_rate_Q"], errors="coerce")

    # Derived metrics (stay NaN if inputs missing)
    d["ess_frac"] = d["ess"] / d["N"]
    d["rel_se"] = d["se"] / np.clip(d["p_hat"], 1e-12, np.inf)
    d["weight_ratio_max_mean"] = d["max_weight"] / np.clip(d["mean_weight"], 1e-12, np.inf)

    num_cols = [
        "p_hat", "se", "rel_se", "ess", "ess_frac", "mean_weight", "max_weight",
        "weight_ratio_max_mean", "hit_rate_Q"
    ]

    def _agg(sub: pd.DataFrame) -> dict:
        out: dict = {"n_rows": int(len(sub))}
        for c in num_cols:
            x = pd.to_numeric(sub[c], errors="coerce")
            out[f"{c}_mean"] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
            out[f"{c}_median"] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
            out[f"{c}_q10"] = float(np.nanquantile(x, 0.10)) if np.isfinite(x).any() else np.nan
            out[f"{c}_q90"] = float(np.nanquantile(x, 0.90)) if np.isfinite(x).any() else np.nan
        return out

    rows: list[dict] = []
    rows.append({"scope": "overall", **_agg(d)})

    if group_cols:
        for keys, sub in d.groupby(list(group_cols), sort=True):
            keys_t = keys if isinstance(keys, tuple) else (keys,)
            row = {"scope": "group", **{c: v for c, v in zip(group_cols, keys_t)}, **_agg(sub)}
            rows.append(row)

    return pd.DataFrame(rows)


def _date_cluster_bootstrap_scores(
    with_labels: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap over dates (clusters): resample dates with replacement, keep all query rows for each sampled date.

    Returns:
      overall_summary_df, by_group_summary_df
    """
    if n_boot <= 0:
        return (
            pd.DataFrame(columns=["metric", "mean", "q025", "q500", "q975", "n_boot"]),
            pd.DataFrame(columns=[*group_cols, "metric", "mean", "q025", "q500", "q975", "n_boot"]),
        )

    d = with_labels.copy()
    d = d[np.isfinite(pd.to_numeric(d["y"], errors="coerce"))].copy()
    if d.empty:
        return (
            pd.DataFrame(columns=["metric", "mean", "q025", "q500", "q975", "n_boot"]),
            pd.DataFrame(columns=[*group_cols, "metric", "mean", "q025", "q500", "q975", "n_boot"]),
        )

    d["Date"] = pd.to_datetime(d["Date"])
    dates = pd.Index(sorted(d["Date"].dropna().unique()))
    if len(dates) == 0:
        return (
            pd.DataFrame(columns=["metric", "mean", "q025", "q500", "q975", "n_boot"]),
            pd.DataFrame(columns=[*group_cols, "metric", "mean", "q025", "q500", "q975", "n_boot"]),
        )

    blocks = [d[d["Date"] == dt].copy() for dt in dates]
    rng = np.random.default_rng(int(seed))

    overall_recs: list[dict] = []
    by_group_recs: list[dict] = []

    for b in range(int(n_boot)):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        boot_df = pd.concat([blocks[i] for i in picks], ignore_index=True)

        s = score_forecasts(boot_df, p_col="p_hat", y_col="y")
        overall_recs.append(
            {
                "boot": b,
                "n": int(s.get("n", 0)),
                "mean_log_score": float(s.get("mean_log_score", np.nan)),
                "mean_brier": float(s.get("mean_brier", np.nan)),
            }
        )

        sb = score_forecasts_by_group(boot_df, group_cols=group_cols, p_col="p_hat", y_col="y")
        if not sb.empty:
            sb = sb.copy()
            sb["boot"] = int(b)
            by_group_recs.extend(sb.to_dict(orient="records"))

    overall_boot = pd.DataFrame(overall_recs)
    by_group_boot = pd.DataFrame(by_group_recs)

    def _summ(df_in: pd.DataFrame, metrics: list[str], group_cols_local: tuple[str, ...]) -> pd.DataFrame:
        rows: list[dict] = []
        if df_in.empty:
            return pd.DataFrame(columns=[*group_cols_local, "metric", "mean", "q025", "q500", "q975", "n_boot"])
        if group_cols_local:
            groups = df_in.groupby(list(group_cols_local), sort=True)
        else:
            groups = [((), df_in)]

        for keys, sub in groups:
            keys_t = keys if isinstance(keys, tuple) else ((keys,) if group_cols_local else tuple())
            for m in metrics:
                x = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                row = {
                    "metric": m,
                    "mean": float(np.mean(x)) if len(x) else np.nan,
                    "q025": float(np.quantile(x, 0.025)) if len(x) else np.nan,
                    "q500": float(np.quantile(x, 0.50)) if len(x) else np.nan,
                    "q975": float(np.quantile(x, 0.975)) if len(x) else np.nan,
                    "n_boot": int(len(x)),
                }
                for c, v in zip(group_cols_local, keys_t):
                    row[c] = v
                rows.append(row)
        cols = [*group_cols_local, "metric", "mean", "q025", "q500", "q975", "n_boot"]
        return pd.DataFrame(rows, columns=cols)

    overall_summary = _summ(overall_boot, ["mean_log_score", "mean_brier"], tuple())
    by_group_summary = _summ(by_group_boot, ["mean_log_score", "mean_brier"], tuple(group_cols))

    return overall_summary, by_group_summary


# -----------------------------
# Naive-vs-IS extensive checks
# -----------------------------
def _parse_prob_bands(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("probability bands must contain at least two comma-separated values")
    if sorted(vals) != vals:
        raise ValueError("probability bands must be sorted ascending")
    if vals[0] < 0 or vals[-1] > 1:
        raise ValueError("probability bands must lie in [0,1]")
    return vals


def _select_rows_for_naive_checks(
    forecasts: pd.DataFrame,
    *,
    mode: str,
    p_lo: float,
    p_hi: float,
    max_total: int,
    max_per_stratum: int,
    prob_bands: list[float],
    naive_n: int,
    min_expected_hits: float,
) -> pd.DataFrame:
    """
    Richer selection than "head(max_checks)".
    """
    df = forecasts.copy()
    if "tau_star" in df.columns:
        df = df[df["tau_star"].fillna(0).astype(int) > 0]
    df = df[np.isfinite(pd.to_numeric(df["p_hat"], errors="coerce"))]
    df = df[(df["p_hat"].astype(float) >= p_lo) & (df["p_hat"].astype(float) <= p_hi)]
    df = df.sort_values(["Date", "Tdays", "delta"]).copy()

    if df.empty:
        return df

    mode = str(mode).lower()

    if mode == "sample":
        return df.head(int(max_total)).copy()

    if mode == "all_feasible":
        # Heuristic feasibility: naive expected hits should not be too tiny.
        df = df[(float(naive_n) * df["p_hat"].astype(float)) >= float(min_expected_hits)]
        if max_total > 0:
            df = df.head(int(max_total))
        return df.copy()

    if mode == "stratified":
        # Stratify by (Tdays, delta, probability band)
        edges = np.asarray(prob_bands, dtype=float)
        p = df["p_hat"].astype(float).to_numpy()
        band = np.digitize(p, edges, right=False) - 1
        band = np.clip(band, 0, len(edges) - 2)
        df["p_band_idx"] = band
        df["p_band"] = [f"[{edges[i]:.3g},{edges[i+1]:.3g})" for i in band]

        # Optional feasibility filter
        df = df[(float(naive_n) * df["p_hat"].astype(float)) >= float(min_expected_hits)]

        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(["Tdays", "delta", "p_band_idx"], sort=True):
            parts.append(sub.head(int(max_per_stratum)))
        if not parts:
            return df.head(0).copy()

        out = pd.concat(parts, ignore_index=False).sort_values(["Date", "Tdays", "delta"]).copy()
        if max_total > 0 and len(out) > int(max_total):
            out = out.head(int(max_total))
        return out

    raise ValueError("naive_check_mode must be one of: sample, stratified, all_feasible")


def _run_naive_vs_is_checks(
    *,
    eval_df: pd.DataFrame,
    state: pd.DataFrame,
    params_json: Path,
    instantons_dir: Path,
    p_lo: float,
    p_hi: float,
    naive_check_mode: str,
    naive_prob_bands: list[float],
    naive_max_total: int,
    naive_max_per_stratum: int,
    naive_n: int,
    is_n: int,
    min_expected_hits: float,
    seed_is: int,
    seed_mc: int,
) -> tuple[pd.DataFrame, dict]:
    params = load_params_json(params_json)

    sel = _select_rows_for_naive_checks(
        eval_df,
        mode=naive_check_mode,
        p_lo=float(p_lo),
        p_hi=float(p_hi),
        max_total=int(naive_max_total),
        max_per_stratum=int(naive_max_per_stratum),
        prob_bands=list(naive_prob_bands),
        naive_n=int(naive_n),
        min_expected_hits=float(min_expected_hits),
    )

    checks_rows: list[dict] = []
    for j, r in sel.reset_index(drop=True).iterrows():
        date = pd.to_datetime(r["Date"])
        date_str = date.strftime("%Y-%m-%d")
        tdays = int(r["Tdays"])
        delta = float(r["delta"])
        d = float(log_drawdown_threshold(delta))

        if date not in state.index:
            continue
        srow = state.loc[date]
        x0, v0, d0 = float(srow["X"]), float(srow["V_proxy"]), float(srow["D"])

        u = load_u_from_cached_instanton(
            Path(instantons_dir),
            r,
            lam=float(r["lam"]),
            umax=float(r["umax"]),
        )

        # Independent diagnostic IS rerun
        hit_is, L_is = simulate_is_paths(
            params=params,
            x0=x0, v0=v0, d0=d0,
            d_thresh=d,
            n_steps=tdays,
            u=u,
            n_paths=int(is_n),
            seed=int(seed_is + j),
        )
        is_res = estimate_probability_is(hit_is, L_is)

        # Naive MC baseline
        mc_res = simulate_naive_mc(
            params=params,
            x0=x0, v0=v0, d0=d0,
            d_thresh=d,
            n_steps=tdays,
            n_paths=int(naive_n),
            seed=int(seed_mc + j),
        )

        z = compare_estimates_zscore(is_res.p_hat, is_res.se, mc_res.p_hat, mc_res.se)

        checks_rows.append(
            {
                "Date": date_str,
                "Tdays": tdays,
                "delta": delta,
                "p_hat_forecast": float(r["p_hat"]),
                "se_forecast": float(r["se"]),
                "is_n": int(is_n),
                "mc_n": int(naive_n),
                "p_is": float(is_res.p_hat),
                "se_is": float(is_res.se),
                "p_mc": float(mc_res.p_hat),
                "se_mc": float(mc_res.se),
                "z": float(z),
                "abs_z": float(abs(z)),
                "ess_is": float(is_res.ess),
                "ess_frac_is": float(is_res.ess / max(1.0, float(is_n))),
                "max_weight_is": float(is_res.max_weight),
                "rel_diff_is_mc": float((is_res.p_hat - mc_res.p_hat) / max(1e-12, mc_res.p_hat)) if mc_res.p_hat > 0 else np.nan,
            }
        )

    checks = pd.DataFrame(checks_rows)
    if checks.empty:
        summary = {
            "n_checks": 0,
            "z_mean": np.nan,
            "z_std": np.nan,
            "frac_abs_z_gt_2": np.nan,
            "frac_abs_z_gt_3": np.nan,
        }
        return checks, summary

    z = checks["z"].astype(float).to_numpy()
    absz = np.abs(z)
    summary = {
        "n_checks": int(len(checks)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1)) if len(z) > 1 else 0.0,
        "z_median": float(np.median(z)),
        "abs_z_median": float(np.median(absz)),
        "frac_abs_z_gt_2": float(np.mean(absz > 2.0)),
        "frac_abs_z_gt_3": float(np.mean(absz > 3.0)),
        "mean_abs_rel_diff_is_mc": float(np.nanmean(np.abs(checks["rel_diff_is_mc"].to_numpy(dtype=float)))),
    }
    return checks, summary


# -----------------------------
# Plot helpers
# -----------------------------
def _patch_src_plotting_monthly_date_axis() -> None:
    """
    Make src.plotting time-series plots use the old 05-style monthly x-axis:
      - monthly ticks
      - YYYY-MM labels
      - 45° rotation

    This patches src.plotting._format_date_axis used by:
      plot_forecast_timeseries / plot_ess_timeseries / plot_weight_diagnostics_timeseries
    """
    def _format_date_axis_monthly(ax: plt.Axes) -> None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=45)

    # Monkey-patch the helper used internally by src.plotting
    plotting_mod._format_date_axis = _format_date_axis_monthly  # type: ignore[attr-defined]

def _save_naive_check_plots(checks: pd.DataFrame, out_dir: Path) -> dict:
    plots: dict[str, str] = {}
    if checks.empty:
        return plots

    # Scatter p_is vs p_mc
    fig1 = plt.figure(figsize=(5.5, 5.0), dpi=180)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = checks["p_mc"].astype(float).to_numpy()
    y = checks["p_is"].astype(float).to_numpy()
    ax1.scatter(x, y, alpha=0.8, s=24)
    lo = float(np.nanmin(np.r_[x, y])) if len(x) else 0.0
    hi = float(np.nanmax(np.r_[x, y])) if len(x) else 1.0
    lo = min(lo, 0.0)
    hi = max(hi, 1e-6)
    ax1.plot([lo, hi], [lo, hi], linestyle="--")
    ax1.set_xlabel("Naive MC estimate")
    ax1.set_ylabel("IS rerun estimate")
    ax1.set_title("Naive MC vs IS (diagnostic reruns)")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    fig1.tight_layout()
    p1 = out_dir / "naive_vs_is_scatter.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)
    plots["naive_vs_is_scatter"] = str(p1.name)

    # Z-score histogram
    fig2 = plt.figure(figsize=(6.0, 4.0), dpi=180)
    ax2 = fig2.add_subplot(1, 1, 1)
    z = checks["z"].astype(float).to_numpy()
    ax2.hist(z, bins=min(20, max(5, len(z))), alpha=0.85)
    ax2.axvline(0.0, linestyle="--")
    ax2.axvline(2.0, linestyle=":", linewidth=1.2)
    ax2.axvline(-2.0, linestyle=":", linewidth=1.2)
    ax2.set_xlabel("z-score (IS - MC)")
    ax2.set_ylabel("Count")
    ax2.set_title("Naive-vs-IS z-score diagnostics")
    fig2.tight_layout()
    p2 = out_dir / "naive_vs_is_z_hist.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    plots["naive_vs_is_z_hist"] = str(p2.name)

    return plots


def _make_core_plots(
    *,
    with_labels: pd.DataFrame,
    rel_grouped_enriched: pd.DataFrame,
    out_plots_dir: Path,
    plot_max_groups: int,
) -> dict:
    """
    Create core reliability + time-series diagnostics plots.
    """
    out_plots_dir = ensure_dir(out_plots_dir)
    manifest: dict[str, list[str] | str | dict] = {"files": [], "groups": {}}

    # Combined reliability (all groups)
    p_rel_all = out_plots_dir / "reliability_all_groups.png"
    plot_reliability(rel_grouped_enriched, out_path=p_rel_all)
    manifest["files"].append(p_rel_all.name)

    # Pick groups by row count (usually small: default grid gives 4 groups)
    grp_counts = (
        with_labels.groupby(["Tdays", "delta"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "Tdays", "delta"], ascending=[False, True, True])
    )
    groups = grp_counts.head(int(plot_max_groups))[["Tdays", "delta"]].to_records(index=False)

    for rec in groups:
        T = int(rec[0])
        dlt = float(rec[1])
        gkey = f"T{T}_delta{dlt:.2f}"
        manifest["groups"][gkey] = []

        # Reliability (single group)
        p_rel = out_plots_dir / f"reliability_{gkey}.png"
        plot_reliability(rel_grouped_enriched, out_path=p_rel, tdays=T, delta=dlt)
        manifest["groups"][gkey].append(p_rel.name)

        # Forecast timeseries (+ realised labels if present)
        p_fc = out_plots_dir / f"forecast_{gkey}.png"
        plot_forecast_timeseries(with_labels, tdays=T, delta=dlt, out_path=p_fc)
        manifest["groups"][gkey].append(p_fc.name)

        # ESS timeseries
        if "ess" in with_labels.columns:
            p_ess = out_plots_dir / f"ess_{gkey}.png"
            plot_ess_timeseries(with_labels, tdays=T, delta=dlt, out_path=p_ess)
            manifest["groups"][gkey].append(p_ess.name)

        # Weight diagnostics
        if {"mean_weight", "max_weight"}.issubset(with_labels.columns):
            p_w = out_plots_dir / f"weights_{gkey}.png"
            plot_weight_diagnostics_timeseries(with_labels, tdays=T, delta=dlt, out_path=p_w)
            manifest["groups"][gkey].append(p_w.name)

    return manifest


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate forecasts against realised hit-by-horizon labels and run diagnostics: "
            "scores, calibration, uncertainty summaries, optional bootstrap, naive-vs-IS checks, plots."
        )
    )

    # Preferred usage: point at run directory created by script 03
    ap.add_argument("--run_dir", type=str, default=None, help="Run directory from script 03 (contains forecasts/, instantons/, diagnostics/).")

    # Forecast inputs (used if run_dir not provided)
    ap.add_argument("--forecasts_csv", type=str, default=None, help="Forecasts CSV; default=latest in --forecasts_dir.")
    ap.add_argument("--forecasts_dir", type=str, default="outputs/forecasts")
    ap.add_argument("--forecasts_pattern", type=str, default="*forecasts_*.csv")

    # State / params / instantons (auto-pinned from forecast metadata when possible)
    ap.add_argument("--state_csv", type=str, default=None)
    ap.add_argument("--state_dir", type=str, default="data/processed")
    ap.add_argument("--state_pattern", type=str, default="state_*.csv")

    ap.add_argument("--params_json", type=str, default=None)
    ap.add_argument("--params_dir", type=str, default="outputs/params")
    ap.add_argument("--params_pattern", type=str, default="sv_params_*.json")

    ap.add_argument("--instantons_dir", type=str, default=None, help="Defaults to forecast metadata output path if available.")

    # Outputs
    ap.add_argument("--out_dir", type=str, default=None, help="Defaults to run_dir/diagnostics or sibling diagnostics/ if forecasts are in a run folder.")

    # Reliability / calibration
    ap.add_argument("--reliability_bins", type=int, default=10)

    # Bootstrap uncertainty on scores
    ap.add_argument("--bootstrap_reps", type=int, default=200, help="Date-cluster bootstrap reps for score CI summaries. Set 0 to disable.")
    ap.add_argument("--bootstrap_seed", type=int, default=2026)

    # Plots
    ap.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plot_max_groups", type=int, default=12)

    # Naive-vs-IS checks (authoritative diagnostics stage)
    ap.add_argument("--run_naive_checks", action="store_true")
    ap.add_argument("--naive_check_mode", type=str, default="stratified", choices=["sample", "stratified", "all_feasible"])
    ap.add_argument("--naive_prob_bands", type=str, default="0.01,0.03,0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--p_lo", type=float, default=0.01)
    ap.add_argument("--p_hi", type=float, default=0.50)
    ap.add_argument("--naive_min_expected_hits", type=float, default=20.0, help="Feasibility filter: require naive_n * p_hat >= this.")
    ap.add_argument("--naive_max_total", type=int, default=200, help="Global cap on naive checks (<=0 means no cap).")
    ap.add_argument("--naive_max_per_stratum", type=int, default=5, help="For stratified mode: cap per (Tdays,delta,p-band) stratum.")
    ap.add_argument("--naive_n", type=int, default=20000)
    ap.add_argument("--is_n", type=int, default=5000)
    ap.add_argument("--seed_is", type=int, default=11)
    ap.add_argument("--seed_mc", type=int, default=23)

    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    resolved = _resolve_inputs(args)
    forecasts_csv: Path = resolved["forecasts_csv"]
    state_csv: Path = resolved["state_csv"]
    params_json: Path = resolved["params_json"]
    instantons_dir: Path = resolved["instantons_dir"]
    out_dir: Path = resolved["out_dir"]
    forecast_meta = resolved["forecast_meta"]

    forecasts = read_forecasts_csv(forecasts_csv)
    state = read_state_csv(state_csv)

    # -----------------
    # FORCED WINDOW
    # -----------------
    start_dt = pd.to_datetime("2007-01-01")
    end_dt = pd.to_datetime("2008-10-31")

    forecasts = forecasts[(forecasts["Date"] >= start_dt) & (forecasts["Date"] <= end_dt)].copy()

    # --- Realised labels + scores ---
    with_labels = add_realised_labels_to_forecasts(forecasts, state)
    scores = score_forecasts(with_labels, p_col="p_hat", y_col="y")
    scores_by = score_forecasts_by_group(with_labels, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y")

    # --- Reliability / calibration ---
    rel_overall = reliability_bins(with_labels, p_col="p_hat", y_col="y", n_bins=int(args.reliability_bins))
    rel_grouped = reliability_bins_by_group(
        with_labels, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y", n_bins=int(args.reliability_bins)
    )
    rel_overall_enriched = _enrich_reliability_bins(rel_overall)
    rel_grouped_enriched = _enrich_reliability_bins(rel_grouped)

    calib_overall = _calibration_summary_from_reliability(rel_overall_enriched, group_cols=tuple())
    calib_by_group = _calibration_summary_from_reliability(rel_grouped_enriched, group_cols=("Tdays", "delta"))

    # --- IS diagnostics summaries from forecast outputs ---
    is_diag_summary = _summarise_is_diagnostics(with_labels, group_cols=("Tdays", "delta"))

    # --- Bootstrap uncertainty on scores (date-cluster) ---
    boot_overall_scores, boot_by_group_scores = _date_cluster_bootstrap_scores(
        with_labels,
        n_boot=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed),
        group_cols=("Tdays", "delta"),
    )

    # --- Write core outputs ---
    out_eval_csv = out_dir / f"eval_joined_{forecasts_csv.stem}_{stamp}.csv"
    out_scores_by_csv = out_dir / f"scores_by_group_{forecasts_csv.stem}_{stamp}.csv"
    # IMPORTANT (unit tests): emit exactly ONE reliability_*.csv, and it must be the grouped table.
    out_rel_csv = out_dir / f"reliability_{forecasts_csv.stem}_{stamp}.csv"
    out_calib_overall_csv = out_dir / f"calibration_overall_{forecasts_csv.stem}_{stamp}.csv"
    out_calib_by_csv = out_dir / f"calibration_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_isdiag_csv = out_dir / f"is_diagnostics_summary_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_overall_csv = out_dir / f"bootstrap_scores_overall_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_by_csv = out_dir / f"bootstrap_scores_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_meta = out_dir / f"eval_{forecasts_csv.stem}_{stamp}.json"

    with_labels.to_csv(out_eval_csv, index=False)
    scores_by.to_csv(out_scores_by_csv, index=False)
    rel_grouped_enriched.to_csv(out_rel_csv, index=False)
    calib_overall.to_csv(out_calib_overall_csv, index=False)
    calib_by_group.to_csv(out_calib_by_csv, index=False)
    is_diag_summary.to_csv(out_isdiag_csv, index=False)
    boot_overall_scores.to_csv(out_boot_overall_csv, index=False)
    boot_by_group_scores.to_csv(out_boot_by_csv, index=False)

    # --- Plots ---
    plot_manifest = {}
    if bool(args.make_plots):
        _patch_src_plotting_monthly_date_axis()
        plots_dir = ensure_dir(out_dir / f"plots_{forecasts_csv.stem}_{stamp}")
        plot_manifest = _make_core_plots(
            with_labels=with_labels,
            rel_grouped_enriched=rel_grouped_enriched,
            out_plots_dir=plots_dir,
            plot_max_groups=int(args.plot_max_groups),
        )

    # --- Optional naive-vs-IS checks ---
    naive_checks_meta = None
    if args.run_naive_checks:
        bands = _parse_prob_bands(str(args.naive_prob_bands))
        checks_df, checks_summary = _run_naive_vs_is_checks(
            eval_df=with_labels,
            state=state,
            params_json=Path(params_json),
            instantons_dir=Path(instantons_dir),
            p_lo=float(args.p_lo),
            p_hi=float(args.p_hi),
            naive_check_mode=str(args.naive_check_mode),
            naive_prob_bands=bands,
            naive_max_total=int(args.naive_max_total),
            naive_max_per_stratum=int(args.naive_max_per_stratum),
            naive_n=int(args.naive_n),
            is_n=int(args.is_n),
            min_expected_hits=float(args.naive_min_expected_hits),
            seed_is=int(args.seed_is),
            seed_mc=int(args.seed_mc),
        )

        out_checks_csv = out_dir / f"naive_mc_checks_{forecasts_csv.stem}_{stamp}.csv"
        checks_df.to_csv(out_checks_csv, index=False)

        naive_plots = {}
        if bool(args.make_plots):
            _patch_src_plotting_monthly_date_axis()
            naive_plots_dir = ensure_dir(out_dir / f"naive_check_plots_{forecasts_csv.stem}_{stamp}")
            naive_plots = _save_naive_check_plots(checks_df, naive_plots_dir)

        naive_checks_meta = {
            "params_json": str(Path(params_json).resolve()),
            "instantons_dir": str(Path(instantons_dir).resolve()),
            "selection": {
                "mode": str(args.naive_check_mode),
                "p_lo": float(args.p_lo),
                "p_hi": float(args.p_hi),
                "prob_bands": bands,
                "naive_min_expected_hits": float(args.naive_min_expected_hits),
                "naive_max_total": int(args.naive_max_total),
                "naive_max_per_stratum": int(args.naive_max_per_stratum),
            },
            "reruns": {
                "is_n": int(args.is_n),
                "naive_n": int(args.naive_n),
                "seed_is_base": int(args.seed_is),
                "seed_mc_base": int(args.seed_mc),
            },
            "summary": checks_summary,
            "outputs": {
                "checks_csv": str(out_checks_csv.name),
                "plots": naive_plots,
            },
        }

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Unit tests expect these exact keys:
        "scores": scores,
        "scores_by_query": scores_by.to_dict(orient="records"),
        "inputs": {
            "forecasts_csv": str(forecasts_csv.resolve()),
            "state_csv": str(state_csv.resolve()),
            "params_json": str(Path(params_json).resolve()),
            "instantons_dir": str(Path(instantons_dir).resolve()),
        },
        "forecast_metadata_detected": bool(forecast_meta is not None),
        "window_eval": {
            "start": str(pd.to_datetime(start_dt).date()),
            "end": str(pd.to_datetime(end_dt).date()),
        },
        "reliability_bins": int(args.reliability_bins),
        "bootstrap": {
            "reps": int(args.bootstrap_reps),
            "seed": int(args.bootstrap_seed),
            "date_clustered": True,
        },
        "outputs": {
            "eval_joined_csv": str(out_eval_csv.name),
            "scores_by_group_csv": str(out_scores_by_csv.name),
            "reliability_csv": str(out_rel_csv.name),
            "calibration_overall_csv": str(out_calib_overall_csv.name),
            "calibration_by_group_csv": str(out_calib_by_csv.name),
            "is_diagnostics_summary_csv": str(out_isdiag_csv.name),
            "bootstrap_scores_overall_csv": str(out_boot_overall_csv.name),
            "bootstrap_scores_by_group_csv": str(out_boot_by_csv.name),
            "plots": plot_manifest,
        },
        "naive_mc_checks": naive_checks_meta,
        "cli_args": vars(args),
    }

    save_json(out_meta, meta)

    print("Wrote:")
    print(" ", out_eval_csv)
    print(" ", out_scores_by_csv)
    print(" ", out_rel_csv)
    print(" ", out_calib_overall_csv)
    print(" ", out_calib_by_csv)
    print(" ", out_isdiag_csv)
    print(" ", out_boot_overall_csv)
    print(" ", out_boot_by_csv)
    print(" ", out_meta)
    if args.run_naive_checks and naive_checks_meta:
        print(" ", out_dir / naive_checks_meta["outputs"]["checks_csv"])


if __name__ == "__main__":
    main()