from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from datetime import datetime, timezone
from typing import Optional

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
    empirical_rate_baseline_scores,
    empirical_rate_baseline_scores_by_group,
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
    resolve_input_file,
    file_fingerprint,
)
from src.plotting import (
    PlotSpec,
    plot_forecast_timeseries,
    plot_reliability,
    plot_ess_timeseries,
    plot_weight_diagnostics_timeseries,
)


def _maybe_infer_inputs_from_run_dir(run_dir: Path) -> tuple[Path, Path, Path]:
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
    p = forecasts_csv.resolve()
    if p.parent.name == "forecasts":
        return p.parent.parent / "diagnostics"
    return None


def _find_latest_file_recursive(root: Path, pattern: str) -> Path:
    matches = list(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern={pattern!r} under {Path(root).resolve()}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _maybe_infer_latest_run_dir(runs_root: Path = Path("outputs") / "runs") -> Path | None:
    """Infer most recent run directory under outputs/runs.

    A run directory is considered valid if it contains a forecasts/ folder.
    Returns None if no such directory exists.
    """
    root = Path(runs_root)
    if not root.exists():
        return None

    candidates: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "forecasts").exists():
            candidates.append(p)

    if not candidates:
        return None

    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))


def _assert_fingerprint_matches(label: str, expected: dict | None, actual_path: Path) -> None:
    if not expected:
        return
    actual = file_fingerprint(actual_path)
    if expected.get("sha256") != actual.get("sha256"):
        raise RuntimeError(
            f"{label} fingerprint mismatch. Forecast metadata expects {expected.get('path')} with sha256={expected.get('sha256')}, "
            f"but resolved {actual.get('path')} with sha256={actual.get('sha256')}."
        )


def _resolve_inputs(args: argparse.Namespace) -> dict:
    meta = None

    if args.run_dir:
        run_dir = Path(args.run_dir)
        forecasts_csv, instantons_dir_default, out_dir_default = _maybe_infer_inputs_from_run_dir(run_dir)
        out_dir = ensure_dir(out_dir_default if args.out_dir is None else Path(args.out_dir))
    else:
        run_dir = None
        used_run_fallback = False

        if args.forecasts_csv:
            forecasts_csv = Path(args.forecasts_csv)
        else:
            try:
                forecasts_csv = resolve_input_file(
                    directory=Path(args.forecasts_dir),
                    pattern=args.forecasts_pattern,
                    allow_latest=bool(args.allow_latest_inputs),
                    purpose="forecasts input",
                )
            except FileNotFoundError:
                # Backward-compatible convenience:
                # script 03 writes to outputs/runs/<run>/forecasts/, but the historical
                # default for this script was outputs/forecasts. If the user didn't
                # specify a forecasts path and the default directory has no matches,
                # infer the latest run folder instead of failing.
                if str(args.forecasts_dir).replace("\\", "/") == "outputs/forecasts":
                    inferred_run = _maybe_infer_latest_run_dir(Path("outputs") / "runs")
                    if inferred_run is not None:
                        run_dir = inferred_run
                        forecasts_csv, instantons_dir_default, out_dir_default = _maybe_infer_inputs_from_run_dir(run_dir)
                        out_dir = ensure_dir(out_dir_default if args.out_dir is None else Path(args.out_dir))
                        used_run_fallback = True
                    else:
                        raise
                else:
                    if not bool(args.allow_latest_inputs):
                        raise
                    forecasts_csv = _find_latest_file_recursive(Path("outputs") / "runs", args.forecasts_pattern)

        if not used_run_fallback:
            if args.instantons_dir:
                instantons_dir_default = Path(args.instantons_dir)
            else:
                p = forecasts_csv.resolve()
                instantons_dir_default = p.parent.parent / "instantons" if p.parent.name == "forecasts" else Path("outputs/instantons")

            inferred_out = _infer_default_out_dir_from_forecasts(forecasts_csv)
            out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (inferred_out or Path("outputs/diagnostics")))

    meta = maybe_load_forecast_metadata(forecasts_csv)

    if args.state_csv:
        state_csv = Path(args.state_csv)
    else:
        state_from_meta = meta.get("inputs", {}).get("state_source_csv") if isinstance(meta, dict) else None
        if state_from_meta:
            state_csv = Path(state_from_meta)
        else:
            state_csv = resolve_input_file(
                directory=Path(args.state_dir),
                pattern=args.state_pattern,
                allow_latest=bool(args.allow_latest_inputs),
                purpose="state input",
            )

    if args.params_json:
        params_json = Path(args.params_json)
    else:
        params_from_meta = meta.get("inputs", {}).get("params_json") if isinstance(meta, dict) else None
        if params_from_meta:
            params_json = Path(params_from_meta)
        else:
            params_json = resolve_input_file(
                directory=Path(args.params_dir),
                pattern=args.params_pattern,
                allow_latest=bool(args.allow_latest_inputs),
                purpose="parameter input",
            )

    if args.instantons_dir:
        instantons_dir = Path(args.instantons_dir)
    else:
        inst_from_meta = meta.get("outputs", {}).get("instantons_dir") if isinstance(meta, dict) else None
        instantons_dir = Path(inst_from_meta) if inst_from_meta else instantons_dir_default

    if isinstance(meta, dict):
        fp = meta.get("input_fingerprints", {})
        _assert_fingerprint_matches("state input", fp.get("state"), state_csv)
        _assert_fingerprint_matches("parameter input", fp.get("params"), params_json)

    return {
        "run_dir": run_dir,
        "forecasts_csv": forecasts_csv,
        "state_csv": state_csv,
        "params_json": params_json,
        "instantons_dir": instantons_dir,
        "out_dir": out_dir,
        "forecast_meta": meta,
    }


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n)) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def _enrich_reliability_bins(rel: pd.DataFrame) -> pd.DataFrame:
    r = rel.copy()
    if r.empty:
        return r

    count_f = pd.to_numeric(r["count"], errors="coerce").to_numpy(dtype=float)
    y_mean_f = pd.to_numeric(r["y_mean"], errors="coerce").to_numpy(dtype=float)
    k_f = np.round(y_mean_f * count_f)

    ci_lo = np.full(len(r), np.nan, dtype=float)
    ci_hi = np.full(len(r), np.nan, dtype=float)
    for i in range(len(r)):
        n_i_f = float(count_f[i]) if i < len(count_f) else np.nan
        k_i_f = float(k_f[i]) if i < len(k_f) else np.nan
        if not (np.isfinite(n_i_f) and np.isfinite(k_i_f)) or n_i_f <= 0:
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
    if rel.empty:
        cols = list(group_cols) + ["n_bins_nonempty", "n_obs", "ece", "mce", "rmsce"]
        return pd.DataFrame(columns=cols)

    r = rel.copy()
    req = {"count", "p_mean", "y_mean"}
    if not req.issubset(r.columns):
        raise KeyError(f"Reliability table missing columns: {req - set(r.columns)}")

    groups = [((), r)] if not group_cols else list(r.groupby(list(group_cols), sort=True))

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
    needed = {"p_hat", "se", "ess", "N", "mean_weight", "max_weight", "hit_rate_Q"}
    d = df.copy()
    for col in needed:
        if col not in d.columns:
            d[col] = np.nan

    for col in ("N", "ess", "se", "p_hat", "mean_weight", "max_weight", "hit_rate_Q"):
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["ess_frac"] = d["ess"] / d["N"]
    rel_se = np.full(len(d), np.nan, dtype=float)
    p = d["p_hat"].to_numpy(dtype=float)
    se = d["se"].to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(se) & (p > 0.0)
    rel_se[mask] = se[mask] / p[mask]
    d["rel_se"] = rel_se
    d["weight_ratio_max_mean"] = d["max_weight"] / np.clip(d["mean_weight"], 1e-12, np.inf)

    num_cols = [
        "p_hat", "se", "rel_se", "ess", "ess_frac", "mean_weight", "max_weight",
        "weight_ratio_max_mean", "hit_rate_Q"
    ]

    def _agg(sub: pd.DataFrame) -> dict:
        out: dict = {"n_rows": int(len(sub)), "n_positive_p": int(np.sum(pd.to_numeric(sub["p_hat"], errors="coerce") > 0.0))}
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


def _moving_block_bootstrap_scores(
    with_labels: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    block_length_dates: int,
    group_cols: tuple[str, ...] = ("Tdays", "delta"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    unique_dates = np.array(sorted(d["Date"].dropna().unique()))
    n_dates = len(unique_dates)
    if n_dates == 0:
        return (
            pd.DataFrame(columns=["metric", "mean", "q025", "q500", "q975", "n_boot"]),
            pd.DataFrame(columns=[*group_cols, "metric", "mean", "q025", "q500", "q975", "n_boot"]),
        )

    block_len = max(1, min(int(block_length_dates), n_dates))
    date_frames = {dt: d[d["Date"] == dt].copy() for dt in unique_dates}
    max_start = max(1, n_dates - block_len + 1)
    rng = np.random.default_rng(int(seed))

    overall_recs: list[dict] = []
    by_group_recs: list[dict] = []

    for b in range(int(n_boot)):
        picked_dates: list[pd.Timestamp] = []
        while len(picked_dates) < n_dates:
            start_idx = int(rng.integers(0, max_start))
            block = list(unique_dates[start_idx : start_idx + block_len])
            picked_dates.extend(block)
        picked_dates = picked_dates[:n_dates]
        boot_df = pd.concat([date_frames[dt] for dt in picked_dates], ignore_index=True)

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
        groups = [((), df_in)] if not group_cols_local else df_in.groupby(list(group_cols_local), sort=True)

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
        df = df[(float(naive_n) * df["p_hat"].astype(float)) >= float(min_expected_hits)]
        return df.head(int(max_total)).copy() if max_total > 0 else df.copy()

    if mode == "stratified":
        edges = np.asarray(prob_bands, dtype=float)
        p = df["p_hat"].astype(float).to_numpy()
        band = np.digitize(p, edges, right=False) - 1
        band = np.clip(band, 0, len(edges) - 2)
        df["p_band_idx"] = band
        df["p_band"] = [f"[{edges[i]:.3g},{edges[i+1]:.3g})" for i in band]
        df = df[(float(naive_n) * df["p_hat"].astype(float)) >= float(min_expected_hits)]
        parts: list[pd.DataFrame] = []
        for _, sub in df.groupby(["Tdays", "delta", "p_band_idx"], sort=True):
            parts.append(sub.head(int(max_per_stratum)))
        if not parts:
            return df.head(0).copy()
        out = pd.concat(parts, ignore_index=False).sort_values(["Date", "Tdays", "delta"]).copy()
        return out.head(int(max_total)).copy() if max_total > 0 and len(out) > int(max_total) else out

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
        return checks, {
            "n_checks": 0,
            "z_mean": np.nan,
            "z_std": np.nan,
            "frac_abs_z_gt_2": np.nan,
            "frac_abs_z_gt_3": np.nan,
        }

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


def _patch_src_plotting_monthly_date_axis() -> None:
    def _format_date_axis_monthly(ax: plt.Axes, *, major_month_interval: int | None = None, fmt: str | None = None) -> None:
        interval = 1 if major_month_interval is None else int(major_month_interval)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt or "%Y-%m"))
        ax.tick_params(axis="x", labelrotation=45)

    plotting_mod._format_date_axis = _format_date_axis_monthly  # type: ignore[attr-defined]


def _save_naive_check_plots(checks: pd.DataFrame, out_dir: Path) -> dict:
    plots: dict[str, str] = {}
    if checks.empty:
        return plots

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
    out_plots_dir = ensure_dir(out_plots_dir)
    manifest: dict[str, list[str] | str | dict] = {"files": [], "groups": {}}

    p_rel_all = out_plots_dir / "reliability_all_groups.png"
    plot_reliability(rel_grouped_enriched, out_path=p_rel_all)
    manifest["files"].append(p_rel_all.name)

    delta_counts = (
        with_labels.groupby("delta", sort=True)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "delta"], ascending=[False, True])
    )

    deltas = delta_counts.head(int(plot_max_groups))["delta"].to_list()

    for dlt in deltas:
        dlt = float(dlt)
        gkey = f"delta{dlt:.2f}"
        manifest["groups"][gkey] = []

        tdays_values = sorted(
            pd.unique(
                with_labels.loc[np.isclose(with_labels["delta"].astype(float), dlt), "Tdays"].astype(int)
            )
        )

        p_rel = out_plots_dir / f"reliability_{gkey}.png"
        plot_reliability(rel_grouped_enriched, out_path=p_rel, delta=dlt)
        manifest["groups"][gkey].append(p_rel.name)

        p_fc = out_plots_dir / f"forecast_{gkey}.png"
        fc_spec = PlotSpec(font_size=13, title_size=14, label_size=13, tick_size=12)
        plot_forecast_timeseries(
            with_labels,
            tdays=tdays_values,
            delta=dlt,
            out_path=p_fc,
            spec=fc_spec,
            date_tick_months=3,
        )
        manifest["groups"][gkey].append(p_fc.name)

        if "ess" in with_labels.columns:
            p_ess = out_plots_dir / f"ess_{gkey}.png"
            plot_ess_timeseries(with_labels, tdays=tdays_values, delta=dlt, out_path=p_ess)
            manifest["groups"][gkey].append(p_ess.name)

        if {"mean_weight", "max_weight"}.issubset(with_labels.columns):
            p_w = out_plots_dir / f"weights_{gkey}.png"
            plot_weight_diagnostics_timeseries(with_labels, tdays=tdays_values, delta=dlt, out_path=p_w)
            manifest["groups"][gkey].append(p_w.name)

    return manifest


def _add_origin_hit_flag(forecasts: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    out = forecasts.copy()
    st = state.copy()
    st.index = pd.to_datetime(st.index)
    st = st.sort_index()
    if "D" not in st.columns:
        raise KeyError("State CSV must contain D column to compute already-hit-at-origin flags.")

    d_map = st["D"].astype(float)
    dates = pd.to_datetime(out["Date"])
    if "d" in out.columns:
        d_thresh = out["d"].astype(float).to_numpy(dtype=float)
    else:
        d_thresh = np.array([float(log_drawdown_threshold(float(x))) for x in out["delta"].astype(float)], dtype=float)
        out["d"] = d_thresh

    d0 = d_map.reindex(dates).to_numpy(dtype=float)
    already_hit = d0 >= d_thresh
    out["d0_at_origin"] = d0
    out["already_hit_at_origin"] = already_hit.astype(bool)

    if "tau_star" in out.columns:
        tau0 = out["tau_star"].fillna(-1).astype(int).to_numpy() == 0
        mismatch = np.where(tau0 != already_hit)[0]
        if len(mismatch) > 0:
            raise RuntimeError(
                f"Mismatch between tau_star==0 and origin-hit flag on {len(mismatch)} rows. "
                "This suggests an inconsistency between the forecast file and the state file."
            )
    return out


def _score_comparison(model_by: pd.DataFrame, baseline_by: pd.DataFrame, *, group_cols: tuple[str, ...] = ("Tdays", "delta")) -> pd.DataFrame:
    if model_by.empty and baseline_by.empty:
        return pd.DataFrame(columns=[*group_cols, "n", "mean_log_score_model", "mean_log_score_baseline", "log_score_improvement", "mean_brier_model", "mean_brier_baseline", "brier_improvement"])
    m = model_by.copy()
    b = baseline_by.copy()
    merged = m.merge(b, on=list(group_cols) + ["n"], how="outer", suffixes=("_model", "_baseline"))
    merged["log_score_improvement"] = merged["mean_log_score_baseline"] - merged["mean_log_score_model"]
    merged["brier_improvement"] = merged["mean_brier_baseline"] - merged["mean_brier_model"]
    return merged


def _probability_ordering_diagnostics(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    p_col: str = "p_hat",
    se_col: str = "se",
    sigma_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Check the four natural monotonicity relations on the committed grid:
      p(20d, 0.20) >= p(20d, 0.30)
      p(60d, 0.20) >= p(60d, 0.30)
      p(60d, 0.20) >= p(20d, 0.20)
      p(60d, 0.30) >= p(20d, 0.30)

    Because these are Monte Carlo estimates, we report both raw violations and
    violations scaled by the combined Monte Carlo standard error.

    Notes
    -----
    This function is intentionally robust to partial forecast grids. If the
    incoming dataframe does not contain any of the required (Tdays, delta)
    combinations for the configured checks, it returns an empty dataframe with
    the expected output schema instead of raising during sort.
    """
    out_cols = [
        "Date",
        "check",
        "lhs_Tdays",
        "lhs_delta",
        "rhs_Tdays",
        "rhs_delta",
        "lhs_p_hat",
        "rhs_p_hat",
        "lhs_se",
        "rhs_se",
        "combined_se",
        "diff_lhs_minus_rhs",
        "z_diff",
        "raw_violation",
        "sigma_violation",
    ]

    need = {date_col, "Tdays", "delta", p_col}
    if not need.issubset(df.columns):
        raise KeyError(f"Missing columns for ordering diagnostics: {need - set(df.columns)}")

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    q = df[[date_col, "Tdays", "delta", p_col] + ([se_col] if se_col in df.columns else [])].copy()
    q[date_col] = pd.to_datetime(q[date_col], errors="coerce")
    q = q[q[date_col].notna()].copy()
    if q.empty:
        return pd.DataFrame(columns=out_cols)

    q["delta_key"] = pd.to_numeric(q["delta"], errors="coerce").astype(float).round(10)
    q["Tdays"] = pd.to_numeric(q["Tdays"], errors="coerce").astype("Int64")
    q[p_col] = pd.to_numeric(q[p_col], errors="coerce")
    q = q[q["Tdays"].notna()].copy()
    q["Tdays"] = q["Tdays"].astype(int)

    if se_col in q.columns:
        q[se_col] = pd.to_numeric(q[se_col], errors="coerce")

    if q.empty:
        return pd.DataFrame(columns=out_cols)

    p_piv = q.pivot_table(index=date_col, columns=["Tdays", "delta_key"], values=p_col, aggfunc="first")
    if p_piv.empty:
        return pd.DataFrame(columns=out_cols)

    if se_col in q.columns:
        se_piv = q.pivot_table(index=date_col, columns=["Tdays", "delta_key"], values=se_col, aggfunc="first")
    else:
        se_piv = None

    pairs = [
        ("delta_monotone_T20", (20, 0.20), (20, 0.30)),
        ("delta_monotone_T60", (60, 0.20), (60, 0.30)),
        ("horizon_monotone_d20", (60, 0.20), (20, 0.20)),
        ("horizon_monotone_d30", (60, 0.30), (20, 0.30)),
    ]

    available_cols = set(p_piv.columns.tolist())
    applicable_pairs: list[tuple[str, tuple[int, float], tuple[int, float]]] = []
    for label, lhs, rhs in pairs:
        lhs_key = (int(lhs[0]), round(float(lhs[1]), 10))
        rhs_key = (int(rhs[0]), round(float(rhs[1]), 10))
        if lhs_key in available_cols and rhs_key in available_cols:
            applicable_pairs.append((label, lhs, rhs))

    if not applicable_pairs:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict] = []
    sigma_cutoff = abs(float(sigma_threshold))

    for dt in p_piv.index:
        for label, lhs, rhs in applicable_pairs:
            lhs_key = (int(lhs[0]), round(float(lhs[1]), 10))
            rhs_key = (int(rhs[0]), round(float(rhs[1]), 10))

            p_lhs_raw = p_piv.loc[dt, lhs_key]
            p_rhs_raw = p_piv.loc[dt, rhs_key]
            if pd.isna(p_lhs_raw) or pd.isna(p_rhs_raw):
                continue

            p_lhs = float(p_lhs_raw)
            p_rhs = float(p_rhs_raw)
            diff = p_lhs - p_rhs

            se_lhs = np.nan
            se_rhs = np.nan
            se_comb = np.nan
            z = np.nan
            violation_sigma = False

            if se_piv is not None:
                if lhs_key in se_piv.columns:
                    se_lhs_raw = se_piv.loc[dt, lhs_key]
                    if pd.notna(se_lhs_raw):
                        se_lhs = float(se_lhs_raw)

                if rhs_key in se_piv.columns:
                    se_rhs_raw = se_piv.loc[dt, rhs_key]
                    if pd.notna(se_rhs_raw):
                        se_rhs = float(se_rhs_raw)

                if np.isfinite(se_lhs) and np.isfinite(se_rhs):
                    se_comb = float(np.sqrt(se_lhs * se_lhs + se_rhs * se_rhs))
                    if se_comb > 0.0:
                        z = float(diff / se_comb)
                        violation_sigma = bool(z < -sigma_cutoff)

            rows.append(
                {
                    "Date": pd.to_datetime(dt),
                    "check": label,
                    "lhs_Tdays": int(lhs[0]),
                    "lhs_delta": float(lhs[1]),
                    "rhs_Tdays": int(rhs[0]),
                    "rhs_delta": float(rhs[1]),
                    "lhs_p_hat": p_lhs,
                    "rhs_p_hat": p_rhs,
                    "lhs_se": float(se_lhs) if np.isfinite(se_lhs) else np.nan,
                    "rhs_se": float(se_rhs) if np.isfinite(se_rhs) else np.nan,
                    "combined_se": float(se_comb) if np.isfinite(se_comb) else np.nan,
                    "diff_lhs_minus_rhs": float(diff),
                    "z_diff": float(z) if np.isfinite(z) else np.nan,
                    "raw_violation": bool(diff < 0.0),
                    "sigma_violation": bool(violation_sigma),
                }
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    return pd.DataFrame(rows, columns=out_cols).sort_values(["Date", "check"]).reset_index(drop=True)


def _ordering_summary(diag: pd.DataFrame) -> dict:
    if diag.empty:
        return {
            "n_checks": 0,
            "raw_violations": 0,
            "sigma_violations": 0,
            "max_negative_gap": np.nan,
            "min_z": np.nan,
        }
    diff = pd.to_numeric(diag["diff_lhs_minus_rhs"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(diag["z_diff"], errors="coerce").to_numpy(dtype=float)
    return {
        "n_checks": int(len(diag)),
        "raw_violations": int(np.sum(diag["raw_violation"].astype(bool))),
        "sigma_violations": int(np.sum(diag["sigma_violation"].astype(bool))),
        "max_negative_gap": float(np.min(diff)) if len(diff) else np.nan,
        "min_z": float(np.nanmin(z)) if np.isfinite(z).any() else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate forecasts against realised labels and run diagnostics: scores, calibration, uncertainty summaries, "
            "optional bootstrap, naive-vs-IS checks, plots."
        )
    )

    ap.add_argument("--run_dir", type=str, default=None, help="Run directory from script 03 (contains forecasts/, instantons/, diagnostics/).")
    ap.add_argument("--forecasts_csv", type=str, default=None, help="Forecasts CSV; default=resolved from --run_dir or unique match in --forecasts_dir.")
    ap.add_argument("--forecasts_dir", type=str, default="outputs/forecasts")
    ap.add_argument("--forecasts_pattern", type=str, default="*forecasts_*.csv")
    ap.add_argument("--state_csv", type=str, default=None)
    ap.add_argument("--state_dir", type=str, default="data/processed")
    ap.add_argument("--state_pattern", type=str, default="state_*.csv")
    ap.add_argument("--params_json", type=str, default=None)
    ap.add_argument("--params_dir", type=str, default="outputs/params")
    ap.add_argument("--params_pattern", type=str, default="sv_params_*.json")
    ap.add_argument("--instantons_dir", type=str, default=None, help="Defaults to forecast metadata output path if available.")
    ap.add_argument("--out_dir", type=str, default=None, help="Defaults to run_dir/diagnostics or sibling diagnostics/ if forecasts are in a run folder.")
    ap.add_argument("--allow_latest_inputs", action="store_true", help="Allow newest matching inputs when file patterns are ambiguous.")

    ap.add_argument("--reliability_bins", type=int, default=10)

    ap.add_argument("--bootstrap_reps", type=int, default=200, help="Moving-block bootstrap reps for score CI summaries. Set 0 to disable.")
    ap.add_argument("--bootstrap_seed", type=int, default=2026)
    ap.add_argument("--bootstrap_block_days", type=int, default=None, help="Moving-block bootstrap length in trading days. Default=max forecast horizon in the sample.")

    ap.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plot_max_groups", type=int, default=12)

    ap.add_argument("--run_naive_checks", action="store_true")
    ap.add_argument("--naive_check_mode", type=str, default="stratified", choices=["sample", "stratified", "all_feasible"])
    ap.add_argument("--naive_prob_bands", type=str, default="0.01,0.03,0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--p_lo", type=float, default=0.01)
    ap.add_argument("--p_hi", type=float, default=0.50)
    ap.add_argument("--naive_min_expected_hits", type=float, default=20.0)
    ap.add_argument("--naive_max_total", type=int, default=200)
    ap.add_argument("--naive_max_per_stratum", type=int, default=5)
    ap.add_argument("--naive_n", type=int, default=20000)
    ap.add_argument("--is_n", type=int, default=5000)
    ap.add_argument("--seed_is", type=int, default=11)
    ap.add_argument("--seed_mc", type=int, default=23)
    
    ap.add_argument("--strict_ordering", action="store_true")
    ap.add_argument("--ordering_sigma_threshold", type=float, default=3.0)

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

    start_dt = pd.to_datetime("2007-01-01")
    end_dt = pd.to_datetime("2008-10-31")
    forecasts = forecasts[(forecasts["Date"] >= start_dt) & (forecasts["Date"] <= end_dt)].copy()
    forecasts = _add_origin_hit_flag(forecasts, state)

    with_labels = add_realised_labels_to_forecasts(forecasts, state)
    prehit = with_labels[~with_labels["already_hit_at_origin"].astype(bool)].copy()

    scores_full = score_forecasts(with_labels, p_col="p_hat", y_col="y")
    scores_full_by = score_forecasts_by_group(with_labels, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y")
    scores_prehit = score_forecasts(prehit, p_col="p_hat", y_col="y")
    scores_prehit_by = score_forecasts_by_group(prehit, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y")

    baseline_full = empirical_rate_baseline_scores(with_labels, y_col="y")
    baseline_full_by = empirical_rate_baseline_scores_by_group(with_labels, group_cols=("Tdays", "delta"), y_col="y")
    baseline_prehit = empirical_rate_baseline_scores(prehit, y_col="y")
    baseline_prehit_by = empirical_rate_baseline_scores_by_group(prehit, group_cols=("Tdays", "delta"), y_col="y")

    comparison_full_by = _score_comparison(scores_full_by, baseline_full_by)
    comparison_prehit_by = _score_comparison(scores_prehit_by, baseline_prehit_by)

    rel_full = reliability_bins_by_group(with_labels, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y", n_bins=int(args.reliability_bins))
    rel_prehit = reliability_bins_by_group(prehit, group_cols=("Tdays", "delta"), p_col="p_hat", y_col="y", n_bins=int(args.reliability_bins))
    rel_full_enriched = _enrich_reliability_bins(rel_full)
    rel_prehit_enriched = _enrich_reliability_bins(rel_prehit)

    calib_full = _calibration_summary_from_reliability(rel_full_enriched, group_cols=("Tdays", "delta"))
    calib_prehit = _calibration_summary_from_reliability(rel_prehit_enriched, group_cols=("Tdays", "delta"))

    is_diag_full = _summarise_is_diagnostics(with_labels, group_cols=("Tdays", "delta"))
    is_diag_prehit = _summarise_is_diagnostics(prehit, group_cols=("Tdays", "delta"))
    
    ordering_full = _probability_ordering_diagnostics(
        with_labels,
        sigma_threshold=float(args.ordering_sigma_threshold),
    )
    ordering_prehit = _probability_ordering_diagnostics(
        prehit,
        sigma_threshold=float(args.ordering_sigma_threshold),
    )
    ordering_full_summary = _ordering_summary(ordering_full)
    ordering_prehit_summary = _ordering_summary(ordering_prehit)

    if args.bootstrap_block_days is None:
        block_days = int(max(1, forecasts["Tdays"].astype(int).max())) if not forecasts.empty else 1
    else:
        block_days = int(args.bootstrap_block_days)

    boot_full_overall, boot_full_by = _moving_block_bootstrap_scores(
        with_labels,
        n_boot=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed),
        block_length_dates=block_days,
        group_cols=("Tdays", "delta"),
    )
    boot_prehit_overall, boot_prehit_by = _moving_block_bootstrap_scores(
        prehit,
        n_boot=int(args.bootstrap_reps),
        seed=int(args.bootstrap_seed) + 1,
        block_length_dates=block_days,
        group_cols=("Tdays", "delta"),
    )

    out_eval_csv = out_dir / f"eval_joined_{forecasts_csv.stem}_{stamp}.csv"
    out_scores_full_by_csv = out_dir / f"scores_full_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_scores_prehit_by_csv = out_dir / f"scores_prehit_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_baseline_full_by_csv = out_dir / f"baseline_full_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_baseline_prehit_by_csv = out_dir / f"baseline_prehit_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_comparison_full_by_csv = out_dir / f"score_comparison_full_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_comparison_prehit_by_csv = out_dir / f"score_comparison_prehit_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_rel_full_csv = out_dir / f"reliability_full_{forecasts_csv.stem}_{stamp}.csv"
    # NOTE: Keep exactly one file matching the legacy glob "reliability_*.csv".
    # Older tests expect a single reliability output file; the pre-hit variant is
    # still written but under a name that won't match that glob.
    out_rel_prehit_csv = out_dir / f"reliabilityprehit_{forecasts_csv.stem}_{stamp}.csv"
    out_calib_full_csv = out_dir / f"calibration_full_{forecasts_csv.stem}_{stamp}.csv"
    out_calib_prehit_csv = out_dir / f"calibration_prehit_{forecasts_csv.stem}_{stamp}.csv"
    out_isdiag_full_csv = out_dir / f"is_diagnostics_full_{forecasts_csv.stem}_{stamp}.csv"
    out_isdiag_prehit_csv = out_dir / f"is_diagnostics_prehit_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_full_overall_csv = out_dir / f"bootstrap_scores_full_overall_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_full_by_csv = out_dir / f"bootstrap_scores_full_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_prehit_overall_csv = out_dir / f"bootstrap_scores_prehit_overall_{forecasts_csv.stem}_{stamp}.csv"
    out_boot_prehit_by_csv = out_dir / f"bootstrap_scores_prehit_by_group_{forecasts_csv.stem}_{stamp}.csv"
    out_meta = out_dir / f"eval_{forecasts_csv.stem}_{stamp}.json"
    out_ordering_full_csv = out_dir / f"ordering_full_{forecasts_csv.stem}_{stamp}.csv"
    out_ordering_prehit_csv = out_dir / f"ordering_prehit_{forecasts_csv.stem}_{stamp}.csv"

    with_labels.to_csv(out_eval_csv, index=False)
    scores_full_by.to_csv(out_scores_full_by_csv, index=False)
    scores_prehit_by.to_csv(out_scores_prehit_by_csv, index=False)
    baseline_full_by.to_csv(out_baseline_full_by_csv, index=False)
    baseline_prehit_by.to_csv(out_baseline_prehit_by_csv, index=False)
    comparison_full_by.to_csv(out_comparison_full_by_csv, index=False)
    comparison_prehit_by.to_csv(out_comparison_prehit_by_csv, index=False)
    rel_full_enriched.to_csv(out_rel_full_csv, index=False)
    rel_prehit_enriched.to_csv(out_rel_prehit_csv, index=False)
    calib_full.to_csv(out_calib_full_csv, index=False)
    calib_prehit.to_csv(out_calib_prehit_csv, index=False)
    is_diag_full.to_csv(out_isdiag_full_csv, index=False)
    is_diag_prehit.to_csv(out_isdiag_prehit_csv, index=False)
    boot_full_overall.to_csv(out_boot_full_overall_csv, index=False)
    boot_full_by.to_csv(out_boot_full_by_csv, index=False)
    boot_prehit_overall.to_csv(out_boot_prehit_overall_csv, index=False)
    boot_prehit_by.to_csv(out_boot_prehit_by_csv, index=False)
    ordering_full.to_csv(out_ordering_full_csv, index=False)
    ordering_prehit.to_csv(out_ordering_prehit_csv, index=False)

    plot_manifest = {}
    if bool(args.make_plots):
        _patch_src_plotting_monthly_date_axis()
        plots_dir = ensure_dir(out_dir / f"plots_{forecasts_csv.stem}_{stamp}")
        plot_manifest = _make_core_plots(
            with_labels=with_labels,
            rel_grouped_enriched=rel_full_enriched,
            out_plots_dir=plots_dir,
            plot_max_groups=int(args.plot_max_groups),
        )

    naive_checks_meta = None
    if args.run_naive_checks:
        bands = _parse_prob_bands(str(args.naive_prob_bands))
        checks_df, checks_summary = _run_naive_vs_is_checks(
            eval_df=prehit,
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
                "sample": "pre-hit subset only",
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

    model_vs_baseline_full = {
        "mean_log_score_improvement": float(baseline_full["mean_log_score"] - scores_full["mean_log_score"]) if np.isfinite(baseline_full["mean_log_score"]) and np.isfinite(scores_full["mean_log_score"]) else np.nan,
        "mean_brier_improvement": float(baseline_full["mean_brier"] - scores_full["mean_brier"]) if np.isfinite(baseline_full["mean_brier"]) and np.isfinite(scores_full["mean_brier"]) else np.nan,
    }
    model_vs_baseline_prehit = {
        "mean_log_score_improvement": float(baseline_prehit["mean_log_score"] - scores_prehit["mean_log_score"]) if np.isfinite(baseline_prehit["mean_log_score"]) and np.isfinite(scores_prehit["mean_log_score"]) else np.nan,
        "mean_brier_improvement": float(baseline_prehit["mean_brier"] - scores_prehit["mean_brier"]) if np.isfinite(baseline_prehit["mean_brier"]) and np.isfinite(scores_prehit["mean_brier"]) else np.nan,
    }

    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Backward-compatible keys used by tests and older downstream tooling.
        "scores": scores_full,
        "scores_by_query": scores_full_by.to_dict(orient="records"),
        "scores_full": scores_full,
        "scores_prehit": scores_prehit,
        "baseline_full": baseline_full,
        "baseline_prehit": baseline_prehit,
        "model_vs_baseline_full": model_vs_baseline_full,
        "model_vs_baseline_prehit": model_vs_baseline_prehit,
        "scores_full_by_query": scores_full_by.to_dict(orient="records"),
        "scores_prehit_by_query": scores_prehit_by.to_dict(orient="records"),
        "baseline_full_by_query": baseline_full_by.to_dict(orient="records"),
        "baseline_prehit_by_query": baseline_prehit_by.to_dict(orient="records"),
        "inputs": {
            "forecasts_csv": str(forecasts_csv.resolve()),
            "state_csv": str(state_csv.resolve()),
            "params_json": str(Path(params_json).resolve()),
            "instantons_dir": str(Path(instantons_dir).resolve()),
        },
        "input_fingerprints": {
            "forecasts": file_fingerprint(forecasts_csv),
            "state": file_fingerprint(state_csv),
            "params": file_fingerprint(params_json),
        },
        "forecast_metadata_detected": bool(forecast_meta is not None),
        "window_eval": {
            "start": str(pd.to_datetime(start_dt).date()),
            "end": str(pd.to_datetime(end_dt).date()),
        },
        "sample_counts": {
            "rows_full": int(len(with_labels)),
            "rows_with_realised_label_full": int(np.sum(np.isfinite(pd.to_numeric(with_labels["y"], errors="coerce")))),
            "rows_prehit": int(len(prehit)),
            "rows_with_realised_label_prehit": int(np.sum(np.isfinite(pd.to_numeric(prehit["y"], errors="coerce")))),
        },
        "reliability_bins": int(args.reliability_bins),
        "bootstrap": {
            "reps": int(args.bootstrap_reps),
            "seed": int(args.bootstrap_seed),
            "method": "moving_block_by_date",
            "block_days": int(block_days),
        },
        "ordering_checks": {
            "sigma_threshold": float(args.ordering_sigma_threshold),
            "full_summary": ordering_full_summary,
            "prehit_summary": ordering_prehit_summary,
            "outputs": {
                "ordering_full_csv": str(out_ordering_full_csv.name),
                "ordering_prehit_csv": str(out_ordering_prehit_csv.name),
            },
        },
        "outputs": {
            "eval_joined_csv": str(out_eval_csv.name),
            "scores_full_by_group_csv": str(out_scores_full_by_csv.name),
            "scores_prehit_by_group_csv": str(out_scores_prehit_by_csv.name),
            "baseline_full_by_group_csv": str(out_baseline_full_by_csv.name),
            "baseline_prehit_by_group_csv": str(out_baseline_prehit_by_csv.name),
            "score_comparison_full_by_group_csv": str(out_comparison_full_by_csv.name),
            "score_comparison_prehit_by_group_csv": str(out_comparison_prehit_by_csv.name),
            "reliability_full_csv": str(out_rel_full_csv.name),
            "reliability_prehit_csv": str(out_rel_prehit_csv.name),
            "calibration_full_csv": str(out_calib_full_csv.name),
            "calibration_prehit_csv": str(out_calib_prehit_csv.name),
            "is_diagnostics_full_csv": str(out_isdiag_full_csv.name),
            "is_diagnostics_prehit_csv": str(out_isdiag_prehit_csv.name),
            "bootstrap_scores_full_overall_csv": str(out_boot_full_overall_csv.name),
            "bootstrap_scores_full_by_group_csv": str(out_boot_full_by_csv.name),
            "bootstrap_scores_prehit_overall_csv": str(out_boot_prehit_overall_csv.name),
            "bootstrap_scores_prehit_by_group_csv": str(out_boot_prehit_by_csv.name),
            "plots": plot_manifest,
        },
        "naive_mc_checks": naive_checks_meta,
        "cli_args": vars(args),
    }
    
    if bool(args.strict_ordering) and int(ordering_prehit_summary["sigma_violations"]) > 0:
        raise RuntimeError(
            "Significant probability-ordering violations detected in the pre-hit subset. "
            f"See {out_ordering_prehit_csv} for details."
        )

    save_json(out_meta, meta)

    print("Wrote:")
    for p in [
        out_eval_csv,
        out_scores_full_by_csv,
        out_scores_prehit_by_csv,
        out_baseline_full_by_csv,
        out_baseline_prehit_by_csv,
        out_comparison_full_by_csv,
        out_comparison_prehit_by_csv,
        out_rel_full_csv,
        out_rel_prehit_csv,
        out_calib_full_csv,
        out_calib_prehit_csv,
        out_isdiag_full_csv,
        out_isdiag_prehit_csv,
        out_boot_full_overall_csv,
        out_boot_full_by_csv,
        out_boot_prehit_overall_csv,
        out_boot_prehit_by_csv,
        out_meta,
    ]:
        print(" ", p)
    if args.run_naive_checks and naive_checks_meta:
        print(" ", out_dir / naive_checks_meta["outputs"]["checks_csv"])


if __name__ == "__main__":
    main()
