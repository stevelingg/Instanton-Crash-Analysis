from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import logging
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.constants import (
    DELTA_GRID,
    TDAYS_GRID,
    LAMBDA_IS,
    UMAX,
    HIT_EPS,
    log_drawdown_threshold,
)
from src.fit import SVParams
from src.io import (
    ensure_dir,
    save_json,
    resolve_input_file,
    read_state_csv,
    load_params_json,
    make_state_tasks,
    file_fingerprint,
)
from src.instanton import select_instanton_tau_star, InstantonSolution, MAMSettings
from src.importance_sampling import cap_control, simulate_is_paths, estimate_probability_is


def _stable_query_seed(base_seed: int, date_str: str, tdays: int, delta: float) -> int:
    payload = f"{int(base_seed)}|{date_str}|{int(tdays)}|{float(delta):.12g}".encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    seed64 = int.from_bytes(h, "little", signed=False)
    return int(seed64 % (2**32 - 1))


def save_instanton_npz(out_dir: Path, date: str, tdays: int, delta: float, sol: InstantonSolution) -> Path:
    fname = f"instanton_{date}_T{tdays}_delta{delta:.2f}_tau{sol.tau}_am{int(sol.alpha_m)}.npz"
    f = out_dir / fname
    np.savez_compressed(
        f,
        tau=sol.tau,
        alpha_m=sol.alpha_m,
        action=sol.action,
        u_star=sol.u_star,
        x_path=sol.x_path,
        v_path=sol.v_path,
        dtilde_tau=sol.dtilde_tau,
        dexact_tau=sol.dexact_tau,
        prehit_max_violation=sol.prehit_max_violation,
        hit_err=sol.hit_err,
    )
    return f


def _run_block(
    tasks: list[tuple[str, float, float, float]],
    *,
    tdays_grid: tuple[int, ...],
    delta_grid: tuple[float, ...],
    params: SVParams,
    n_paths: int,
    lam: float,
    umax: float,
    base_seed: int,
    seed_mode: str,
    refine_radius: int,
    mam_settings: MAMSettings,
    instantons_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    warm_cache: dict[tuple[int, float], InstantonSolution] = {}

    for (date_str, x0, v0, d0) in tasks:
        for tdays in tdays_grid:
            for delta in delta_grid:
                d = float(log_drawdown_threshold(float(delta)))

                if float(d0) >= float(d):
                    rows.append(
                        {
                            "Date": date_str,
                            "Tdays": int(tdays),
                            "delta": float(delta),
                            "d": float(d),
                            "tau_star": 0,
                            "alpha_m": float("nan"),
                            "action": float("nan"),
                            "dtilde_tau": float(d0),
                            "dexact_tau": float(d0),
                            "exact_shortfall": float("nan"),
                            "prehit_max_violation": 0.0,
                            "hit_err": 0.0,
                            "lam": float(lam),
                            "umax": float(umax),
                            "N": int(n_paths),
                            "seed_mode": str(seed_mode),
                            "seed_used": int(base_seed if seed_mode == "fixed" else _stable_query_seed(base_seed, date_str, int(tdays), float(delta))),
                            "p_hat": 1.0,
                            "se": 0.0,
                            "ess": float(n_paths),
                            "mean_weight": 1.0,
                            "max_weight": 1.0,
                            "hit_rate_Q": 1.0,
                            "instanton_file": "",
                        }
                    )
                    continue

                warm = warm_cache.get((int(tdays), float(delta)))
                sol = select_instanton_tau_star(
                    params=params,
                    x0=x0,
                    v0=v0,
                    d0=d0,
                    d=d,
                    n_steps=int(tdays),
                    refine_radius=int(refine_radius),
                    settings=mam_settings,
                    warm_start=warm,
                )

                u_star_full = np.zeros((int(tdays), 2), dtype=float)
                if sol.tau > 0:
                    u_star_full[: sol.tau] = sol.u_star
                u = cap_control(float(lam) * u_star_full, umax=float(umax))

                if str(seed_mode).lower() == "fixed":
                    qseed = int(base_seed)
                elif str(seed_mode).lower() == "per_query":
                    qseed = _stable_query_seed(int(base_seed), date_str, int(tdays), float(delta))
                else:
                    raise ValueError("seed_mode must be 'fixed' or 'per_query'")

                hit, L = simulate_is_paths(
                    params=params,
                    x0=x0,
                    v0=v0,
                    d0=d0,
                    d_thresh=d,
                    n_steps=int(tdays),
                    u=u,
                    n_paths=int(n_paths),
                    seed=int(qseed),
                )
                res = estimate_probability_is(hit, L)

                inst_path = save_instanton_npz(instantons_dir, date_str, int(tdays), float(delta), sol)
                warm_cache[(int(tdays), float(delta))] = sol

                rows.append(
                    {
                        "Date": date_str,
                        "Tdays": int(tdays),
                        "delta": float(delta),
                        "d": float(d),
                        "tau_star": int(sol.tau),
                        "alpha_m": float(sol.alpha_m),
                        "action": float(sol.action),
                        "dtilde_tau": float(sol.dtilde_tau),
                        "dexact_tau": float(sol.dexact_tau),
                        "exact_shortfall": float(d - sol.dexact_tau),
                        "prehit_max_violation": float(sol.prehit_max_violation),
                        "hit_err": float(sol.hit_err),
                        "lam": float(lam),
                        "umax": float(umax),
                        "N": int(n_paths),
                        "seed_mode": str(seed_mode),
                        "seed_used": int(qseed),
                        "p_hat": float(res.p_hat),
                        "se": float(res.se),
                        "ess": float(res.ess),
                        "mean_weight": float(res.mean_weight),
                        "max_weight": float(res.max_weight),
                        "hit_rate_Q": float(res.hit_rate_under_Q),
                        "instanton_file": str(inst_path.name),
                    }
                )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Daily forecasting loop (instanton + IS) on the committed query grid. Forecasting only."
    )

    ap.add_argument("--demo", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--run_dir", type=str, default=None, help="Explicit run directory. If omitted, created under --runs_root.")
    ap.add_argument("--runs_root", type=str, default="outputs/runs", help="Parent directory for generated run folders.")
    ap.add_argument("--run_name", type=str, default=None, help="Optional custom run folder name.")

    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--state_csv", type=str, default=None, help="Explicit state CSV path. Preferred for reproducibility.")
    ap.add_argument("--state_pattern", type=str, default="state_*.csv")
    ap.add_argument("--params_dir", type=str, default="outputs/params")
    ap.add_argument("--params_json", type=str, default=None, help="Explicit params JSON path. Preferred for reproducibility.")
    ap.add_argument("--params_pattern", type=str, default="sv_params_*.json")
    ap.add_argument("--allow_latest_inputs", action="store_true", help="Allow newest matching state/params files when patterns are ambiguous.")

    ap.add_argument("--tdays", type=int, default=None)
    ap.add_argument("--delta", type=float, default=None)

    ap.add_argument("--n_paths", type=int, default=5000)
    ap.add_argument("--lam", type=float, default=LAMBDA_IS)
    ap.add_argument("--umax", type=float, default=UMAX)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seed_mode", type=str, default="per_query", choices=["fixed", "per_query"])

    ap.add_argument("--mam_max_iter", type=int, default=200)
    ap.add_argument("--mam_step0", type=float, default=0.5)
    ap.add_argument("--mam_grad_eps", type=float, default=1e-4)
    ap.add_argument("--refine_radius", type=int, default=5)

    ap.add_argument("--jobs", type=int, default=1, help="Number of worker processes. 1 = sequential.")
    ap.add_argument("--chunk_days", type=int, default=None, help="Contiguous days per worker block (default: auto).")
    ap.add_argument(
        "--allow_parallel_nondeterminism",
        action="store_true",
        help="Allow jobs>1 even though warm-start differences across blocks can change results.",
    )

    ap.add_argument(
        "--skip_diagnostics_dir",
        action="store_true",
        help="Do not create the diagnostics/ directory in the run folder (forecast CSV + instantons only).",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    if int(args.jobs) > 1 and not bool(args.allow_parallel_nondeterminism):
        raise RuntimeError(
            "jobs>1 is disabled by default for publishable runs because block-local warm starts can change the selected instanton. "
            "Re-run with --allow_parallel_nondeterminism only if you explicitly accept that risk."
        )

    start = pd.to_datetime("2007-01-01")
    end = pd.to_datetime("2008-10-31")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_type = "demo" if bool(args.demo) else "run"

    if args.run_dir:
        run_dir = ensure_dir(args.run_dir)
    else:
        default_name = args.run_name or f"{run_type}_{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}_{stamp}"
        run_dir = ensure_dir(Path(args.runs_root) / default_name)

    forecasts_dir = ensure_dir(run_dir / "forecasts")
    instantons_dir = ensure_dir(run_dir / "instantons")
    diagnostics_dir = run_dir / "diagnostics"
    if not bool(args.skip_diagnostics_dir):
        diagnostics_dir = ensure_dir(diagnostics_dir)

    state_path = resolve_input_file(
        explicit_path=args.state_csv,
        directory=Path(args.processed_dir),
        pattern=str(args.state_pattern),
        allow_latest=bool(args.allow_latest_inputs),
        purpose="state input",
    )
    params_path = resolve_input_file(
        explicit_path=args.params_json,
        directory=Path(args.params_dir),
        pattern=str(args.params_pattern),
        allow_latest=bool(args.allow_latest_inputs),
        purpose="parameter input",
    )

    state_full = read_state_csv(state_path)
    params = load_params_json(params_path)

    tdays_grid = (int(args.tdays),) if args.tdays is not None else tuple(int(x) for x in TDAYS_GRID)
    delta_grid = (float(args.delta),) if args.delta is not None else tuple(float(x) for x in DELTA_GRID)

    state_window = state_full.loc[start:end].copy()
    tasks = make_state_tasks(state_window)
    if len(tasks) == 0:
        raise RuntimeError("No usable state rows in the forced window 2007-01-01 to 2008-10-31.")

    mam_settings = MAMSettings(
        max_iter=int(args.mam_max_iter),
        step0=float(args.mam_step0),
        grad_eps=float(args.mam_grad_eps),
    )

    log.info(
        "%s | %s to %s | days=%d | tdays_grid=%s | delta_grid=%s | N=%d | lam=%s | umax=%s | eps=%g | jobs=%d | seed_mode=%s",
        run_type.upper(),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        len(tasks),
        list(tdays_grid),
        [float(x) for x in delta_grid],
        int(args.n_paths),
        float(args.lam),
        float(args.umax),
        float(HIT_EPS),
        int(args.jobs),
        str(args.seed_mode),
    )
    log.info("Run directory: %s", run_dir.resolve())
    log.info("State input: %s", state_path.resolve())
    log.info("Params input: %s", params_path.resolve())

    t0 = perf_counter()
    rows: list[dict] = []

    if int(args.jobs) <= 1:
        rows = _run_block(
            tasks,
            tdays_grid=tuple(int(x) for x in tdays_grid),
            delta_grid=tuple(float(x) for x in delta_grid),
            params=params,
            n_paths=int(args.n_paths),
            lam=float(args.lam),
            umax=float(args.umax),
            base_seed=int(args.seed),
            seed_mode=str(args.seed_mode),
            refine_radius=int(args.refine_radius),
            mam_settings=mam_settings,
            instantons_dir=Path(instantons_dir),
        )
    else:
        chunk = int(args.chunk_days) if args.chunk_days is not None else int(math.ceil(len(tasks) / int(args.jobs)))
        blocks = [tasks[i : i + chunk] for i in range(0, len(tasks), chunk)]
        log.info("Parallel blocks=%d | chunk_days=%d", len(blocks), chunk)

        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = [
                ex.submit(
                    _run_block,
                    block,
                    tdays_grid=tuple(int(x) for x in tdays_grid),
                    delta_grid=tuple(float(x) for x in delta_grid),
                    params=params,
                    n_paths=int(args.n_paths),
                    lam=float(args.lam),
                    umax=float(args.umax),
                    base_seed=int(args.seed),
                    seed_mode=str(args.seed_mode),
                    refine_radius=int(args.refine_radius),
                    mam_settings=mam_settings,
                    instantons_dir=Path(instantons_dir),
                )
                for block in blocks
            ]

            done = 0
            for fut in as_completed(futs):
                part = fut.result()
                rows.extend(part)
                done += 1
                log.info("Completed blocks: %d/%d (rows=%d)", done, len(futs), len(rows))

    forecasts_df = pd.DataFrame(rows)
    if not forecasts_df.empty:
        forecasts_df = forecasts_df.sort_values(["Date", "Tdays", "delta"]).reset_index(drop=True)

    out_csv = forecasts_dir / f"forecasts_{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}_{stamp}.csv"
    out_meta = out_csv.with_name(out_csv.stem + "_metadata.json")
    forecasts_df.to_csv(out_csv, index=False)

    state_fp = file_fingerprint(state_path)
    params_fp = file_fingerprint(params_path)
    meta_forecasts = {
        "run_type": run_type,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "inputs": {
            "state_source_csv": str(state_path.resolve()),
            "params_json": str(params_path.resolve()),
        },
        "input_fingerprints": {
            "state": state_fp,
            "params": params_fp,
        },
        "params": asdict(params),
        "window": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "grid": {"tdays": [int(x) for x in tdays_grid], "delta": [float(x) for x in delta_grid]},
        "framework": {
            "forward_looking_excludes_origin": False,
            "tau0_regime": "if d0 >= d then tau_d=0 and event already true at origin => p_hat=1 (no instanton/IS needed)",
            "HIT_EPS": float(HIT_EPS),
        },
        "is": {
            "N": int(args.n_paths),
            "lam": float(args.lam),
            "umax": float(args.umax),
            "seed_base": int(args.seed),
            "seed_mode": str(args.seed_mode),
        },
        "mam": {"max_iter": int(args.mam_max_iter), "refine_radius": int(args.refine_radius)},
        "parallel": {
            "jobs": int(args.jobs),
            "chunk_days": int(args.chunk_days) if args.chunk_days is not None else None,
            "allow_parallel_nondeterminism": bool(args.allow_parallel_nondeterminism),
        },
        "outputs": {
            "forecasts_csv": str(out_csv.resolve()),
            "forecasts_meta": str(out_meta.resolve()),
            "instantons_dir": str(Path(instantons_dir).resolve()),
            "diagnostics_dir": str(Path(diagnostics_dir).resolve()),
        },
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    save_json(out_meta, meta_forecasts)

    log.info("Wrote forecasts: %s", out_csv.resolve())
    log.info("Wrote metadata: %s", out_meta.resolve())
    log.info("Total wall time: %.2fs", perf_counter() - t0)
    log.info("Next step: run scripts/04_run_evaluation.py --run_dir %s", run_dir.resolve())


if __name__ == "__main__":
    main()
