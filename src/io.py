from __future__ import annotations

import json
from typing import Dict, Any, Optional

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import log_drawdown_threshold
from .fit import SVParams
from .importance_sampling import cap_control


def find_latest_file(directory: str | Path, pattern: str) -> Path:
    """Return most recently modified file matching pattern in directory."""
    directory = Path(directory)
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match pattern={pattern!r} in {directory.resolve()}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    

def read_state_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
    return df


def read_forecasts_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise KeyError(f"Forecasts CSV missing 'Date' column: {path}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_latest_state_with_path(processed_dir: Path, pattern: str = "state_*.csv") -> tuple[pd.DataFrame, Path]:
    p = find_latest_file(processed_dir, pattern)
    return read_state_csv(p), p


def load_params_json(path: Path) -> SVParams:
    js = load_json(path)
    return SVParams(**js["params"])


def load_latest_params_with_path(params_dir: Path, pattern: str = "sv_params_*.json") -> tuple[SVParams, Path]:
    p = find_latest_file(params_dir, pattern)
    return load_params_json(p), p


def make_state_tasks(state_window: pd.DataFrame) -> list[tuple[str, float, float, float]]:
    """
    Convert Date-indexed state window into compact tasks:
      (date_str, x0, v0, d0)
    Skips rows with missing X/V_proxy/D.
    """
    tasks: list[tuple[str, float, float, float]] = []
    for date, row in state_window.iterrows():
        if pd.isna(row.get("X")) or pd.isna(row.get("V_proxy")) or pd.isna(row.get("D")):
            continue
        tasks.append((date.strftime("%Y-%m-%d"), float(row["X"]), float(row["V_proxy"]), float(row["D"])))
    return tasks


def load_u_from_cached_instanton(
    instantons_dir: Path,
    row: pd.Series,
    *,
    lam: float,
    umax: float,
) -> np.ndarray:
    """
    Rebuild IS control used in forecasting from cached instanton:
      u_star_full[:tau] = u_star
      u_star_full[tau:] = 0
      u = cap_control(lam * u_star_full, umax)
    """
    tdays = int(row["Tdays"])
    tau = int(row["tau_star"])
    inst_file = str(row.get("instanton_file", ""))

    if not inst_file:
        raise FileNotFoundError("Row has no instanton_file; cannot reconstruct u.")

    p = instantons_dir / inst_file
    if not p.exists():
        raise FileNotFoundError(f"Instanton file not found: {p}")

    data = np.load(p, allow_pickle=False)
    u_star = np.asarray(data["u_star"], dtype=float)  # shape (tau, 2)

    u_star_full = np.zeros((tdays, 2), dtype=float)
    if tau > 0:
        u_star_full[:tau] = u_star

    return cap_control(lam * u_star_full, umax=umax)


def find_forecast_metadata_path(forecasts_csv: Path) -> Path:
    return forecasts_csv.with_name(forecasts_csv.stem + "_metadata.json")


def maybe_load_forecast_metadata(forecasts_csv: Path) -> dict | None:
    p = find_forecast_metadata_path(forecasts_csv)
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None