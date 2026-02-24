from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fit import fit_sv_params_qmle_from_state
from src.io import find_latest_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def _read_state_latest() -> pd.DataFrame:
    state_path = find_latest_file(DATA_PROCESSED, "state_*.csv")
    df = pd.read_csv(state_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")


def test_qmle_outputs_are_valid() -> None:
    state = _read_state_latest()
    params, diag = fit_sv_params_qmle_from_state(state, train_end="2006-12-31", maxiter=50)

    assert np.isfinite(params.mu)
    assert params.kappa > 0
    assert params.vbar > 0
    assert params.xi > 0
    assert -1.0 < params.rho < 1.0

    # Because Step 2 uses a variance proxy (not true latent V), eps variances can deviate from 1.
    # We still enforce "reasonable" ranges: not degenerate and not exploding.
    m = diag["moments"]
    assert abs(m["eps1_mean"]) < 0.20
    assert 0.4 < m["eps1_var"] < 2.0
    assert abs(m["eps2_mean"]) < 0.20
    assert 0.4 < m["eps2_var"] < 2.0

    opt = diag["optimiser"]
    assert "success" in opt