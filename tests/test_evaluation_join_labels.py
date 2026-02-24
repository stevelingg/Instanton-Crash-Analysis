from __future__ import annotations

import pandas as pd

from src.evaluation import add_realised_labels_to_forecasts


def test_add_realised_labels_aligns_to_rows() -> None:
    """
    Regression test for a subtle join bug:
    labels were computed per (Tdays, delta) group and concatenated, then assigned by position.
    If forecasts have multiple groups interleaved by date, this misassigns y to the wrong rows.

    Note: tau_star==0 is NOT interpreted as "already-hit => p=1 deterministically".
    It is just a field on the forecast table. The realised label y is defined purely by
    the forward-looking window in the realised D series.
    """
    state = pd.DataFrame(
        {"D": [0.30, 0.24, 0.25, 0.05]},
        index=pd.to_datetime(["2008-09-09", "2008-09-10", "2008-09-11", "2008-09-12"]),
    )

    forecasts = pd.DataFrame(
        [
            {"Date": "2008-09-10", "Tdays": 2, "delta": 0.30, "tau_star": 1, "p_hat": 0.20},
            {"Date": "2008-09-09", "Tdays": 2, "delta": 0.20, "tau_star": 0, "p_hat": 1.00},
            {"Date": "2008-09-10", "Tdays": 2, "delta": 0.20, "tau_star": 0, "p_hat": 1.00},
            {"Date": "2008-09-09", "Tdays": 2, "delta": 0.30, "tau_star": 1, "p_hat": 0.10},
        ]
    )

    out = add_realised_labels_to_forecasts(forecasts, state)

    # add_realised_labels_to_forecasts sorts by (Date, Tdays, delta)
    expected_y = [1.0, 0.0, 1.0, 0.0]
    assert out["y"].to_list() == expected_y