"""Tests for the daily/annual simulation orchestrator."""

import numpy as np

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.simulate import step_day, step_year


def _setup_grid(n: int):
    """Allocate fields with boundary mask, slope, rainfall, and initial conditions."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    fields.z.from_numpy(z)

    fields.R.from_numpy(np.full((n, n), 0.01, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 5.0, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)
    return fields


_DAILY = {
    "dx": 1.0,
    "n_manning": 0.03,
    "cn": 1.0,
    "alpha": 1.0,
    "k2": 5.0,
    "W0": 0.2,
    "g_max": 0.1,
    "k1": 0.1,
    "rw": 0.01,
    "c": 1.0,
    "d": 0.01,
    "Dp": 0.01,
    "c1": 0.01,
    "c2": 1.0,
    "dt": 1.0,
    "n_picard": 10,
}


def test_step_day_nonnegative():
    """All state variables must remain non-negative after one day."""
    n = 16
    fields = _setup_grid(n)

    step_day(fields, **_DAILY)

    assert np.all(fields.Q_out.to_numpy() >= 0)
    assert np.all(fields.M.to_numpy() >= 0)
    assert np.all(fields.V.to_numpy() >= 0)


def test_step_day_produces_flow():
    """After one daily step with rainfall, some cells should have discharge."""
    n = 16
    fields = _setup_grid(n)

    step_day(fields, **_DAILY)

    Q = fields.Q_out.to_numpy()
    mask = fields.mask.to_numpy()
    assert np.sum(Q[mask == 1]) > 0, "Should produce some discharge"


def test_step_year_runs():
    """step_year should complete without error on a small grid."""
    n = 8
    fields = _setup_grid(n)

    step_year(
        fields,
        p=1.1,
        gamma=0.01,
        m_exp=1.0,
        n_exp=1.0,
        K_max=0.1,
        K_min=0.001,
        P_min=0.001,
        P_max=0.1,
        v_low=5.0,
        v_high=20.0,
        days_per_year=3,  # short for speed
        **_DAILY,
    )

    assert np.all(fields.Q_out.to_numpy() >= 0)
    assert np.all(fields.V.to_numpy() >= 0)
    assert np.all(fields.M.to_numpy() >= 0)
