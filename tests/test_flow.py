"""Tests for flow routing — MFD fractions and gather-based water routing."""

import numpy as np

from src.fields import allocate, swap_buffers
from src.flow import compute_flow_fractions, route_water


def _setup_grid(n: int):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


def test_flow_fractions_sum_to_one():
    """Outgoing fractions from each interior cell must sum to <= 1."""
    n = 16
    fields = _setup_grid(n)

    # Tilted plane: z = x + y
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            z[i, j] = float(i + j)
    fields.z.from_numpy(z)

    dx, p = 1.0, 1.1
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)

    frac = fields.flow_frac.to_numpy()
    mask = fields.mask.to_numpy()

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 1:
                total = frac[i, j, :].sum()
                assert total <= 1.0 + 1e-5, f"Fractions > 1 at ({i},{j}): {total}"
                # Interior cells on a slope should have some outflow
                if i > 1 or j > 1:  # not at bottom-left corner
                    assert total > 0.0, f"No outflow at ({i},{j})"


def test_flow_fractions_flat_zero():
    """On a flat surface, all fractions should be zero (no slope)."""
    n = 8
    fields = _setup_grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)

    dx, p = 1.0, 1.1
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)

    frac = fields.flow_frac.to_numpy()
    mask = fields.mask.to_numpy()

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 1:
                assert np.allclose(frac[i, j, :], 0.0), (
                    f"Non-zero fraction on flat: ({i},{j})"
                )


def test_flow_fractions_steepest_gets_most():
    """The steepest downslope neighbor should receive the largest fraction."""
    n = 8
    fields = _setup_grid(n)

    # Create a surface where cell (3,3) has one steep neighbor
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[3, 3] = 10.0
    z[4, 3] = 5.0  # steep drop south
    z[3, 4] = 9.0  # gentle drop east
    fields.z.from_numpy(z)

    dx, p = 1.0, 1.1
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)

    frac = fields.flow_frac.to_numpy()
    # Direction 6 = (1,0) = south
    assert frac[3, 3, 6] > frac[3, 3, 4], (
        "Steepest neighbor should get largest fraction"
    )


def test_route_water_rainfall_accumulates():
    """Uniform rainfall should produce increasing Q_out downslope."""
    n = 16
    fields = _setup_grid(n)

    # Linear slope: z decreases with row index
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    fields.z.from_numpy(z)

    R = np.full((n, n), 0.01, dtype=np.float32)  # 10 mm/day
    fields.R.from_numpy(R)

    dx, p = 1.0, 1.1
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)

    # Initialize Q_out to zero
    Q_out_np = np.zeros((n, n), dtype=np.float32)
    fields.Q_out.from_numpy(Q_out_np)
    I_np = np.zeros((n, n), dtype=np.float32)
    fields.I_inf.from_numpy(I_np)

    # Run a few routing steps for flow to propagate
    for _ in range(20):
        route_water(
            fields.Q_out,
            fields.Q_out_new,
            fields.R,
            fields.I_inf,
            fields.h,
            fields.z,
            fields.flow_frac,
            fields.mask,
            dx,
            0.03,
            1.0,
        )
        swap_buffers(fields.Q_out_new, fields.Q_out)

    Q = fields.Q_out.to_numpy()

    # Downslope cells (higher row index) should have more discharge
    mid_col = n // 2
    Q_col = Q[1:-1, mid_col]
    # Check increasing trend (allow some noise from boundary effects)
    assert Q_col[-1] > Q_col[0], (
        f"Q should increase downslope: top={Q_col[0]:.4f}, bottom={Q_col[-1]:.4f}"
    )


def test_route_water_nonnegative():
    """Discharge and flow depth must never go negative."""
    n = 16
    fields = _setup_grid(n)

    rng = np.random.default_rng(42)
    z = rng.uniform(0.0, 10.0, (n, n)).astype(np.float32)
    fields.z.from_numpy(z)

    R = rng.uniform(0.0, 0.05, (n, n)).astype(np.float32)
    fields.R.from_numpy(R)
    fields.I_inf.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))

    dx, p = 1.0, 1.1
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)

    for _ in range(10):
        route_water(
            fields.Q_out,
            fields.Q_out_new,
            fields.R,
            fields.I_inf,
            fields.h,
            fields.z,
            fields.flow_frac,
            fields.mask,
            dx,
            0.03,
            1.0,
        )
        swap_buffers(fields.Q_out_new, fields.Q_out)

    Q = fields.Q_out.to_numpy()
    h = fields.h.to_numpy()
    assert np.all(Q >= 0.0), f"Negative Q: min={Q.min()}"
    assert np.all(h >= 0.0), f"Negative h: min={h.min()}"
