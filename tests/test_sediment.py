"""Tests for sediment transport and elevation update."""

import numpy as np

from src.fields import allocate, swap_buffers
from src.flow import compute_flow_fractions
from src.sediment import sediment_transport, update_elevation

# Shared K/P parameters for sediment tests
_KP = {
    "K_max": 0.1,
    "K_min": 0.001,
    "P_min": 0.001,
    "P_max": 0.1,
    "v_low": 5.0,
    "v_high": 20.0,
}


def _setup_grid(n: int):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


def test_sediment_nonnegative():
    """Sediment flux must never go negative."""
    n = 16
    fields = _setup_grid(n)

    rng = np.random.default_rng(42)
    z = rng.uniform(0.0, 10.0, (n, n)).astype(np.float32)
    fields.z.from_numpy(z)

    Q = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    fields.Q_out.from_numpy(Q)
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    for _ in range(10):
        sediment_transport(
            fields.S,
            fields.S_new,
            fields.Q_out,
            fields.z,
            fields.V,
            fields.flow_frac,
            fields.mask,
            dx=1.0,
            gamma=0.01,
            m_exp=1.0,
            n_exp=1.0,
            **_KP,
        )
        swap_buffers(fields.S_new, fields.S)

    S_final = fields.S.to_numpy()
    assert np.all(S_final >= 0.0), f"Negative sediment: min={S_final.min()}"


def test_sediment_increases_downslope():
    """Sediment flux should accumulate downslope (more water, more transport)."""
    n = 16
    fields = _setup_grid(n)

    # Linear slope
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    fields.z.from_numpy(z)

    # Give flow that increases downslope (mimicking routed water)
    Q = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Q[i, :] = float(i) * 0.1
    fields.Q_out.from_numpy(Q)
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    for _ in range(10):
        sediment_transport(
            fields.S,
            fields.S_new,
            fields.Q_out,
            fields.z,
            fields.V,
            fields.flow_frac,
            fields.mask,
            dx=1.0,
            gamma=0.01,
            m_exp=1.0,
            n_exp=1.0,
            **_KP,
        )
        swap_buffers(fields.S_new, fields.S)

    S_final = fields.S.to_numpy()
    mid_col = n // 2
    # Mid-slope cell should have more sediment than upper cell
    # (avoid near-boundary rows where slope_max → 0 due to masked neighbors)
    mid_row = n // 2
    assert S_final[mid_row, mid_col] > S_final[2, mid_col], (
        "Sediment should accumulate downslope"
    )


def test_sediment_zero_on_flat():
    """No sediment transport on a flat surface with no incoming flow."""
    n = 8
    fields = _setup_grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    sediment_transport(
        fields.S,
        fields.S_new,
        fields.Q_out,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        dx=1.0,
        gamma=0.01,
        m_exp=1.0,
        n_exp=1.0,
        **_KP,
    )

    S_final = fields.S_new.to_numpy()
    assert np.allclose(S_final, 0.0, atol=1e-8), (
        f"Sediment should be zero on flat: max={S_final.max()}"
    )


def test_elevation_erodes_bare_soil():
    """Bare soil (low V) should erode under flow (z decreases)."""
    n = 8
    fields = _setup_grid(n)

    # Sloped surface
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    V = np.full((n, n), 1.0, dtype=np.float32)  # bare soil (V <= 5 → K=max)
    fields.V.from_numpy(V)
    Q = np.full((n, n), 0.5, dtype=np.float32)
    fields.Q_out.from_numpy(Q)

    update_elevation(
        fields.z,
        fields.Q_out,
        fields.V,
        fields.mask,
        dx=1.0,
        dt=1.0,
        **_KP,
    )

    z_final = fields.z.to_numpy()
    mask = fields.mask.to_numpy()

    # With V=1 (bare): K=K_max=0.1, P=P_min=0.001 → net erosion
    # dz = dt * (P - K) * Q * slope < 0
    interior = (mask == 1) & (z0 > 1.1)  # cells with slope > 0
    assert np.all(z_final[interior] < z0[interior]), "Bare soil should erode under flow"


def test_elevation_deposits_with_vegetation():
    """Dense vegetation (high V) should favor deposition (z increases)."""
    n = 8
    fields = _setup_grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    V = np.full((n, n), 30.0, dtype=np.float32)  # dense veg (V >= 20 → P=max)
    fields.V.from_numpy(V)
    Q = np.full((n, n), 0.5, dtype=np.float32)
    fields.Q_out.from_numpy(Q)

    update_elevation(
        fields.z,
        fields.Q_out,
        fields.V,
        fields.mask,
        dx=1.0,
        dt=1.0,
        **_KP,
    )

    z_final = fields.z.to_numpy()
    mask = fields.mask.to_numpy()

    # With V=30 (dense): K=K_min=0.001, P=P_max=0.1 → net deposition
    interior = (mask == 1) & (z0 > 1.1)
    assert np.all(z_final[interior] > z0[interior]), (
        "Dense vegetation should cause deposition"
    )
