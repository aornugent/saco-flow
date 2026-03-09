"""Tests for sediment transport and elevation update."""

import numpy as np

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.sediment import sediment_transport, update_elevation


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

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    V = np.full((n, n), 10.0, dtype=np.float32)
    fields.V.from_numpy(V)

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
            K_max=0.1,
            K_min=0.001,
            P_min=0.001,
            P_max=0.1,
            v_low=5.0,
            v_high=20.0,
        )
        fields.swap("S")

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

    V = np.full((n, n), 10.0, dtype=np.float32)
    fields.V.from_numpy(V)

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
            K_max=0.1,
            K_min=0.001,
            P_min=0.001,
            P_max=0.1,
            v_low=5.0,
            v_high=20.0,
        )
        fields.swap("S")

    S_final = fields.S.to_numpy()
    mid_col = n // 2
    # Downslope (higher row) should have more sediment
    assert S_final[n - 2, mid_col] > S_final[2, mid_col], (
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

    V = np.full((n, n), 10.0, dtype=np.float32)
    fields.V.from_numpy(V)

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
        K_max=0.1,
        K_min=0.001,
        P_min=0.001,
        P_max=0.1,
        v_low=5.0,
        v_high=20.0,
    )

    S_final = fields.S_new.to_numpy()
    assert np.allclose(S_final, 0.0, atol=1e-8), (
        f"Sediment should be zero on flat: max={S_final.max()}"
    )


def test_elevation_decreases_on_erosion():
    """z should decrease when S_new > gathered S_0 (net erosion)."""
    n = 8
    fields = _setup_grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    # S = 0 everywhere -> gathered S_0 ~ 0
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    # S_new > 0 -> erosion (outgoing > incoming)
    S_new = np.full((n, n), 1.0, dtype=np.float32)
    fields.S_new.from_numpy(S_new)

    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, 1.0
    )

    z_final = fields.z.to_numpy()
    mask = fields.mask.to_numpy()
    interior = mask == 1
    # S_0 ~ 0, S_new = 1 -> dz = (0 - 1)/1 = -1 -> erosion
    assert np.all(z_final[interior] <= z0[interior]), (
        "Erosion should decrease elevation"
    )


def test_elevation_increases_on_deposition():
    """z should increase when gathered S_0 > S_new (net deposition)."""
    n = 8
    fields = _setup_grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    # S has values -> neighbors gather S_0 > 0
    S = np.full((n, n), 5.0, dtype=np.float32)
    fields.S.from_numpy(S)
    # S_new = 0 -> net deposition
    fields.S_new.from_numpy(np.zeros((n, n), dtype=np.float32))

    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, 1.0
    )

    z_final = fields.z.to_numpy()
    # A downslope cell that receives flow should have z increase
    mid = n // 2
    assert z_final[mid, mid] >= z0[mid, mid], "Deposition should increase elevation"
