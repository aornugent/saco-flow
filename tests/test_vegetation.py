"""Tests for vegetation dynamics — growth, mortality, and dispersal."""

import numpy as np

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.vegetation import vegetation_step


def _setup_grid(n: int):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


def test_vegetation_growth_with_moisture():
    """Vegetation should increase when soil moisture is high."""
    n = 8
    fields = _setup_grid(n)

    V0 = np.full((n, n), 5.0, dtype=np.float32)
    M0 = np.full((n, n), 0.5, dtype=np.float32)  # plenty of moisture
    fields.V.from_numpy(V0)
    fields.M.from_numpy(M0)
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))

    # Flat terrain -> no flow dispersal
    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    # Growth > mortality: c*g_max*M/(M+k1)*V - d*V > 0
    # With c=1, g_max=0.1, k1=0.1, d=0.01: 1*0.1*0.5/0.6*5 - 0.01*5 = 0.367
    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.flow_frac,
        fields.mask,
        c=1.0,
        g_max=0.1,
        k1=0.1,
        d=0.01,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )

    V_final = fields.V_new.to_numpy()
    mask = fields.mask.to_numpy()
    interior = V_final[mask == 1]
    assert np.all(interior > 5.0), f"Vegetation should grow: min={interior.min():.4f}"


def test_vegetation_mortality_no_water():
    """Without moisture, vegetation should decline due to mortality."""
    n = 8
    fields = _setup_grid(n)

    V0 = np.full((n, n), 10.0, dtype=np.float32)
    M0 = np.full((n, n), 0.0, dtype=np.float32)  # no moisture
    fields.V.from_numpy(V0)
    fields.M.from_numpy(M0)
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.flow_frac,
        fields.mask,
        c=1.0,
        g_max=0.1,
        k1=0.1,
        d=0.05,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )

    V_final = fields.V_new.to_numpy()
    mask = fields.mask.to_numpy()
    interior = V_final[mask == 1]
    assert np.all(interior < 10.0), (
        f"Vegetation should decline: max={interior.max():.4f}"
    )


def test_vegetation_diffusion_smooths():
    """Isotropic seed dispersal should smooth a vegetation step function."""
    n = 16
    fields = _setup_grid(n)

    V0 = np.zeros((n, n), dtype=np.float32)
    V0[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 20.0
    fields.V.from_numpy(V0)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    # Run with strong diffusion, zero growth/mortality/flow-dispersal
    for _ in range(50):
        vegetation_step(
            fields.V,
            fields.V_new,
            fields.M,
            fields.Q_out,
            fields.flow_frac,
            fields.mask,
            c=0.0,
            g_max=0.0,
            k1=0.1,
            d=0.0,
            Dp=0.5,
            dx=1.0,
            c1=0.0,
            c2=1.0,
            dt=0.1,
        )
        fields.swap("V")

    V_final = fields.V.to_numpy()
    mask = fields.mask.to_numpy()
    interior = V_final[mask == 1]

    # Standard deviation should be reduced from initial step
    assert np.std(interior) < 8.0, (
        f"Seed dispersal did not smooth: std={np.std(interior):.4f}"
    )


def test_vegetation_nonnegative():
    """Vegetation density must never go negative."""
    n = 8
    fields = _setup_grid(n)

    rng = np.random.default_rng(77)
    V0 = rng.uniform(0.0, 0.5, (n, n)).astype(np.float32)
    fields.V.from_numpy(V0)
    fields.M.from_numpy(np.full((n, n), 0.0, dtype=np.float32))
    fields.Q_out.from_numpy(np.zeros((n, n), dtype=np.float32))

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)

    for _ in range(50):
        vegetation_step(
            fields.V,
            fields.V_new,
            fields.M,
            fields.Q_out,
            fields.flow_frac,
            fields.mask,
            c=1.0,
            g_max=0.1,
            k1=0.1,
            d=0.5,
            Dp=0.1,
            dx=1.0,
            c1=0.0,
            c2=1.0,
            dt=0.1,
        )
        fields.swap("V")

    V_final = fields.V.to_numpy()
    assert np.all(V_final >= 0.0), f"Negative vegetation: min={V_final.min()}"
