"""Tests for soil moisture dynamics — infiltration, uptake, and drainage."""

import numpy as np

from src.fields import allocate
from src.soil_moisture import infiltration_step, soil_moisture_step


def _setup_grid(n: int):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


def test_infiltration_increases_with_depth():
    """Higher flow depth should produce higher infiltration."""
    n = 8
    fields = _setup_grid(n)

    V0 = np.full((n, n), 10.0, dtype=np.float32)
    fields.V.from_numpy(V0)

    h_low = np.full((n, n), 0.01, dtype=np.float32)
    h_high = np.full((n, n), 0.10, dtype=np.float32)

    fields.h.from_numpy(h_low)
    infiltration_step(
        fields.I_inf, fields.h, fields.V, fields.mask, alpha=1.0, k2=5.0, W0=0.2
    )
    I_low = fields.I_inf.to_numpy().copy()

    fields.h.from_numpy(h_high)
    infiltration_step(
        fields.I_inf, fields.h, fields.V, fields.mask, alpha=1.0, k2=5.0, W0=0.2
    )
    I_high = fields.I_inf.to_numpy()

    mask = fields.mask.to_numpy()
    assert np.all(I_high[mask == 1] > I_low[mask == 1]), (
        "Higher flow depth should give more infiltration"
    )


def test_infiltration_increases_with_vegetation():
    """More vegetation should enhance infiltration."""
    n = 8
    fields = _setup_grid(n)

    h0 = np.full((n, n), 0.05, dtype=np.float32)
    fields.h.from_numpy(h0)

    V_low = np.full((n, n), 1.0, dtype=np.float32)
    V_high = np.full((n, n), 20.0, dtype=np.float32)

    fields.V.from_numpy(V_low)
    infiltration_step(
        fields.I_inf, fields.h, fields.V, fields.mask, alpha=1.0, k2=5.0, W0=0.2
    )
    I_low = fields.I_inf.to_numpy().copy()

    fields.V.from_numpy(V_high)
    infiltration_step(
        fields.I_inf, fields.h, fields.V, fields.mask, alpha=1.0, k2=5.0, W0=0.2
    )
    I_high = fields.I_inf.to_numpy()

    mask = fields.mask.to_numpy()
    assert np.all(I_high[mask == 1] > I_low[mask == 1]), (
        "Higher vegetation should enhance infiltration"
    )


def test_soil_moisture_increases_with_infiltration():
    """Soil moisture should increase when infiltration exceeds losses."""
    n = 8
    fields = _setup_grid(n)

    M0 = np.full((n, n), 0.1, dtype=np.float32)
    fields.M.from_numpy(M0)
    V0 = np.full((n, n), 1.0, dtype=np.float32)  # low vegetation
    fields.V.from_numpy(V0)
    I0 = np.full((n, n), 0.5, dtype=np.float32)  # high infiltration
    fields.I_inf.from_numpy(I0)

    # I=0.5 >> uptake + drainage for small V and small M
    soil_moisture_step(
        fields.M,
        fields.M_new,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=0.01,
        k1=0.1,
        rw=0.01,
        dt=1.0,
    )

    M_final = fields.M_new.to_numpy()
    mask = fields.mask.to_numpy()
    interior = M_final[mask == 1]
    assert np.all(interior > 0.1), f"Moisture should increase: min={interior.min():.4f}"


def test_soil_moisture_decreases_without_infiltration():
    """Without infiltration, moisture should decrease from uptake and loss."""
    n = 8
    fields = _setup_grid(n)

    M0 = np.full((n, n), 0.5, dtype=np.float32)
    fields.M.from_numpy(M0)
    V0 = np.full((n, n), 10.0, dtype=np.float32)
    fields.V.from_numpy(V0)
    I0 = np.zeros((n, n), dtype=np.float32)  # no infiltration
    fields.I_inf.from_numpy(I0)

    soil_moisture_step(
        fields.M,
        fields.M_new,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=0.1,
        k1=0.1,
        rw=0.1,
        dt=1.0,
    )

    M_final = fields.M_new.to_numpy()
    mask = fields.mask.to_numpy()
    interior = M_final[mask == 1]
    assert np.all(interior < 0.5), f"Moisture should decrease: max={interior.max():.4f}"


def test_soil_moisture_nonnegative():
    """Soil moisture must never go negative."""
    n = 8
    fields = _setup_grid(n)

    rng = np.random.default_rng(55)
    M0 = rng.uniform(0.0, 0.01, (n, n)).astype(np.float32)
    fields.M.from_numpy(M0)
    V0 = np.full((n, n), 20.0, dtype=np.float32)  # heavy uptake
    fields.V.from_numpy(V0)
    I0 = np.zeros((n, n), dtype=np.float32)
    fields.I_inf.from_numpy(I0)

    for _ in range(20):
        soil_moisture_step(
            fields.M,
            fields.M_new,
            fields.I_inf,
            fields.V,
            fields.mask,
            g_max=0.5,
            k1=0.1,
            rw=0.5,
            dt=1.0,
        )
        fields.swap("M")

    M_final = fields.M.to_numpy()
    assert np.all(M_final >= 0.0), f"Negative moisture: min={M_final.min()}"
