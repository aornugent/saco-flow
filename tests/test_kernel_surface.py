"""Standalone tests for surface water kernels: diffusion wave, Green-Ampt, adaptive dt."""

import numpy as np
import pytest

from src.surface import (
    compute_adaptive_dt,
    diffusion_wave_step,
    infiltration_green_ampt,
)

# ---------------------------------------------------------------------------
# Section 1: Diffusion wave kernel
# ---------------------------------------------------------------------------


def test_diffwave_flat_conservation(grid):
    """Water mound on flat plane conserves total mass (interior, away from boundary)."""
    n = 20  # larger grid so mound doesn't reach boundary
    fields = grid(n)
    # Flat elevation
    fields.z.from_numpy(np.zeros((n, n), dtype=np.float32))

    # Place a water mound in the centre
    h = np.zeros((n, n), dtype=np.float32)
    h[9:11, 9:11] = 10.0  # 10 mm, well away from boundary
    fields.h.from_numpy(h)
    mask = fields.mask.to_numpy()

    total_before = np.sum(h[mask == 1])

    dx = 5.0
    n_M = 0.05
    dt = 0.0001  # very small substep [hr]

    for _ in range(20):  # few steps so water doesn't reach boundary
        diffusion_wave_step(fields.h, fields.h_new, fields.z, fields.mask, dx, dt, n_M)
        fields.swap("h")

    h_after = fields.h.to_numpy()
    total_after = np.sum(h_after[mask == 1])

    assert total_after == pytest.approx(total_before, rel=1e-4), (
        f"Mass not conserved: {total_before:.4f} -> {total_after:.4f}"
    )


def test_diffwave_slope_drains(grid):
    """Water on a tilted plane flows downhill; mass conserved."""
    n = 10
    fields = grid(n)

    # Linear slope: row 0 is high, row n-1 is low
    dx = 5.0  # [m]
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.01 * dx  # 1% slope [m]
    fields.z.from_numpy(z)

    # Uniform thin layer on upper half
    h = np.zeros((n, n), dtype=np.float32)
    h[1:5, 1:-1] = 5.0  # 5 mm
    fields.h.from_numpy(h)
    mask = fields.mask.to_numpy()

    total_before = np.sum(h[mask == 1])
    mean_row_before = np.mean(h[1:-1, 1:-1], axis=1)

    n_M = 0.05
    dt = 0.0001  # [hr]

    for _ in range(200):
        diffusion_wave_step(fields.h, fields.h_new, fields.z, fields.mask, dx, dt, n_M)
        fields.swap("h")

    h_after = fields.h.to_numpy()
    total_after = np.sum(h_after[mask == 1])
    mean_row_after = np.mean(h_after[1:-1, 1:-1], axis=1)

    # With open boundaries, some water exits the domain
    assert total_after <= total_before, "Water created from nothing"
    assert total_after < total_before, "No water drained (expected boundary outflow)"

    # Water shifted downhill: centre of mass moved to higher row index
    rows = np.arange(1, n - 1)
    com_before = np.average(rows, weights=mean_row_before + 1e-12)
    com_after = np.average(rows, weights=mean_row_after + 1e-12)
    assert com_after > com_before, "Water did not flow downhill"


def test_diffwave_cfl_stability(grid):
    """No negative depths at CFL-appropriate dt."""
    n = 10
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.01
    fields.z.from_numpy(z)

    h = np.zeros((n, n), dtype=np.float32)
    h[3, 5] = 50.0  # single-cell spike, 50 mm
    fields.h.from_numpy(h)

    dx = 5.0
    n_M = 0.05
    dt = 0.0001  # small enough for CFL

    for _ in range(100):
        diffusion_wave_step(fields.h, fields.h_new, fields.z, fields.mask, dx, dt, n_M)
        fields.swap("h")

    h_after = fields.h.to_numpy()
    assert np.all(h_after >= 0.0), "Negative water depths detected"


def test_diffwave_no_flow_without_gradient(grid):
    """Uniform h on flat plane: interior cells away from boundary unchanged.

    Cells adjacent to boundary lose water to open boundary (h_boundary=0),
    but deep interior cells with uniform eta see no gradient.
    """
    n = 12
    fields = grid(n)
    fields.z.from_numpy(np.zeros((n, n), dtype=np.float32))

    h = np.ones((n, n), dtype=np.float32) * 5.0
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0  # boundary
    fields.h.from_numpy(h)

    diffusion_wave_step(fields.h, fields.h_new, fields.z, fields.mask, 5.0, 0.001, 0.05)

    h_new = fields.h_new.to_numpy()
    # Deep interior cells (>1 cell from boundary) should be unchanged
    deep_interior = h_new[3:-3, 3:-3]
    np.testing.assert_allclose(deep_interior, 5.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 2: Green-Ampt infiltration kernel
# ---------------------------------------------------------------------------


def test_greenampt_dry_soil_rate(grid):
    """Initial rate on dry soil approximates K_s * (1 + psi*dtheta/F) for small F."""
    n = 6
    fields = grid(n)

    h = np.ones((n, n), dtype=np.float32) * 100.0  # plenty of ponded water
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0
    fields.h.from_numpy(h)
    fields.F_inf.from_numpy(np.full((n, n), 0.01, dtype=np.float32))  # near-zero F
    fields.V.from_numpy(
        np.ones((n, n), dtype=np.float32) * 100.0
    )  # high veg → K_eff ≈ K_s

    K_s = 15.0  # mm/hr
    psi_f = 110.0
    delta_theta = 0.35
    k2 = 18.0
    W0 = 0.05
    dt = 0.001  # hr
    F0 = 5.0  # moderate initial F to avoid extreme rate / f32 precision issues

    fields.F_inf.from_numpy(np.full((n, n), F0, dtype=np.float32))

    infiltration_green_ampt(
        fields.h,
        fields.F_inf,
        fields.V,
        fields.mask,
        K_s,
        psi_f,
        delta_theta,
        k2,
        W0,
        dt,
    )

    F_after = fields.F_inf.to_numpy()
    mask = fields.mask.to_numpy()
    I_vol = F_after[mask == 1] - F0
    # K_eff = K_s * (V + k2*W0) / (V + k2) with V=100
    V_test = 100.0
    K_eff = K_s * (V_test + k2 * W0) / (V_test + k2)
    expected_rate = K_eff * (1.0 + psi_f * delta_theta / F0)
    expected_vol = expected_rate * dt

    np.testing.assert_allclose(I_vol, expected_vol, rtol=0.05)


def test_greenampt_ponded_bounded(grid):
    """Infiltrated volume never exceeds ponded depth."""
    n = 6
    fields = grid(n)

    h = np.ones((n, n), dtype=np.float32) * 0.1  # very thin ponding, 0.1 mm
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0
    fields.h.from_numpy(h)
    fields.F_inf.fill(0.01)
    fields.V.fill(50.0)

    K_s = 15.0
    psi_f = 110.0
    delta_theta = 0.35
    dt = 1.0  # large dt to force rate * dt >> h

    infiltration_green_ampt(
        fields.h,
        fields.F_inf,
        fields.V,
        fields.mask,
        K_s,
        psi_f,
        delta_theta,
        18.0,
        0.05,
        dt,
    )

    h_after = fields.h.to_numpy()
    assert np.all(h_after >= 0.0), "Negative h after infiltration"

    # Interior cells should have h = 0 (all infiltrated)
    mask = fields.mask.to_numpy()
    np.testing.assert_allclose(h_after[mask == 1], 0.0, atol=1e-6)


def test_greenampt_vegetation_contrast(grid):
    """Higher infiltration under vegetation than bare soil."""
    n = 8
    fields = grid(n)

    h = np.ones((n, n), dtype=np.float32) * 20.0
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0
    fields.h.from_numpy(h)
    fields.F_inf.fill(1.0)  # some prior infiltration

    # Left half bare (V=0), right half vegetated (V=50)
    V = np.zeros((n, n), dtype=np.float32)
    V[:, n // 2 :] = 50.0
    fields.V.from_numpy(V)

    K_s = 15.0
    psi_f = 110.0
    delta_theta = 0.35
    k2 = 18.0
    W0 = 0.05
    dt = 0.01

    infiltration_green_ampt(
        fields.h,
        fields.F_inf,
        fields.V,
        fields.mask,
        K_s,
        psi_f,
        delta_theta,
        k2,
        W0,
        dt,
    )

    F_after = fields.F_inf.to_numpy()

    bare_inf = np.mean(F_after[1:-1, 1 : n // 2])
    veg_inf = np.mean(F_after[1:-1, n // 2 : -1])
    assert veg_inf > bare_inf, (
        f"Vegetation should infiltrate more: bare={bare_inf:.4f}, veg={veg_inf:.4f}"
    )


# ---------------------------------------------------------------------------
# Section 3: Adaptive dt kernel
# ---------------------------------------------------------------------------


def test_adaptive_dt_scales_with_h(grid):
    """dt_adapt decreases as max(h) increases; equals dt_max when dry."""
    n = 8
    fields = grid(n)

    # Tilted plane so slope is non-trivial (1% grade)
    dx = 5.0
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.01 * dx  # [m]
    fields.z.from_numpy(z)

    n_M = 0.05
    cfl = 0.4
    dt_max = 0.01  # hr

    # Dry domain → dt_max
    fields.h.fill(0.0)
    compute_adaptive_dt(
        fields.h, fields.z, fields.mask, dx, n_M, cfl, dt_max, fields.dt_adapt
    )
    assert fields.dt_adapt[None] == pytest.approx(dt_max)

    # Small h → large dt
    h = np.zeros((n, n), dtype=np.float32)
    h[4, 4] = 1.0  # 1 mm
    fields.h.from_numpy(h)
    compute_adaptive_dt(
        fields.h, fields.z, fields.mask, dx, n_M, cfl, dt_max, fields.dt_adapt
    )
    dt_small_h = fields.dt_adapt[None]

    # Large h → small dt
    h[4, 4] = 100.0  # 100 mm
    fields.h.from_numpy(h)
    compute_adaptive_dt(
        fields.h, fields.z, fields.mask, dx, n_M, cfl, dt_max, fields.dt_adapt
    )
    dt_large_h = fields.dt_adapt[None]

    assert dt_large_h < dt_small_h, (
        f"Larger h should give smaller dt: dt(1mm)={dt_small_h:.6f}, dt(100mm)={dt_large_h:.6f}"
    )
