"""Integration tests for the storm event driver: conservation and runoff-runon feedback."""

import numpy as np
import pytest

from src.fields import allocate
from src.flow import compute_flow_fractions, prepare_levels
from src.params import Params
from src.surface import step_storm


@pytest.fixture
def storm_grid():
    """Factory: allocate fields with slope and precomputed flow fractions for storm tests."""

    def _make(n, slope_pct=1.0, dx=5.0):
        fields = allocate(n)
        mask = np.ones((n, n), dtype=np.int32)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        fields.mask.from_numpy(mask)

        # Linear slope: row 0 high, row n-1 low
        z = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            z[i, :] = float(n - 1 - i) * slope_pct / 100.0 * dx  # [m]
        fields.z.from_numpy(z)

        compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 2.0)
        prepare_levels(fields)
        return fields

    return _make


def test_storm_water_conservation(storm_grid):
    """Total rainfall >= infiltration + remaining surface water.

    With open boundaries, water can exit the domain. The budget is:
    rainfall = infiltration + ponded + boundary_outflow.
    Since we can't track boundary outflow directly, verify that
    infiltration + ponded <= rainfall (no water created).
    """
    n = 12
    fields = storm_grid(n)
    fields.V.fill(10.0)  # moderate vegetation

    params = Params(
        dx=5.0,
        K_s=5.0,  # moderate infiltration
        psi_f=110.0,
        delta_theta=0.35,
        storm_intensity=10.0,
        cfl=0.4,
        dt_max=0.005,
    )

    rain_mm = 5.0  # 5 mm event
    mask = fields.mask.to_numpy()
    n_interior = np.sum(mask == 1)

    step_storm(fields, params, rain_mm)

    h_after = fields.h.to_numpy()
    F_after = fields.F_inf.to_numpy()

    total_ponded = np.sum(h_after[mask == 1])
    total_infiltrated = np.sum(F_after[mask == 1])
    total_input = rain_mm * n_interior

    # No water created: infiltration + ponded <= input
    balance = total_infiltrated + total_ponded
    assert balance <= total_input * 1.01, (
        f"Water created: input={total_input:.2f}, "
        f"infiltrated={total_infiltrated:.2f}, ponded={total_ponded:.2f}"
    )
    # Significant infiltration occurred
    assert total_infiltrated > 0, "No infiltration occurred"
    # Most water accounted for (not all lost to boundary)
    assert balance > total_input * 0.3, (
        f"Too much water lost: balance={balance:.2f} vs input={total_input:.2f}"
    )


def test_storm_runon_concentration(storm_grid):
    """Bare→vegetated transition concentrates infiltration in vegetated zone.

    On a slope, runoff from bare upper cells flows to vegetated lower cells.
    Vegetated cells should infiltrate more than they receive as direct rainfall.
    """
    n = 16
    fields = storm_grid(n, slope_pct=2.0)

    # Upper half bare, lower half vegetated
    V = np.zeros((n, n), dtype=np.float32)
    V[n // 2 :, :] = 40.0  # vegetation in lower half
    fields.V.from_numpy(V)

    params = Params(
        dx=5.0,
        K_s=20.0,  # high K_s so veg zone can absorb runon
        psi_f=110.0,
        delta_theta=0.35,
        storm_intensity=10.0,
        cfl=0.4,
        dt_max=0.005,
    )

    rain_mm = 8.0
    step_storm(fields, params, rain_mm)

    F_after = fields.F_inf.to_numpy()
    mask = fields.mask.to_numpy()

    # Mean infiltration in bare vs vegetated interior rows
    bare_rows = slice(1, n // 2)
    veg_rows = slice(n // 2, n - 1)
    cols = slice(1, n - 1)

    bare_mean = np.mean(F_after[bare_rows, cols][mask[bare_rows, cols] == 1])
    veg_mean = np.mean(F_after[veg_rows, cols][mask[veg_rows, cols] == 1])

    assert veg_mean > bare_mean, (
        f"Vegetated zone should infiltrate more via runon: "
        f"bare={bare_mean:.2f} mm, veg={veg_mean:.2f} mm"
    )


def test_storm_dry_day_no_op(storm_grid):
    """Zero rainfall produces zero infiltration and zero Q_daily."""
    n = 8
    fields = storm_grid(n)
    fields.V.fill(10.0)

    params = Params(dx=5.0)
    step_storm(fields, params, 0.0)

    I_inf = fields.I_inf.to_numpy()
    Q_daily = fields.Q_daily.to_numpy()

    np.testing.assert_allclose(I_inf, 0.0, atol=1e-8)
    np.testing.assert_allclose(Q_daily, 0.0, atol=1e-8)


def test_storm_I_inf_interface(storm_grid):
    """I_inf matches cumulative F_inf after a storm (interface contract)."""
    n = 10
    fields = storm_grid(n)
    fields.V.fill(20.0)

    params = Params(
        dx=5.0,
        K_s=10.0,
        storm_intensity=10.0,
        cfl=0.4,
        dt_max=0.005,
    )

    step_storm(fields, params, 4.0)

    I_inf = fields.I_inf.to_numpy()
    F_inf = fields.F_inf.to_numpy()
    mask = fields.mask.to_numpy()

    # I_inf should equal F_inf for interior cells
    np.testing.assert_allclose(
        I_inf[mask == 1],
        F_inf[mask == 1],
        atol=1e-6,
        err_msg="I_inf should match cumulative F_inf",
    )
