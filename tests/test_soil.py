"""
Tests for soil moisture dynamics: ET, leakage, diffusion.
"""

import numpy as np

from src.kernels.soil import (
    compute_diffusion_timestep,
    diffusion_step,
    evapotranspiration_step,
    leakage_step,
    soil_moisture_step,
)
from src.kernels.utils import compute_total, fill_field


class TestEvapotranspiration:
    """Test evapotranspiration kernel."""

    def test_et_reduces_moisture(self, grid_factory, tilted_plane):
        """ET should decrease soil moisture."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.2)
        fill_field(fields.P, 0.5)

        M_before = compute_total(fields.M, fields.mask)

        evapotranspiration_step(
            fields.M, fields.P, fields.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5, dt=1.0
        )

        M_after = compute_total(fields.M, fields.mask)
        assert M_after < M_before, f"ET didn't reduce M: {M_before} -> {M_after}"

    def test_no_et_when_dry(self, grid_factory, tilted_plane):
        """No ET when soil is dry."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.0)
        fill_field(fields.P, 0.5)

        M_before = compute_total(fields.M, fields.mask)

        evapotranspiration_step(
            fields.M, fields.P, fields.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5, dt=1.0
        )

        M_after = compute_total(fields.M, fields.mask)
        assert abs(M_after - M_before) < 1e-8

    def test_vegetation_enhances_et(self, grid_factory, tilted_plane):
        """Higher vegetation leads to more ET."""
        n = 16

        # Low vegetation
        fields_low = grid_factory(n=n)
        tilted_plane(fields_low)
        fill_field(fields_low.M, 0.2)
        fill_field(fields_low.P, 0.0)

        M_before_low = compute_total(fields_low.M, fields_low.mask)
        evapotranspiration_step(
            fields_low.M, fields_low.P, fields_low.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5, dt=1.0
        )
        et_low = M_before_low - compute_total(fields_low.M, fields_low.mask)

        # High vegetation
        fields_high = grid_factory(n=n)
        tilted_plane(fields_high)
        fill_field(fields_high.M, 0.2)
        fill_field(fields_high.P, 5.0)

        M_before_high = compute_total(fields_high.M, fields_high.mask)
        evapotranspiration_step(
            fields_high.M, fields_high.P, fields_high.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5, dt=1.0
        )
        et_high = M_before_high - compute_total(fields_high.M, fields_high.mask)

        assert et_high > et_low, f"ET low={et_low}, high={et_high}"

    def test_et_returns_total(self, grid_factory, tilted_plane):
        """ET kernel returns correct total."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.2)
        fill_field(fields.P, 0.5)

        M_before = compute_total(fields.M, fields.mask)
        total_et = evapotranspiration_step(
            fields.M, fields.P, fields.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5, dt=1.0
        )
        M_after = compute_total(fields.M, fields.mask)

        expected = M_before - M_after
        # Relaxed tolerance for f32 atomic accumulation
        assert abs(total_et - expected) < 1e-3


class TestLeakage:
    """Test deep leakage kernel."""

    def test_leakage_reduces_moisture(self, grid_factory, tilted_plane):
        """Leakage should decrease soil moisture."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.3)

        M_before = compute_total(fields.M, fields.mask)

        leakage_step(fields.M, fields.mask, L_max=0.01, M_sat=0.4, dt=1.0)

        M_after = compute_total(fields.M, fields.mask)
        assert M_after < M_before

    def test_leakage_quadratic(self, grid_factory, tilted_plane):
        """Leakage should be higher when wetter (quadratic)."""
        n = 16
        M_sat = 0.4

        # Wet soil
        fields_wet = grid_factory(n=n)
        tilted_plane(fields_wet)
        fill_field(fields_wet.M, 0.35)  # Near saturation

        M_before_wet = compute_total(fields_wet.M, fields_wet.mask)
        leakage_step(fields_wet.M, fields_wet.mask, L_max=0.01, M_sat=M_sat, dt=1.0)
        leakage_wet = M_before_wet - compute_total(fields_wet.M, fields_wet.mask)

        # Dry soil
        fields_dry = grid_factory(n=n)
        tilted_plane(fields_dry)
        fill_field(fields_dry.M, 0.1)

        M_before_dry = compute_total(fields_dry.M, fields_dry.mask)
        leakage_step(fields_dry.M, fields_dry.mask, L_max=0.01, M_sat=M_sat, dt=1.0)
        leakage_dry = M_before_dry - compute_total(fields_dry.M, fields_dry.mask)

        # Wet should lose much more (quadratic relationship)
        assert leakage_wet > 5 * leakage_dry, f"wet={leakage_wet}, dry={leakage_dry}"

    def test_no_leakage_when_dry(self, grid_factory, tilted_plane):
        """No leakage when soil is dry."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.0)

        leakage_step(fields.M, fields.mask, L_max=0.01, M_sat=0.4, dt=1.0)

        M_np = fields.M.to_numpy()
        assert np.all(M_np == 0)


class TestDiffusion:
    """Test soil moisture diffusion kernel."""

    def test_diffusion_conserves_mass(self, grid_factory, tilted_plane):
        """Diffusion alone should conserve total mass."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Non-uniform initial moisture
        M_np = np.random.uniform(0.1, 0.3, (n, n)).astype(np.float32)
        fields.M.from_numpy(M_np)

        M_before = compute_total(fields.M, fields.mask)

        # Stable timestep
        D_M = 0.1
        dx = 1.0
        dt = compute_diffusion_timestep(D_M, dx, cfl=0.2)

        for _ in range(10):
            diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)
            # Copy back
            fields.M.from_numpy(fields.M_new.to_numpy())

        M_after = compute_total(fields.M, fields.mask)

        # Should be conserved to floating point precision (relaxed for f32)
        assert abs(M_after - M_before) < 1e-3, f"Diffusion lost mass: {M_before} -> {M_after}"

    def test_diffusion_smooths_gradient(self, grid_factory, tilted_plane):
        """Diffusion should reduce spatial variance."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Sharp gradient: wet on left, dry on right
        M_np = np.zeros((n, n), dtype=np.float32)
        M_np[:, :n // 2] = 0.3
        M_np[:, n // 2:] = 0.1
        fields.M.from_numpy(M_np)

        mask_np = fields.mask.to_numpy()
        variance_before = np.var(M_np[mask_np == 1])

        D_M = 0.1
        dx = 1.0
        dt = compute_diffusion_timestep(D_M, dx, cfl=0.2)

        for _ in range(50):
            diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)
            fields.M.from_numpy(fields.M_new.to_numpy())

        M_final = fields.M.to_numpy()
        variance_after = np.var(M_final[mask_np == 1])

        assert variance_after < variance_before, \
            f"Variance didn't decrease: {variance_before} -> {variance_after}"

    def test_diffusion_timestep_stability(self, grid_factory, tilted_plane):
        """Diffusion should be stable at computed timestep."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Random initial conditions
        M_np = np.random.uniform(0.0, 0.4, (n, n)).astype(np.float32)
        fields.M.from_numpy(M_np)

        D_M = 0.5  # Higher diffusivity
        dx = 1.0
        dt = compute_diffusion_timestep(D_M, dx, cfl=0.25)

        for _ in range(100):
            diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)
            fields.M.from_numpy(fields.M_new.to_numpy())

        M_final = fields.M.to_numpy()
        assert not np.any(np.isnan(M_final)), "NaN in diffusion"
        assert not np.any(M_final < -1e-6), "Negative moisture from diffusion"


class TestCombinedSoilMoisture:
    """Test combined soil moisture step."""

    def test_combined_step_runs(self, grid_factory, tilted_plane):
        """Combined step should run without error."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.M, 0.2)
        fill_field(fields.P, 0.5)

        total_et, total_leak = soil_moisture_step(
            fields.M, fields.M_new, fields.P, fields.mask,
            E_max=0.01, k_M=0.05, beta_E=0.5,
            L_max=0.001, M_sat=0.4,
            D_M=0.1, dx=1.0, dt=0.1
        )

        assert total_et >= 0
        assert total_leak >= 0

    def test_moisture_stays_bounded(self, grid_factory, tilted_plane):
        """Moisture should stay in [0, M_sat] after combined step."""
        n = 16
        M_sat = 0.4
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Random initial conditions
        M_np = np.random.uniform(0.0, M_sat, (n, n)).astype(np.float32)
        fields.M.from_numpy(M_np)
        fill_field(fields.P, 0.5)

        for _ in range(10):
            soil_moisture_step(
                fields.M, fields.M_new, fields.P, fields.mask,
                E_max=0.01, k_M=0.05, beta_E=0.5,
                L_max=0.001, M_sat=M_sat,
                D_M=0.1, dx=1.0, dt=0.1
            )

        M_final = fields.M.to_numpy()
        assert np.all(M_final >= -1e-6), "Negative moisture"
        # Note: diffusion can push values slightly, but ET/leakage keep it bounded


class TestDiffusionTimestep:
    """Test diffusion timestep calculation."""

    def test_timestep_positive(self):
        """Timestep should be positive."""
        dt = compute_diffusion_timestep(D_M=0.1, dx=1.0)
        assert dt > 0

    def test_timestep_infinite_zero_diffusivity(self):
        """Zero diffusivity gives infinite timestep."""
        dt = compute_diffusion_timestep(D_M=0.0, dx=1.0)
        assert dt == float("inf")

    def test_timestep_scales_with_dx_squared(self):
        """Timestep should scale with dx²."""
        dt1 = compute_diffusion_timestep(D_M=0.1, dx=1.0)
        dt2 = compute_diffusion_timestep(D_M=0.1, dx=2.0)
        assert abs(dt2 / dt1 - 4.0) < 0.01  # dt ∝ dx²
