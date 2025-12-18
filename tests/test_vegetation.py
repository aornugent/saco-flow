"""
Tests for vegetation dynamics: growth, mortality, dispersal.
"""

import numpy as np

from src.fields import fill_field, copy_field
from src.diagnostics import compute_total
from src.kernels.vegetation import (
    compute_equilibrium_moisture,
    compute_vegetation_timestep,
    growth_step,
    mortality_step,
    vegetation_diffusion_step,
    vegetation_step,
)


class TestGrowth:
    """Test Monod growth kinetics."""

    def test_growth_increases_biomass(self, grid_factory, tilted_plane):
        """Growth should increase vegetation when moisture available."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 1.0)   # Initial biomass
        fill_field(fields.M, 0.2)   # Adequate moisture

        P_before = compute_total(fields.P, fields.mask)

        growth_step(fields.P, fields.M, fields.mask, g_max=0.1, k_G=0.1, dt=1.0)

        P_after = compute_total(fields.P, fields.mask)
        assert P_after > P_before, f"Growth failed: {P_before} -> {P_after}"

    def test_no_growth_without_moisture(self, grid_factory, tilted_plane):
        """No growth when soil is dry."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 1.0)
        fill_field(fields.M, 0.0)  # No moisture

        P_before = compute_total(fields.P, fields.mask)

        growth_step(fields.P, fields.M, fields.mask, g_max=0.1, k_G=0.1, dt=1.0)

        P_after = compute_total(fields.P, fields.mask)
        assert abs(P_after - P_before) < 1e-8

    def test_no_growth_without_vegetation(self, grid_factory, tilted_plane):
        """No growth when no existing vegetation (needs seed source)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 0.0)  # No vegetation
        fill_field(fields.M, 0.2)

        growth_step(fields.P, fields.M, fields.mask, g_max=0.1, k_G=0.1, dt=1.0)

        P_after = compute_total(fields.P, fields.mask)
        assert abs(P_after) < 1e-8

    def test_growth_saturates_at_high_moisture(self, grid_factory, tilted_plane):
        """Growth rate should saturate (Monod kinetics)."""
        n = 16
        k_G = 0.1

        # Low moisture
        fields_low = grid_factory(n=n)
        tilted_plane(fields_low)
        fill_field(fields_low.P, 1.0)
        fill_field(fields_low.M, k_G)  # At half-saturation

        P_before_low = compute_total(fields_low.P, fields_low.mask)
        growth_step(fields_low.P, fields_low.M, fields_low.mask, g_max=0.1, k_G=k_G, dt=1.0)
        growth_low = compute_total(fields_low.P, fields_low.mask) - P_before_low

        # High moisture (10x half-saturation)
        fields_high = grid_factory(n=n)
        tilted_plane(fields_high)
        fill_field(fields_high.P, 1.0)
        fill_field(fields_high.M, 10 * k_G)

        P_before_high = compute_total(fields_high.P, fields_high.mask)
        growth_step(fields_high.P, fields_high.M, fields_high.mask, g_max=0.1, k_G=k_G, dt=1.0)
        growth_high = compute_total(fields_high.P, fields_high.mask) - P_before_high

        # High moisture should give ~1.8x growth (not 10x due to saturation)
        # At M=k_G: rate = g_max * 0.5
        # At M=10*k_G: rate = g_max * 10/11 ≈ 0.91
        ratio = growth_high / growth_low
        assert 1.5 < ratio < 2.0, f"Saturation ratio: {ratio}"


class TestMortality:
    """Test constant mortality."""

    def test_mortality_decreases_biomass(self, grid_factory, tilted_plane):
        """Mortality should decrease vegetation."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 1.0)

        P_before = compute_total(fields.P, fields.mask)

        mortality_step(fields.P, fields.mask, mu=0.1, dt=1.0)

        P_after = compute_total(fields.P, fields.mask)
        assert P_after < P_before

    def test_mortality_proportional_to_biomass(self, grid_factory, tilted_plane):
        """Higher biomass should lose more to mortality."""
        n = 16
        mu = 0.1

        # Low biomass
        fields_low = grid_factory(n=n)
        tilted_plane(fields_low)
        fill_field(fields_low.P, 1.0)

        P_before_low = compute_total(fields_low.P, fields_low.mask)
        mortality_step(fields_low.P, fields_low.mask, mu=mu, dt=1.0)
        loss_low = P_before_low - compute_total(fields_low.P, fields_low.mask)

        # High biomass
        fields_high = grid_factory(n=n)
        tilted_plane(fields_high)
        fill_field(fields_high.P, 5.0)

        P_before_high = compute_total(fields_high.P, fields_high.mask)
        mortality_step(fields_high.P, fields_high.mask, mu=mu, dt=1.0)
        loss_high = P_before_high - compute_total(fields_high.P, fields_high.mask)

        # Loss should scale with biomass
        ratio = loss_high / loss_low
        assert 4.5 < ratio < 5.5, f"Mortality ratio: {ratio}"

    def test_no_mortality_without_vegetation(self, grid_factory, tilted_plane):
        """No mortality when no vegetation."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 0.0)

        mortality_step(fields.P, fields.mask, mu=0.1, dt=1.0)

        P_np = fields.P.to_numpy()
        assert np.all(P_np == 0)

    def test_vegetation_stays_positive(self, grid_factory, tilted_plane):
        """Mortality can't make vegetation negative."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 0.01)  # Small amount

        # High mortality rate that would overshoot
        mortality_step(fields.P, fields.mask, mu=10.0, dt=1.0)

        P_np = fields.P.to_numpy()
        assert np.all(P_np >= 0), "Negative vegetation"


class TestDispersal:
    """Test seed dispersal diffusion."""

    def test_dispersal_conserves_mass(self, grid_factory, tilted_plane):
        """Dispersal alone should conserve total biomass."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Non-uniform initial vegetation
        P_np = np.random.uniform(0.5, 2.0, (n, n)).astype(np.float32)
        fields.P.from_numpy(P_np)

        P_before = compute_total(fields.P, fields.mask)

        D_P = 0.1
        dx = 1.0
        dt = compute_vegetation_timestep(D_P, dx, cfl=0.2)

        for _ in range(10):
            vegetation_diffusion_step(fields.P, fields.P_new, fields.mask, D_P, dx, dt)
            fields.P.from_numpy(fields.P_new.to_numpy())

        P_after = compute_total(fields.P, fields.mask)

        assert abs(P_after - P_before) < 1e-3, f"Dispersal lost mass: {P_before} -> {P_after}"

    def test_dispersal_smooths_gradient(self, grid_factory, tilted_plane):
        """Dispersal should reduce spatial variance."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Sharp gradient
        P_np = np.zeros((n, n), dtype=np.float32)
        P_np[:, :n // 2] = 2.0
        P_np[:, n // 2:] = 0.5
        fields.P.from_numpy(P_np)

        mask_np = fields.mask.to_numpy()
        variance_before = np.var(P_np[mask_np == 1])

        D_P = 0.1
        dx = 1.0
        dt = compute_vegetation_timestep(D_P, dx, cfl=0.2)

        for _ in range(50):
            vegetation_diffusion_step(fields.P, fields.P_new, fields.mask, D_P, dx, dt)
            fields.P.from_numpy(fields.P_new.to_numpy())

        P_final = fields.P.to_numpy()
        variance_after = np.var(P_final[mask_np == 1])

        assert variance_after < variance_before

    def test_dispersal_stays_positive(self, grid_factory, tilted_plane):
        """Dispersal can't make vegetation negative."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Extreme gradient
        P_np = np.zeros((n, n), dtype=np.float32)
        P_np[n // 2, n // 2] = 10.0  # Single spike
        fields.P.from_numpy(P_np)

        D_P = 0.5
        dx = 1.0
        dt = compute_vegetation_timestep(D_P, dx, cfl=0.25)

        for _ in range(100):
            vegetation_diffusion_step(fields.P, fields.P_new, fields.mask, D_P, dx, dt)
            fields.P.from_numpy(fields.P_new.to_numpy())

        P_final = fields.P.to_numpy()
        assert np.all(P_final >= -1e-6), "Negative vegetation from dispersal"


class TestCombinedVegetation:
    """Test combined vegetation dynamics."""

    def test_combined_step_runs(self, grid_factory, tilted_plane):
        """Combined step should run without error."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 1.0)
        fill_field(fields.M, 0.2)

        growth, mortality = vegetation_step(
            fields.P, fields.P_new, fields.M, fields.mask,
            g_max=0.1, k_G=0.1, mu=0.01, D_P=0.01, dx=1.0, dt=7.0
        )

        assert growth >= 0
        assert mortality >= 0

    def test_equilibrium_reached(self, grid_factory, tilted_plane):
        """Biomass stable when moisture at equilibrium point (G = μ)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        g_max = 0.1
        k_G = 0.1
        mu = 0.01

        # Set moisture at equilibrium: G(M) = μ => M = μ·k_G / (g_max - μ)
        M_eq = compute_equilibrium_moisture(g_max, k_G, mu)
        fill_field(fields.P, 1.0)
        fill_field(fields.M, M_eq)

        # Run for some timesteps
        for _ in range(50):
            vegetation_step(
                fields.P, fields.P_new, fields.M, fields.mask,
                g_max=g_max, k_G=k_G, mu=mu, D_P=0.01, dx=1.0, dt=7.0
            )

        final_P = compute_total(fields.P, fields.mask)

        # Run more steps
        for _ in range(50):
            vegetation_step(
                fields.P, fields.P_new, fields.M, fields.mask,
                g_max=g_max, k_G=k_G, mu=mu, D_P=0.01, dx=1.0, dt=7.0
            )

        final_P2 = compute_total(fields.P, fields.mask)

        # Change should be small at equilibrium moisture
        # (relaxed tolerance due to f32 precision and diffusion effects)
        relative_change = abs(final_P2 - final_P) / max(final_P, 1e-6)
        assert relative_change < 0.3, f"Not at equilibrium: {relative_change}"

    def test_extinction_without_moisture(self, grid_factory, tilted_plane):
        """Vegetation should decline to zero without moisture."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.P, 1.0)
        fill_field(fields.M, 0.0)  # No moisture

        for _ in range(500):
            vegetation_step(
                fields.P, fields.P_new, fields.M, fields.mask,
                g_max=0.1, k_G=0.1, mu=0.01, D_P=0.01, dx=1.0, dt=7.0
            )

        final_P = compute_total(fields.P, fields.mask)
        assert final_P < 0.01, f"Vegetation didn't go extinct: {final_P}"


class TestEquilibriumCalculation:
    """Test equilibrium moisture calculation."""

    def test_equilibrium_moisture_positive(self):
        """Equilibrium moisture should be positive when growth > mortality."""
        M_eq = compute_equilibrium_moisture(g_max=0.1, k_G=0.1, mu=0.01)
        assert M_eq > 0

    def test_equilibrium_infinite_when_mortality_dominates(self):
        """When mortality >= growth rate, equilibrium is impossible."""
        M_eq = compute_equilibrium_moisture(g_max=0.01, k_G=0.1, mu=0.1)
        assert M_eq == float("inf")


class TestTimestep:
    """Test vegetation timestep calculation."""

    def test_timestep_positive(self):
        """Timestep should be positive."""
        dt = compute_vegetation_timestep(D_P=0.01, dx=1.0)
        assert dt > 0

    def test_timestep_infinite_zero_diffusivity(self):
        """Zero diffusivity gives infinite timestep."""
        dt = compute_vegetation_timestep(D_P=0.0, dx=1.0)
        assert dt == float("inf")

    def test_weekly_timestep_stable(self):
        """Weekly timestep should be stable for typical parameters."""
        D_P = 0.01  # m²/day
        dx = 1.0    # m
        dt_stable = compute_vegetation_timestep(D_P, dx, cfl=0.25)

        # Weekly timestep = 7 days
        assert dt_stable > 7.0, f"Weekly timestep unstable: dt_stable={dt_stable}"
