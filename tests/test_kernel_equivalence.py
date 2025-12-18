"""
Tests for kernel equivalence: fused vs naive implementations.

These tests verify that optimized (fused) kernels produce identical results
to the naive (separate) implementations within floating-point tolerance.

Note: Diffusion now uses a single generic laplacian_diffusion_step from
geometry.py for both soil and vegetation. The duplicate kernels have been
removed as part of the Phase 4 consolidation.
"""

import numpy as np

from src.fields import fill_field
from src.kernels.soil import (
    et_leakage_step_fused,
    evapotranspiration_step,
    leakage_step,
    soil_moisture_step,
    soil_moisture_step_naive,
)
from src.kernels.vegetation import (
    growth_mortality_step_fused,
    growth_step,
    mortality_step,
    vegetation_step,
    vegetation_step_naive,
)


class TestSoilKernelEquivalence:
    """Test fused soil kernels match naive implementations."""

    def test_et_leakage_fused_matches_separate(self, grid_factory, tilted_plane):
        """Fused ET+leakage should match sequential separate kernels."""
        n = 32

        # Parameters
        E_max, k_M, beta_E = 0.01, 0.05, 0.5
        L_max, M_sat = 0.005, 0.4
        dt = 1.0

        # Setup for naive run
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        # Setup for fused run (identical initial state)
        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run naive (separate kernels)
        et_naive = evapotranspiration_step(
            fields_naive.M, fields_naive.P, fields_naive.mask,
            E_max, k_M, beta_E, dt
        )
        leak_naive = leakage_step(
            fields_naive.M, fields_naive.mask,
            L_max, M_sat, dt
        )

        # Run fused
        totals = et_leakage_step_fused(
            fields_fused.M, fields_fused.P, fields_fused.mask,
            E_max, k_M, beta_E, L_max, M_sat, dt
        )
        et_fused, leak_fused = totals[0], totals[1]

        # Compare results
        M_naive = fields_naive.M.to_numpy()
        M_fused = fields_fused.M.to_numpy()

        assert np.allclose(M_naive, M_fused, rtol=1e-5, atol=1e-8), \
            f"M fields differ: max diff = {np.max(np.abs(M_naive - M_fused))}"

        assert abs(et_naive - et_fused) < 1e-5 * max(et_naive, 1e-10), \
            f"ET totals differ: naive={et_naive}, fused={et_fused}"

        assert abs(leak_naive - leak_fused) < 1e-5 * max(leak_naive, 1e-10), \
            f"Leakage totals differ: naive={leak_naive}, fused={leak_fused}"

    def test_soil_moisture_step_fused_matches_naive(self, grid_factory, tilted_plane):
        """Full soil_moisture_step should match naive version."""
        n = 32

        # Parameters
        E_max, k_M, beta_E = 0.01, 0.05, 0.5
        L_max, M_sat = 0.005, 0.4
        D_M, dx, dt = 0.001, 1.0, 1.0

        # Setup for naive run
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        # Setup for fused run
        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run naive
        et_naive, leak_naive = soil_moisture_step_naive(
            fields_naive, E_max, k_M, beta_E, L_max, M_sat, D_M, dx, dt
        )

        # Run fused (default)
        et_fused, leak_fused = soil_moisture_step(
            fields_fused, E_max, k_M, beta_E, L_max, M_sat, D_M, dx, dt
        )

        # Compare M fields (note: both have been swapped, so compare M directly)
        M_naive = fields_naive.M.to_numpy()
        M_fused = fields_fused.M.to_numpy()

        assert np.allclose(M_naive, M_fused, rtol=1e-5, atol=1e-8), \
            f"M fields differ after full step: max diff = {np.max(np.abs(M_naive - M_fused))}"

        assert abs(et_naive - et_fused) < 1e-5 * max(et_naive, 1e-10)
        assert abs(leak_naive - leak_fused) < 1e-5 * max(leak_naive, 1e-10)

class TestVegetationKernelEquivalence:
    """Test fused vegetation kernels match naive implementations."""

    def test_growth_mortality_fused_matches_separate(self, grid_factory, tilted_plane):
        """Fused growth+mortality should match sequential separate kernels."""
        n = 32

        # Parameters
        g_max, k_G, mu = 0.1, 0.1, 0.02
        dt = 7.0  # Weekly timestep

        # Setup for naive run
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        # Setup for fused run
        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run naive (separate kernels)
        growth_naive = growth_step(
            fields_naive.P, fields_naive.M, fields_naive.mask,
            g_max, k_G, dt
        )
        mort_naive = mortality_step(
            fields_naive.P, fields_naive.mask,
            mu, dt
        )

        # Run fused
        totals = growth_mortality_step_fused(
            fields_fused.P, fields_fused.M, fields_fused.mask,
            g_max, k_G, mu, dt
        )
        growth_fused, mort_fused = totals[0], totals[1]

        # Compare results
        P_naive = fields_naive.P.to_numpy()
        P_fused = fields_fused.P.to_numpy()

        assert np.allclose(P_naive, P_fused, rtol=1e-5, atol=1e-8), \
            f"P fields differ: max diff = {np.max(np.abs(P_naive - P_fused))}"

        assert abs(growth_naive - growth_fused) < 1e-5 * max(growth_naive, 1e-10), \
            f"Growth totals differ: naive={growth_naive}, fused={growth_fused}"

        assert abs(mort_naive - mort_fused) < 1e-5 * max(mort_naive, 1e-10), \
            f"Mortality totals differ: naive={mort_naive}, fused={mort_fused}"

    def test_vegetation_step_fused_matches_naive(self, grid_factory, tilted_plane):
        """Full vegetation_step should match naive version."""
        n = 32

        # Parameters
        g_max, k_G, mu = 0.1, 0.1, 0.02
        D_P, dx, dt = 0.001, 1.0, 7.0

        # Setup for naive run
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        # Setup for fused run
        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run naive
        growth_naive, mort_naive = vegetation_step_naive(
            fields_naive, g_max, k_G, mu, D_P, dx, dt
        )

        # Run fused (default)
        growth_fused, mort_fused = vegetation_step(
            fields_fused, g_max, k_G, mu, D_P, dx, dt
        )

        # Compare P fields
        P_naive = fields_naive.P.to_numpy()
        P_fused = fields_fused.P.to_numpy()

        assert np.allclose(P_naive, P_fused, rtol=1e-5, atol=1e-8), \
            f"P fields differ after full step: max diff = {np.max(np.abs(P_naive - P_fused))}"

        assert abs(growth_naive - growth_fused) < 1e-5 * max(growth_naive, 1e-10)
        assert abs(mort_naive - mort_fused) < 1e-5 * max(mort_naive, 1e-10)


class TestMultiStepEquivalence:
    """Test equivalence over multiple timesteps."""

    def test_soil_equivalence_multiple_steps(self, grid_factory, tilted_plane):
        """Fused and naive should remain equivalent over multiple steps."""
        n = 32
        n_steps = 10

        # Parameters
        E_max, k_M, beta_E = 0.01, 0.05, 0.5
        L_max, M_sat = 0.005, 0.4
        D_M, dx, dt = 0.001, 1.0, 1.0

        # Setup both
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run multiple steps
        for _ in range(n_steps):
            soil_moisture_step_naive(
                fields_naive, E_max, k_M, beta_E, L_max, M_sat, D_M, dx, dt
            )
            soil_moisture_step(
                fields_fused, E_max, k_M, beta_E, L_max, M_sat, D_M, dx, dt
            )

        # Compare final states
        M_naive = fields_naive.M.to_numpy()
        M_fused = fields_fused.M.to_numpy()

        assert np.allclose(M_naive, M_fused, rtol=1e-4, atol=1e-7), \
            f"M diverged after {n_steps} steps: max diff = {np.max(np.abs(M_naive - M_fused))}"

    def test_vegetation_equivalence_multiple_steps(self, grid_factory, tilted_plane):
        """Fused and naive vegetation should remain equivalent over multiple steps."""
        n = 32
        n_steps = 10

        # Parameters
        g_max, k_G, mu = 0.1, 0.1, 0.02
        D_P, dx, dt = 0.001, 1.0, 7.0

        # Setup both
        fields_naive = grid_factory(n=n)
        tilted_plane(fields_naive)
        fill_field(fields_naive.M, 0.2)
        fill_field(fields_naive.P, 0.5)

        fields_fused = grid_factory(n=n)
        tilted_plane(fields_fused)
        fill_field(fields_fused.M, 0.2)
        fill_field(fields_fused.P, 0.5)

        # Run multiple steps
        for _ in range(n_steps):
            vegetation_step_naive(
                fields_naive, g_max, k_G, mu, D_P, dx, dt
            )
            vegetation_step(
                fields_fused, g_max, k_G, mu, D_P, dx, dt
            )

        # Compare final states
        P_naive = fields_naive.P.to_numpy()
        P_fused = fields_fused.P.to_numpy()

        assert np.allclose(P_naive, P_fused, rtol=1e-4, atol=1e-7), \
            f"P diverged after {n_steps} steps: max diff = {np.max(np.abs(P_naive - P_fused))}"
