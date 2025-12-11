"""
Tests for kernel dynamics validation beyond mass conservation.

These tests verify that the kernel implementations match the physical equations
from Saco et al. (2013) and ecohydro_spec.md, using DefaultParams values.

Key tests:
1. Quantitative rate verification against exact equations
2. Monod/Michaelis-Menten half-saturation properties
3. Equilibrium states and persistence thresholds
4. Analytical solution comparisons
5. Vegetation-water feedback mechanisms
"""

import math

import numpy as np
import pytest

from src.config import DefaultParams
from src.kernels.infiltration import infiltration_step
from src.kernels.soil import (
    compute_diffusion_timestep,
    diffusion_step,
    evapotranspiration_step,
    leakage_step,
)
from src.kernels.vegetation import (
    compute_equilibrium_moisture,
    compute_vegetation_timestep,
    growth_step,
    mortality_step,
    vegetation_diffusion_step,
)
from src.kernels.utils import compute_total, fill_field


class TestMonodKinetics:
    """
    Verify Monod/Michaelis-Menten kinetics: at half-saturation, rate = 50% of max.

    Monod form: rate = V_max * S / (S + K_m)
    At S = K_m: rate = V_max * K_m / (2 * K_m) = V_max / 2
    """

    def test_et_half_saturation(self, grid_factory, tilted_plane):
        """At M = k_M, ET rate should be 50% of maximum (for fixed P)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        E_max = DefaultParams.ET_MAX
        k_M = DefaultParams.K_ET
        beta_E = DefaultParams.BETA_ET
        dt = 1.0

        # Set moisture at half-saturation
        fill_field(fields.M, k_M)
        fill_field(fields.P, 0.0)  # No vegetation enhancement

        M_before = compute_total(fields.M, fields.mask)
        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_et = evapotranspiration_step(
            fields.M, fields.P, fields.mask, E_max, k_M, beta_E, dt
        )

        # Expected ET per cell at half-saturation (no veg): E_max * 0.5 * dt
        expected_et_per_cell = E_max * 0.5 * dt
        expected_total = expected_et_per_cell * n_active

        # Should be 50% of max rate
        assert abs(total_et - expected_total) / expected_total < 0.01, (
            f"At half-saturation, ET should be 50% of max. "
            f"Got {total_et:.6f}, expected {expected_total:.6f}"
        )

    def test_growth_half_saturation(self, grid_factory, tilted_plane):
        """At M = k_G, growth rate should be 50% of maximum."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        g_max = DefaultParams.G_MAX
        k_G = DefaultParams.K_G
        dt = 1.0
        P_initial = 1.0

        # Set moisture at half-saturation
        fill_field(fields.M, k_G)
        fill_field(fields.P, P_initial)

        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_growth = growth_step(fields.P, fields.M, fields.mask, g_max, k_G, dt)

        # Expected growth per cell: g_max * 0.5 * P * dt
        expected_growth_per_cell = g_max * 0.5 * P_initial * dt
        expected_total = expected_growth_per_cell * n_active

        # Should be 50% of max rate
        assert abs(total_growth - expected_total) / expected_total < 0.01, (
            f"At half-saturation, growth should be 50% of max. "
            f"Got {total_growth:.6f}, expected {expected_total:.6f}"
        )

    def test_growth_saturation_approaches_max(self, grid_factory, tilted_plane):
        """At high M >> k_G, growth rate approaches g_max."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        g_max = DefaultParams.G_MAX
        k_G = DefaultParams.K_G
        dt = 1.0
        P_initial = 1.0

        # Set moisture at 10x half-saturation (should give ~91% of max)
        fill_field(fields.M, 10 * k_G)
        fill_field(fields.P, P_initial)

        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_growth = growth_step(fields.P, fields.M, fields.mask, g_max, k_G, dt)

        # At M = 10*k_G: rate = g_max * 10/(10+1) = g_max * 0.909
        expected_fraction = 10.0 / 11.0
        expected_growth_per_cell = g_max * expected_fraction * P_initial * dt
        expected_total = expected_growth_per_cell * n_active

        assert abs(total_growth - expected_total) / expected_total < 0.01, (
            f"At M=10*k_G, growth should be {expected_fraction:.2%} of max. "
            f"Got {total_growth:.6f}, expected {expected_total:.6f}"
        )


class TestInfiltrationDynamics:
    """
    Verify infiltration matches: I = alpha * h * veg_factor * sat_factor * dt

    Where:
    - veg_factor = (P + k_P * W_0) / (P + k_P)
    - sat_factor = max(0, 1 - M / M_sat)
    """

    def test_infiltration_vegetation_factor_bare_soil(self, grid_factory, tilted_plane):
        """With P=0 (bare soil), veg_factor should equal W_0."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        alpha = DefaultParams.K_SAT / DefaultParams.ALPHA_I  # Effective rate
        k_P = DefaultParams.K_P
        W_0 = DefaultParams.W_0
        M_sat = DefaultParams.M_SAT
        dt = 0.1
        h_initial = 0.05

        # Bare soil: P = 0
        fill_field(fields.h, h_initial)
        fill_field(fields.M, 0.0)  # Empty soil (sat_factor = 1)
        fill_field(fields.P, 0.0)

        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_inf = infiltration_step(
            fields.h, fields.M, fields.P, fields.mask, alpha, k_P, W_0, M_sat, dt
        )

        # veg_factor = (0 + k_P * W_0) / (0 + k_P) = W_0
        # Expected infiltration per cell (limited by h_initial)
        expected_inf_per_cell = min(alpha * h_initial * W_0 * 1.0 * dt, h_initial)
        expected_total = expected_inf_per_cell * n_active

        assert abs(total_inf - expected_total) / expected_total < 0.02, (
            f"Bare soil infiltration factor should be W_0={W_0}. "
            f"Got {total_inf:.6f}, expected {expected_total:.6f}"
        )

    def test_infiltration_vegetation_factor_dense(self, grid_factory, tilted_plane):
        """With high P >> k_P, veg_factor should approach 1."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        alpha = 0.1
        k_P = DefaultParams.K_P
        W_0 = DefaultParams.W_0
        M_sat = DefaultParams.M_SAT
        dt = 0.1
        h_initial = 0.05
        P_dense = 100.0  # Very high vegetation

        fill_field(fields.h, h_initial)
        fill_field(fields.M, 0.0)  # Empty soil
        fill_field(fields.P, P_dense)

        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_inf = infiltration_step(
            fields.h, fields.M, fields.P, fields.mask, alpha, k_P, W_0, M_sat, dt
        )

        # veg_factor = (100 + 1*0.2) / (100 + 1) ≈ 0.992 → approaches 1
        veg_factor = (P_dense + k_P * W_0) / (P_dense + k_P)
        expected_inf_per_cell = min(alpha * h_initial * veg_factor * 1.0 * dt, h_initial)
        expected_total = expected_inf_per_cell * n_active

        assert abs(total_inf - expected_total) / expected_total < 0.02, (
            f"Dense vegetation infiltration factor should approach 1. "
            f"Got veg_factor={veg_factor:.4f}"
        )

    def test_infiltration_saturation_factor(self, grid_factory, tilted_plane):
        """Saturation factor should decrease linearly with M/M_sat."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        alpha = 0.1
        k_P = DefaultParams.K_P
        W_0 = DefaultParams.W_0
        M_sat = DefaultParams.M_SAT
        dt = 0.1
        h_initial = 0.1
        P_value = 1.0

        # At 50% saturation
        M_half = 0.5 * M_sat
        fill_field(fields.h, h_initial)
        fill_field(fields.M, M_half)
        fill_field(fields.P, P_value)

        n_active = np.sum(fields.mask.to_numpy() == 1)

        total_inf = infiltration_step(
            fields.h, fields.M, fields.P, fields.mask, alpha, k_P, W_0, M_sat, dt
        )

        veg_factor = (P_value + k_P * W_0) / (P_value + k_P)
        sat_factor = 0.5  # 1 - M_half/M_sat = 0.5
        expected_inf_per_cell = alpha * h_initial * veg_factor * sat_factor * dt
        expected_total = expected_inf_per_cell * n_active

        assert abs(total_inf - expected_total) / expected_total < 0.02, (
            f"At 50% saturation, sat_factor should be 0.5. "
            f"Got {total_inf:.6f}, expected {expected_total:.6f}"
        )


class TestLeakageDynamics:
    """
    Verify leakage matches: L = L_max * (M/M_sat)^2
    """

    def test_leakage_quadratic_relationship(self, grid_factory, tilted_plane):
        """Leakage should follow quadratic (M/M_sat)^2 relationship."""
        n = 16
        M_sat = DefaultParams.M_SAT
        L_max = DefaultParams.LEAKAGE
        dt = 1.0

        # Test at 25%, 50%, 75% saturation
        saturations = [0.25, 0.5, 0.75]
        leakages = []

        for sat_frac in saturations:
            fields = grid_factory(n=n)
            tilted_plane(fields)
            fill_field(fields.M, sat_frac * M_sat)

            M_before = compute_total(fields.M, fields.mask)
            leakage_step(fields.M, fields.mask, L_max, M_sat, dt)
            M_after = compute_total(fields.M, fields.mask)

            leakages.append(M_before - M_after)

        # Leakage ratios should follow (sat_frac)^2 ratios
        # L(0.5)/L(0.25) should be (0.5/0.25)^2 = 4
        ratio_50_25 = leakages[1] / leakages[0]
        expected_ratio_50_25 = (0.5 / 0.25) ** 2

        # L(0.75)/L(0.25) should be (0.75/0.25)^2 = 9
        ratio_75_25 = leakages[2] / leakages[0]
        expected_ratio_75_25 = (0.75 / 0.25) ** 2

        assert abs(ratio_50_25 - expected_ratio_50_25) < 0.2, (
            f"Leakage ratio L(50%)/L(25%) should be 4. Got {ratio_50_25:.2f}"
        )
        assert abs(ratio_75_25 - expected_ratio_75_25) < 0.5, (
            f"Leakage ratio L(75%)/L(25%) should be 9. Got {ratio_75_25:.2f}"
        )


class TestDefaultParamsEquilibrium:
    """
    Test equilibrium states and thresholds with DefaultParams values.
    """

    def test_equilibrium_moisture_calculation(self):
        """Verify equilibrium moisture formula: M_eq = mu * k_G / (g_max - mu)."""
        g_max = DefaultParams.G_MAX  # 0.02
        k_G = DefaultParams.K_G  # 0.1
        mu = DefaultParams.MORTALITY  # 0.001

        # At equilibrium: g_max * M / (M + k_G) = mu
        # Solving: M = mu * k_G / (g_max - mu)
        M_eq_expected = mu * k_G / (g_max - mu)
        M_eq_computed = compute_equilibrium_moisture(g_max, k_G, mu)

        assert abs(M_eq_computed - M_eq_expected) < 1e-10, (
            f"Equilibrium moisture should be {M_eq_expected:.6f}, "
            f"got {M_eq_computed:.6f}"
        )

    def test_vegetation_persistence_requires_gmax_gt_mu(self):
        """Vegetation can only persist if g_max > mu."""
        g_max = DefaultParams.G_MAX
        mu = DefaultParams.MORTALITY

        # With DefaultParams, g_max >> mu, so vegetation can persist
        assert g_max > mu, (
            f"DefaultParams should have g_max > mu for vegetation persistence. "
            f"g_max={g_max}, mu={mu}"
        )

        # Equilibrium moisture should be finite and positive
        M_eq = compute_equilibrium_moisture(g_max, DefaultParams.K_G, mu)
        assert 0 < M_eq < float("inf"), (
            f"Equilibrium moisture should be finite and positive. Got {M_eq}"
        )

    def test_vegetation_growth_above_equilibrium_moisture(self, grid_factory, tilted_plane):
        """Above equilibrium moisture, vegetation should grow (G > mu)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        g_max = DefaultParams.G_MAX
        k_G = DefaultParams.K_G
        mu = DefaultParams.MORTALITY
        dt = 7.0  # Weekly timestep

        M_eq = compute_equilibrium_moisture(g_max, k_G, mu)

        # Set moisture above equilibrium
        fill_field(fields.M, M_eq * 2.0)
        fill_field(fields.P, 1.0)

        P_before = compute_total(fields.P, fields.mask)

        # Apply growth and mortality
        growth_step(fields.P, fields.M, fields.mask, g_max, k_G, dt)
        mortality_step(fields.P, fields.mask, mu, dt)

        P_after = compute_total(fields.P, fields.mask)

        assert P_after > P_before, (
            f"Above equilibrium moisture ({M_eq*2:.4f} > {M_eq:.4f}), "
            f"vegetation should grow. P: {P_before:.4f} -> {P_after:.4f}"
        )

    def test_vegetation_decline_below_equilibrium_moisture(self, grid_factory, tilted_plane):
        """Below equilibrium moisture, vegetation should decline (G < mu)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        g_max = DefaultParams.G_MAX
        k_G = DefaultParams.K_G
        mu = DefaultParams.MORTALITY
        dt = 7.0

        M_eq = compute_equilibrium_moisture(g_max, k_G, mu)

        # Set moisture below equilibrium
        fill_field(fields.M, M_eq * 0.5)
        fill_field(fields.P, 1.0)

        P_before = compute_total(fields.P, fields.mask)

        growth_step(fields.P, fields.M, fields.mask, g_max, k_G, dt)
        mortality_step(fields.P, fields.mask, mu, dt)

        P_after = compute_total(fields.P, fields.mask)

        assert P_after < P_before, (
            f"Below equilibrium moisture ({M_eq*0.5:.4f} < {M_eq:.4f}), "
            f"vegetation should decline. P: {P_before:.4f} -> {P_after:.4f}"
        )


class TestAnalyticalSolutions:
    """
    Compare kernel outputs to analytical solutions for simple cases.
    """

    def test_mortality_exponential_decay(self, grid_factory, tilted_plane):
        """Without growth, vegetation should decay exponentially: P(t) = P0 * exp(-mu*t)."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        mu = DefaultParams.MORTALITY
        P0 = 1.0
        dt = 1.0
        n_steps = 100

        fill_field(fields.P, P0)
        fill_field(fields.M, 0.0)  # No moisture = no growth

        n_active = np.sum(fields.mask.to_numpy() == 1)

        P_history = [P0]
        for _ in range(n_steps):
            mortality_step(fields.P, fields.mask, mu, dt)
            P_total = compute_total(fields.P, fields.mask) / n_active
            P_history.append(P_total)

        # Compare to analytical solution P(t) = P0 * exp(-mu * t)
        t_final = n_steps * dt
        P_analytical = P0 * math.exp(-mu * t_final)
        P_numerical = P_history[-1]

        # Allow some tolerance for discrete vs continuous
        relative_error = abs(P_numerical - P_analytical) / P_analytical
        assert relative_error < 0.05, (
            f"Exponential decay should match analytical solution. "
            f"Numerical: {P_numerical:.6f}, Analytical: {P_analytical:.6f}, "
            f"Error: {relative_error:.2%}"
        )

    def test_diffusion_gaussian_spreading(self, grid_factory, tilted_plane):
        """
        Point source diffusion should spread as Gaussian.

        For 2D diffusion: sigma(t) = sqrt(2 * D * t)
        Variance should increase linearly with time.
        """
        n = 64
        fields = grid_factory(n=n)
        tilted_plane(fields)

        D_M = DefaultParams.D_SOIL
        dx = DefaultParams.DX
        dt = compute_diffusion_timestep(D_M, dx, cfl=0.2)

        # Initialize with point source at center
        M_np = np.zeros((n, n), dtype=np.float32)
        center = n // 2
        M_np[center, center] = 1.0
        fields.M.from_numpy(M_np)

        mask_np = fields.mask.to_numpy()

        # Measure initial variance (should be near zero)
        def compute_variance(M_field):
            """Compute spatial variance of mass distribution."""
            M_arr = M_field.to_numpy() * (mask_np == 1)
            total = M_arr.sum()
            if total < 1e-10:
                return 0.0
            # Compute center of mass
            ii, jj = np.meshgrid(range(n), range(n), indexing='ij')
            cx = np.sum(ii * M_arr) / total
            cy = np.sum(jj * M_arr) / total
            # Compute variance
            var = np.sum(((ii - cx)**2 + (jj - cy)**2) * M_arr) / total
            return var

        var_initial = compute_variance(fields.M)

        # Run diffusion
        n_steps = 50
        for _ in range(n_steps):
            diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)
            fields.M.from_numpy(fields.M_new.to_numpy())

        var_final = compute_variance(fields.M)

        # Variance should have increased (diffusion spreads the mass)
        assert var_final > var_initial, (
            f"Diffusion should spread mass. Variance: {var_initial:.2f} -> {var_final:.2f}"
        )

        # Analytical: var(t) = var(0) + 2*D*t for each dimension, so 4*D*t total
        t_total = n_steps * dt
        expected_var_increase = 4 * D_M * t_total / (dx * dx)  # In grid units

        # The variance increase should be approximately linear with 4*D*t
        actual_var_increase = var_final - var_initial
        # Allow factor of 2 tolerance due to discrete grid and boundary effects
        assert 0.5 * expected_var_increase < actual_var_increase < 2.0 * expected_var_increase, (
            f"Variance increase should match diffusion theory. "
            f"Expected ~{expected_var_increase:.2f}, got {actual_var_increase:.2f}"
        )


class TestVegetationFeedback:
    """
    Test the vegetation-water feedback mechanisms that drive pattern formation.
    """

    def test_infiltration_enhancement_range(self):
        """
        Infiltration enhancement factor should range from W_0 (bare) to 1 (dense veg).

        veg_factor = (P + k_P * W_0) / (P + k_P)
        At P=0: veg_factor = W_0
        As P->inf: veg_factor -> 1
        """
        k_P = DefaultParams.K_P
        W_0 = DefaultParams.W_0

        # P = 0 (bare soil)
        veg_factor_bare = (0 + k_P * W_0) / (0 + k_P)
        assert abs(veg_factor_bare - W_0) < 1e-10, (
            f"Bare soil factor should be W_0={W_0}. Got {veg_factor_bare}"
        )

        # P = k_P (half-saturation)
        veg_factor_half = (k_P + k_P * W_0) / (k_P + k_P)
        expected_half = (1 + W_0) / 2
        assert abs(veg_factor_half - expected_half) < 1e-10, (
            f"At P=k_P, factor should be {expected_half}. Got {veg_factor_half}"
        )

        # P >> k_P (dense vegetation)
        P_dense = 1000 * k_P
        veg_factor_dense = (P_dense + k_P * W_0) / (P_dense + k_P)
        assert veg_factor_dense > 0.99, (
            f"Dense vegetation factor should approach 1. Got {veg_factor_dense}"
        )

    def test_positive_feedback_more_veg_more_infiltration(self, grid_factory, tilted_plane):
        """
        More vegetation should lead to more infiltration, which leads to more soil moisture.
        This is the positive local feedback in the Turing mechanism.
        """
        n = 16
        alpha = 0.1
        k_P = DefaultParams.K_P
        W_0 = DefaultParams.W_0
        M_sat = DefaultParams.M_SAT
        dt = 0.1
        h_initial = 0.1

        infiltration_by_veg = []

        for P_value in [0.0, 1.0, 5.0, 10.0]:
            fields = grid_factory(n=n)
            tilted_plane(fields)

            fill_field(fields.h, h_initial)
            fill_field(fields.M, 0.0)  # Start dry
            fill_field(fields.P, P_value)

            total_inf = infiltration_step(
                fields.h, fields.M, fields.P, fields.mask, alpha, k_P, W_0, M_sat, dt
            )
            infiltration_by_veg.append(total_inf)

        # Infiltration should increase monotonically with vegetation
        for i in range(len(infiltration_by_veg) - 1):
            assert infiltration_by_veg[i] < infiltration_by_veg[i + 1], (
                f"Infiltration should increase with vegetation. "
                f"P={[0, 1, 5, 10][i]} gave {infiltration_by_veg[i]:.6f}, "
                f"P={[0, 1, 5, 10][i+1]} gave {infiltration_by_veg[i+1]:.6f}"
            )

    def test_et_enhancement_by_vegetation(self, grid_factory, tilted_plane):
        """
        Vegetation should enhance ET (transpiration in addition to evaporation).
        This is part of the negative feedback (vegetation depletes water faster).
        """
        n = 16
        E_max = DefaultParams.ET_MAX
        k_M = DefaultParams.K_ET
        beta_E = DefaultParams.BETA_ET
        dt = 1.0
        M_value = 0.2

        et_by_veg = []

        for P_value in [0.0, 1.0, 5.0]:
            fields = grid_factory(n=n)
            tilted_plane(fields)

            fill_field(fields.M, M_value)
            fill_field(fields.P, P_value)

            total_et = evapotranspiration_step(
                fields.M, fields.P, fields.mask, E_max, k_M, beta_E, dt
            )
            et_by_veg.append(total_et)

        # ET should increase with vegetation
        for i in range(len(et_by_veg) - 1):
            assert et_by_veg[i] < et_by_veg[i + 1], (
                f"ET should increase with vegetation. "
                f"P={[0, 1, 5][i]} gave {et_by_veg[i]:.6f}, "
                f"P={[0, 1, 5][i+1]} gave {et_by_veg[i+1]:.6f}"
            )


class TestTimescaleSeparation:
    """
    Verify that DefaultParams give physically realistic timescales.
    """

    def test_diffusion_timestep_reasonable(self):
        """Soil and vegetation diffusion should allow reasonable timesteps."""
        dx = DefaultParams.DX

        # Soil moisture diffusion
        D_soil = DefaultParams.D_SOIL
        dt_soil = compute_diffusion_timestep(D_soil, dx)
        assert dt_soil > 0.1, (
            f"Soil diffusion timestep should allow > 0.1 day steps. Got {dt_soil:.4f}"
        )

        # Vegetation diffusion (slower)
        D_veg = DefaultParams.D_VEG
        dt_veg = compute_vegetation_timestep(D_veg, dx)
        assert dt_veg > 1.0, (
            f"Vegetation diffusion timestep should allow > 1 day steps. Got {dt_veg:.4f}"
        )
        assert dt_veg > dt_soil, (
            f"Vegetation should diffuse slower than soil moisture. "
            f"dt_veg={dt_veg:.2f}, dt_soil={dt_soil:.2f}"
        )

    def test_et_rate_physically_reasonable(self):
        """Max ET should be in reasonable range (few mm/day)."""
        ET_max = DefaultParams.ET_MAX

        # Convert to mm/day
        ET_max_mm = ET_max * 1000

        # Typical semiarid ET: 1-10 mm/day
        assert 1.0 <= ET_max_mm <= 10.0, (
            f"Max ET should be 1-10 mm/day. Got {ET_max_mm:.1f} mm/day"
        )

    def test_growth_timescale_slower_than_surface_dynamics(self):
        """
        Vegetation growth should be slower than surface water dynamics.

        The key timescale separation is:
        - Surface water: hours (storm events)
        - Soil moisture: days to weeks
        - Vegetation: weeks to months

        Vegetation responds to seasonal moisture patterns, not individual storms.
        """
        g_max = DefaultParams.G_MAX
        mu = DefaultParams.MORTALITY

        # Vegetation doubling time (at max growth): t = ln(2) / g_max
        doubling_time = math.log(2) / g_max

        # Vegetation decay time (e-folding): t = 1 / mu
        decay_time = 1.0 / mu

        # Both timescales should be > 1 week (vegetation is slow)
        assert doubling_time > 7.0, (
            f"Vegetation doubling time should be > 1 week. Got {doubling_time:.1f} days"
        )
        assert decay_time > 30.0, (
            f"Vegetation decay time should be > 1 month. Got {decay_time:.1f} days"
        )

        # Compare to storm timescale
        storm_duration = DefaultParams.STORM_DURATION  # days
        assert doubling_time > 100 * storm_duration, (
            f"Vegetation should grow much slower than storm duration. "
            f"Doubling: {doubling_time:.1f} days, Storm: {storm_duration:.2f} days"
        )


class TestPhysicalConstraints:
    """
    Test that physical constraints are maintained.
    """

    def test_all_default_params_positive(self):
        """All DefaultParams should be positive."""
        params = [
            ("DX", DefaultParams.DX),
            ("R_MEAN", DefaultParams.R_MEAN),
            ("STORM_DURATION", DefaultParams.STORM_DURATION),
            ("INTERSTORM", DefaultParams.INTERSTORM),
            ("K_SAT", DefaultParams.K_SAT),
            ("ALPHA_I", DefaultParams.ALPHA_I),
            ("K_P", DefaultParams.K_P),
            ("W_0", DefaultParams.W_0),
            ("M_SAT", DefaultParams.M_SAT),
            ("ET_MAX", DefaultParams.ET_MAX),
            ("K_ET", DefaultParams.K_ET),
            ("BETA_ET", DefaultParams.BETA_ET),
            ("LEAKAGE", DefaultParams.LEAKAGE),
            ("D_SOIL", DefaultParams.D_SOIL),
            ("G_MAX", DefaultParams.G_MAX),
            ("K_G", DefaultParams.K_G),
            ("MORTALITY", DefaultParams.MORTALITY),
            ("D_VEG", DefaultParams.D_VEG),
            ("MANNING_N", DefaultParams.MANNING_N),
            ("MIN_SLOPE", DefaultParams.MIN_SLOPE),
            ("H_THRESHOLD", DefaultParams.H_THRESHOLD),
            ("DRAINAGE_TIME", DefaultParams.DRAINAGE_TIME),
        ]

        for name, value in params:
            assert value > 0, f"DefaultParams.{name} should be positive. Got {value}"

    def test_bare_soil_infiltration_less_than_full(self):
        """W_0 should be less than 1 (bare soil infiltrates less than vegetated)."""
        W_0 = DefaultParams.W_0
        assert 0 < W_0 < 1, (
            f"W_0 should be between 0 and 1. Got {W_0}"
        )

    def test_vegetation_can_persist_with_available_moisture(self):
        """With M_sat worth of moisture, vegetation should be able to persist."""
        g_max = DefaultParams.G_MAX
        k_G = DefaultParams.K_G
        mu = DefaultParams.MORTALITY
        M_sat = DefaultParams.M_SAT

        M_eq = compute_equilibrium_moisture(g_max, k_G, mu)

        assert M_eq < M_sat, (
            f"Equilibrium moisture ({M_eq:.4f}) should be achievable "
            f"within saturation ({M_sat:.4f})"
        )
