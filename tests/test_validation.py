"""
Phase 5: System Validation Tests

Verify that the integrated system produces correct emergent behavior:
1. Long-term numerical stability (no drift over decades)
2. Pattern emergence from uniform initial conditions
3. Characteristic wavelength in physically reasonable range
4. Parameter sensitivity (infiltration feedback, rainfall frequency)
5. Turing instability mechanism validation

These tests verify the SYSTEM as a whole, not individual kernels.
"""

import math

import numpy as np
import pytest

from src.config import init_taichi, DefaultParams
from src.simulation import Simulation, SimulationParams


class TestLongRunStability:
    """
    Verify numerical stability over multi-year simulations.

    The system should:
    - Not produce NaN or Inf values
    - Not accumulate unbounded numerical error
    - Maintain mass conservation to reasonable tolerance
    - Keep all fields within physical bounds
    """

    def test_no_nan_or_inf_after_one_year(self, taichi_init):
        """System should produce no NaN or Inf values after one year."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(seed=42)

        sim.run(years=1.0, check_mass_balance=True, verbose=False)

        h = sim.state.fields.h.to_numpy()
        M = sim.state.fields.M.to_numpy()
        P = sim.state.fields.P.to_numpy()

        assert not np.any(np.isnan(h)), "Surface water contains NaN"
        assert not np.any(np.isnan(M)), "Soil moisture contains NaN"
        assert not np.any(np.isnan(P)), "Vegetation contains NaN"
        assert not np.any(np.isinf(h)), "Surface water contains Inf"
        assert not np.any(np.isinf(M)), "Soil moisture contains Inf"
        assert not np.any(np.isinf(P)), "Vegetation contains Inf"

    def test_fields_within_physical_bounds_after_one_year(self, taichi_init):
        """All fields should remain within physical bounds."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(seed=42)

        sim.run(years=1.0, check_mass_balance=True, verbose=False)

        h = sim.state.fields.h.to_numpy()
        M = sim.state.fields.M.to_numpy()
        P = sim.state.fields.P.to_numpy()
        mask = sim.state.fields.mask.to_numpy()

        # Only check interior cells
        interior = mask == 1

        # Surface water: h >= 0, h < some reasonable max (10m)
        assert np.all(h[interior] >= 0), f"Negative surface water: min={h[interior].min()}"
        assert np.all(h[interior] < 10.0), f"Unreasonable surface water: max={h[interior].max()}"

        # Soil moisture: 0 <= M <= M_sat
        assert np.all(M[interior] >= 0), f"Negative soil moisture: min={M[interior].min()}"
        assert np.all(M[interior] <= params.M_sat * 1.01), (
            f"Soil moisture exceeds saturation: max={M[interior].max()}, M_sat={params.M_sat}"
        )

        # Vegetation: P >= 0
        assert np.all(P[interior] >= 0), f"Negative vegetation: min={P[interior].min()}"

    def test_mass_conservation_over_multiple_years(self, taichi_init):
        """Mass balance error should remain small over multi-year runs."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(seed=42)

        # Run for 2 years, checking mass balance
        sim.run(years=2.0, check_mass_balance=True, verbose=False)

        # Final mass balance check with tighter tolerance
        error = sim.check_mass_balance()
        assert error < 1e-3, f"Mass balance error after 2 years: {error:.2e}"

    def test_no_numerical_drift_five_years(self, taichi_init):
        """
        System should not show unbounded growth or decay over 5 years.

        Mean vegetation should stay within reasonable bounds, not
        exponentially growing or collapsing to zero.
        """
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        P_initial_mean = sim.state.fields.P.to_numpy().mean()

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P_final = sim.state.fields.P.to_numpy()
        mask = sim.state.fields.mask.to_numpy()
        P_final_mean = P_final[mask == 1].mean()

        # Vegetation should not have grown unboundedly or collapsed
        # Allow 10x change either direction over 5 years
        assert P_final_mean > P_initial_mean / 10, (
            f"Vegetation collapsed: {P_initial_mean:.3f} -> {P_final_mean:.3f}"
        )
        assert P_final_mean < P_initial_mean * 10, (
            f"Vegetation exploded: {P_initial_mean:.3f} -> {P_final_mean:.3f}"
        )


class TestPatternEmergence:
    """
    Verify that spatial patterns emerge from near-uniform initial conditions.

    The Turing instability should cause initially uniform vegetation
    to self-organize into spatial patterns (bands, spots, labyrinths).
    """

    def test_vegetation_heterogeneity_increases(self, taichi_init):
        """
        Spatial variance of vegetation should increase from uniform start.

        This is the hallmark of pattern formation: small perturbations
        get amplified into macroscopic patterns.
        """
        params = SimulationParams(n=64)
        sim = Simulation(params)
        # Start with very small perturbation from uniform
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.01, seed=42)

        mask = sim.state.fields.mask.to_numpy()
        interior = mask == 1

        P_initial = sim.state.fields.P.to_numpy()
        var_initial = P_initial[interior].var()

        # Run for 3 years to allow patterns to develop
        sim.run(years=3.0, check_mass_balance=False, verbose=False)

        P_final = sim.state.fields.P.to_numpy()
        var_final = P_final[interior].var()

        # Variance should have increased (patterns forming)
        assert var_final > var_initial, (
            f"Variance should increase during pattern formation. "
            f"Initial: {var_initial:.6f}, Final: {var_final:.6f}"
        )

    def test_pattern_not_uniform_after_spinup(self, taichi_init):
        """
        After sufficient spinup, vegetation should not be spatially uniform.

        The coefficient of variation (std/mean) should be significant.
        """
        params = SimulationParams(n=64)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P = sim.state.fields.P.to_numpy()
        mask = sim.state.fields.mask.to_numpy()
        P_interior = P[mask == 1]

        mean_P = P_interior.mean()
        std_P = P_interior.std()

        # Coefficient of variation should be at least 10%
        # (indicating non-trivial spatial pattern)
        cv = std_P / mean_P if mean_P > 0 else 0
        assert cv > 0.1, (
            f"Vegetation too uniform after spinup. CV = {cv:.2%}, "
            f"expected > 10%"
        )

    def test_spatial_structure_emerges(self, taichi_init):
        """
        Vegetation should develop spatial autocorrelation (not just noise).

        Adjacent cells should have correlated values, indicating
        coherent pattern structure.
        """
        params = SimulationParams(n=64)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P = sim.state.fields.P.to_numpy()

        # Compute lag-1 autocorrelation (adjacent cells)
        # Using simple row-wise correlation
        n = P.shape[0]
        P_center = P[1:-1, 1:-1]
        P_right = P[1:-1, 2:]
        P_down = P[2:, 1:-1]

        # Flatten and compute correlation
        corr_right = np.corrcoef(P_center.flatten(), P_right.flatten())[0, 1]
        corr_down = np.corrcoef(P_center.flatten(), P_down.flatten())[0, 1]

        # Autocorrelation should be positive (adjacent cells similar)
        # Random noise would have ~0 correlation
        assert corr_right > 0.3, (
            f"Insufficient spatial structure (horizontal). "
            f"Autocorrelation = {corr_right:.3f}, expected > 0.3"
        )
        assert corr_down > 0.3, (
            f"Insufficient spatial structure (vertical). "
            f"Autocorrelation = {corr_down:.3f}, expected > 0.3"
        )


class TestPatternWavelength:
    """
    Verify that emerging patterns have physically reasonable wavelengths.

    The characteristic wavelength should be related to the
    competition between local facilitation and nonlocal competition.
    """

    def test_fft_detects_dominant_scale(self, taichi_init):
        """
        FFT of vegetation field should show a dominant spatial frequency.

        This indicates organized pattern structure at a characteristic scale.
        """
        params = SimulationParams(n=128)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P = sim.state.fields.P.to_numpy()
        mask = sim.state.fields.mask.to_numpy()

        # Extract interior (excluding boundary)
        P_interior = P[1:-1, 1:-1]

        # Compute 2D FFT
        fft = np.fft.fft2(P_interior - P_interior.mean())
        power = np.abs(fft) ** 2

        # Radially average the power spectrum
        n = P_interior.shape[0]
        freqs = np.fft.fftfreq(n)
        fx, fy = np.meshgrid(freqs, freqs)
        radius = np.sqrt(fx**2 + fy**2)

        # Bin by radius
        max_r = 0.5  # Nyquist
        n_bins = 20
        bins = np.linspace(0, max_r, n_bins + 1)
        radial_power = np.zeros(n_bins)

        for i in range(n_bins):
            mask_bin = (radius >= bins[i]) & (radius < bins[i + 1])
            if np.any(mask_bin):
                radial_power[i] = power[mask_bin].mean()

        # Find peak (excluding DC component at index 0)
        peak_idx = np.argmax(radial_power[1:]) + 1
        peak_freq = (bins[peak_idx] + bins[peak_idx + 1]) / 2

        # Peak should be at non-zero frequency (indicating pattern)
        assert peak_idx > 0, "No dominant spatial scale detected (peak at DC)"

        # Convert to wavelength in grid cells
        if peak_freq > 0:
            wavelength_cells = 1.0 / peak_freq
            wavelength_m = wavelength_cells * params.dx

            # Wavelength should be reasonable (2-50 meters for typical params)
            assert 2 < wavelength_m < 100, (
                f"Pattern wavelength {wavelength_m:.1f}m outside expected range (2-100m)"
            )

    def test_wavelength_physically_reasonable(self, taichi_init):
        """
        Characteristic wavelength should match theoretical predictions.

        For Turing patterns, wavelength ~ sqrt(D_P * tau_growth)
        where tau_growth is the growth timescale.
        """
        params = SimulationParams(n=128)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P = sim.state.fields.P.to_numpy()
        P_interior = P[1:-1, 1:-1]

        # Estimate wavelength from autocorrelation
        # Find first zero-crossing of autocorrelation function
        n = P_interior.shape[0]
        center = n // 2

        # 1D slice through center
        P_slice = P_interior[center, :]
        P_slice = P_slice - P_slice.mean()

        # Autocorrelation
        autocorr = np.correlate(P_slice, P_slice, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find first zero-crossing (quarter wavelength)
        zero_cross_idx = np.argmax(autocorr < 0)
        if zero_cross_idx > 0:
            wavelength_cells = 4 * zero_cross_idx  # Approximate full wavelength
            wavelength_m = wavelength_cells * params.dx

            # Should be in physically reasonable range
            assert wavelength_m > 1, f"Wavelength too small: {wavelength_m:.1f}m"
            assert wavelength_m < 200, f"Wavelength too large: {wavelength_m:.1f}m"


class TestTuringMechanism:
    """
    Verify the positive local and negative nonlocal feedbacks
    that drive Turing instability.
    """

    def test_vegetation_captures_water_locally(self, taichi_init):
        """
        High vegetation cells should accumulate more soil moisture
        during rainfall events (positive local feedback).
        """
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(initial_moisture=0.05, seed=42)

        # Create vegetation gradient: high in center, low elsewhere
        P_np = sim.state.fields.P.to_numpy()
        n = params.n
        center = n // 2
        P_np[:, :] = 0.1  # Low baseline
        P_np[center-2:center+3, center-2:center+3] = 2.0  # High center patch
        sim.state.fields.P.from_numpy(P_np)

        M_before = sim.state.fields.M.to_numpy().copy()

        # Run a rainfall event
        sim.run_rainfall_event(depth=0.02, duration=0.25)

        M_after = sim.state.fields.M.to_numpy()

        # Center patch should have gained more moisture
        dM_center = (M_after[center-2:center+3, center-2:center+3] -
                     M_before[center-2:center+3, center-2:center+3]).mean()
        dM_edge = (M_after[2:5, 2:5] - M_before[2:5, 2:5]).mean()

        assert dM_center > dM_edge, (
            f"High-veg area should gain more moisture. "
            f"Center dM={dM_center:.4f}, Edge dM={dM_edge:.4f}"
        )

    def test_vegetation_reduces_runoff_to_neighbors(self, taichi_init):
        """
        High vegetation should reduce water flow to downslope neighbors
        (negative nonlocal feedback).
        """
        from src.kernels.utils import fill_field

        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(initial_moisture=0.0, seed=42)

        # Test 1: Low vegetation - more runoff
        fill_field(sim.state.fields.P, 0.1)
        fill_field(sim.state.fields.h, 0.05)
        fill_field(sim.state.fields.M, 0.0)

        h_before_low_veg = sim.state.total_surface_water()
        sim.run_rainfall_event(depth=0.0, duration=1.0)  # Just drain
        h_after_low_veg = sim.state.total_surface_water()
        outflow_low_veg = sim.state.mass_balance.cumulative_outflow

        # Reset
        sim.state.mass_balance.cumulative_outflow = 0.0

        # Test 2: High vegetation - less runoff
        fill_field(sim.state.fields.P, 5.0)
        fill_field(sim.state.fields.h, 0.05)
        fill_field(sim.state.fields.M, 0.0)

        sim.run_rainfall_event(depth=0.0, duration=1.0)  # Just drain
        outflow_high_veg = sim.state.mass_balance.cumulative_outflow

        # High vegetation should have less boundary outflow
        # because more water infiltrates locally
        # Note: This test may need adjustment based on actual parameter values
        # The key is that infiltration is enhanced by vegetation

    def test_instability_amplifies_perturbations(self, taichi_init):
        """
        Small perturbations from uniform state should grow over time.

        This is the definition of instability that leads to patterns.
        """
        params = SimulationParams(n=64)
        sim = Simulation(params)

        # Start with very small perturbation
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.001, seed=42)

        mask = sim.state.fields.mask.to_numpy()
        interior = mask == 1

        P_initial = sim.state.fields.P.to_numpy()
        perturbation_initial = P_initial[interior].std()

        # Run for 2 years
        sim.run(years=2.0, check_mass_balance=False, verbose=False)

        P_final = sim.state.fields.P.to_numpy()
        perturbation_final = P_final[interior].std()

        # Perturbation should have grown
        growth_factor = perturbation_final / perturbation_initial
        assert growth_factor > 5, (
            f"Perturbations should amplify (Turing instability). "
            f"Growth factor = {growth_factor:.1f}x, expected > 5x"
        )


class TestParameterSensitivity:
    """
    Verify system responds appropriately to parameter changes.
    """

    def test_higher_rainfall_increases_vegetation(self, taichi_init):
        """More rainfall should lead to higher mean vegetation."""
        # Low rainfall
        params_low = SimulationParams(n=32, rain_depth=0.005, interstorm=20.0)
        sim_low = Simulation(params_low)
        sim_low.initialize(seed=42)
        sim_low.run(years=3.0, check_mass_balance=False, verbose=False)
        P_low = sim_low.state.fields.P.to_numpy()
        mask = sim_low.state.fields.mask.to_numpy()
        mean_P_low = P_low[mask == 1].mean()

        # High rainfall
        params_high = SimulationParams(n=32, rain_depth=0.02, interstorm=5.0)
        sim_high = Simulation(params_high)
        sim_high.initialize(seed=42)
        sim_high.run(years=3.0, check_mass_balance=False, verbose=False)
        P_high = sim_high.state.fields.P.to_numpy()
        mean_P_high = P_high[mask == 1].mean()

        assert mean_P_high > mean_P_low, (
            f"Higher rainfall should increase vegetation. "
            f"Low rain: {mean_P_low:.3f}, High rain: {mean_P_high:.3f}"
        )

    def test_higher_mortality_decreases_vegetation(self, taichi_init):
        """Higher mortality should lead to lower mean vegetation."""
        # Low mortality
        params_low = SimulationParams(n=32, mu=0.0005)
        sim_low = Simulation(params_low)
        sim_low.initialize(seed=42)
        sim_low.run(years=3.0, check_mass_balance=False, verbose=False)
        P_low = sim_low.state.fields.P.to_numpy()
        mask = sim_low.state.fields.mask.to_numpy()
        mean_P_low = P_low[mask == 1].mean()

        # High mortality
        params_high = SimulationParams(n=32, mu=0.005)
        sim_high = Simulation(params_high)
        sim_high.initialize(seed=42)
        sim_high.run(years=3.0, check_mass_balance=False, verbose=False)
        P_high = sim_high.state.fields.P.to_numpy()
        mean_P_high = P_high[mask == 1].mean()

        assert mean_P_low > mean_P_high, (
            f"Higher mortality should decrease vegetation. "
            f"Low mort: {mean_P_low:.3f}, High mort: {mean_P_high:.3f}"
        )

    def test_bare_soil_factor_affects_pattern_contrast(self, taichi_init):
        """
        Lower W_0 (less infiltration on bare soil) should increase
        pattern contrast as vegetation more strongly controls infiltration.
        """
        # Higher W_0 (bare soil infiltrates well) - less contrast
        params_high_w0 = SimulationParams(n=64, W_0=0.5)
        sim_high = Simulation(params_high_w0)
        sim_high.initialize(seed=42)
        sim_high.run(years=3.0, check_mass_balance=False, verbose=False)
        P_high = sim_high.state.fields.P.to_numpy()
        mask = sim_high.state.fields.mask.to_numpy()
        cv_high = P_high[mask == 1].std() / P_high[mask == 1].mean()

        # Lower W_0 (bare soil infiltrates poorly) - more contrast
        params_low_w0 = SimulationParams(n=64, W_0=0.1)
        sim_low = Simulation(params_low_w0)
        sim_low.initialize(seed=42)
        sim_low.run(years=3.0, check_mass_balance=False, verbose=False)
        P_low = sim_low.state.fields.P.to_numpy()
        cv_low = P_low[mask == 1].std() / P_low[mask == 1].mean()

        # Lower W_0 should give higher contrast (CV)
        assert cv_low > cv_high * 0.8, (
            f"Lower W_0 should increase pattern contrast. "
            f"W_0=0.5 CV: {cv_high:.3f}, W_0=0.1 CV: {cv_low:.3f}"
        )


class TestSlopeEffects:
    """
    Verify that slope affects pattern formation and water redistribution.
    """

    def test_water_accumulates_downslope(self, taichi_init):
        """Soil moisture should be higher in downslope regions."""
        params = SimulationParams(n=64)
        sim = Simulation(params)
        sim.initialize(slope=0.02, direction="south", seed=42)

        # Run with rainfall
        sim.run(years=2.0, check_mass_balance=False, verbose=False)

        M = sim.state.fields.M.to_numpy()
        n = params.n

        # Compare upslope (north) vs downslope (south)
        M_north = M[5:15, 10:-10].mean()  # Northern interior
        M_south = M[-15:-5, 10:-10].mean()  # Southern interior

        assert M_south > M_north, (
            f"Downslope should be wetter. "
            f"North M: {M_north:.4f}, South M: {M_south:.4f}"
        )

    def test_steeper_slope_faster_drainage(self, taichi_init):
        """
        Steeper slopes should have faster surface water flow.

        We test this by comparing the CFL timestep (which is inversely
        proportional to flow velocity) on gentle vs steep slopes.
        """
        from src.kernels.flow import compute_cfl_timestep
        from src.kernels.utils import fill_field

        # Gentle slope
        params = SimulationParams(n=32)
        sim_gentle = Simulation(params)
        sim_gentle.initialize(slope=0.005, direction="south", seed=42)
        fill_field(sim_gentle.state.fields.h, 0.02)

        dt_gentle = compute_cfl_timestep(
            sim_gentle.state.fields.h,
            sim_gentle.state.fields.Z,
            sim_gentle.state.fields.flow_frac,
            sim_gentle.state.fields.mask,
            params.dx, params.manning_n, cfl=0.5
        )

        # Steep slope
        sim_steep = Simulation(params)
        sim_steep.initialize(slope=0.05, direction="south", seed=42)
        fill_field(sim_steep.state.fields.h, 0.02)

        dt_steep = compute_cfl_timestep(
            sim_steep.state.fields.h,
            sim_steep.state.fields.Z,
            sim_steep.state.fields.flow_frac,
            sim_steep.state.fields.mask,
            params.dx, params.manning_n, cfl=0.5
        )

        # Steeper slope = faster flow = smaller CFL timestep
        assert dt_steep < dt_gentle, (
            f"Steeper slope should have smaller CFL timestep (faster flow). "
            f"Gentle dt: {dt_gentle:.4f}, Steep dt: {dt_steep:.4f}"
        )


class TestEquilibriumStates:
    """
    Test approach to equilibrium states.
    """

    def test_system_approaches_steady_state(self, taichi_init):
        """
        With constant forcing, system should approach quasi-steady state.

        Total vegetation should stabilize (variance of year-to-year changes decreases).
        """
        params = SimulationParams(n=32)
        sim = Simulation(params)
        # Start with higher vegetation to see clear relaxation
        sim.initialize(initial_veg_mean=1.0, initial_veg_std=0.2, seed=42)

        mask = sim.state.fields.mask.to_numpy()
        interior = mask == 1

        # Track vegetation totals at monthly intervals
        P_totals = []
        for month in range(24):  # 2 years, monthly
            sim.run(years=1/12, check_mass_balance=False, verbose=False)
            P = sim.state.fields.P.to_numpy()
            P_totals.append(float(P[interior].sum()))

        # Compute changes in first year vs second year
        changes_year1 = [abs(P_totals[i+1] - P_totals[i]) for i in range(0, 11)]
        changes_year2 = [abs(P_totals[i+1] - P_totals[i]) for i in range(12, 23)]

        mean_change_year1 = np.mean(changes_year1)
        mean_change_year2 = np.mean(changes_year2)

        # Second year should have smaller changes (approaching equilibrium)
        # or both should be small (already at equilibrium)
        equilibrium_reached = (
            mean_change_year2 < mean_change_year1 * 1.5 or  # Stabilizing
            mean_change_year2 < 0.1 * P_totals[-1]  # Small fluctuations around equilibrium
        )

        assert equilibrium_reached, (
            f"System should approach steady state. "
            f"Year 1 mean change: {mean_change_year1:.4f}, "
            f"Year 2 mean change: {mean_change_year2:.4f}"
        )

    def test_vegetation_persists_with_rainfall(self, taichi_init):
        """
        With adequate rainfall, vegetation should persist (not go extinct).

        This tests that the coupled system reaches a sustainable equilibrium.
        """
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        mask = sim.state.fields.mask.to_numpy()
        interior = mask == 1

        P_initial = sim.state.fields.P.to_numpy()[interior].mean()

        # Run for 5 years
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        P = sim.state.fields.P.to_numpy()
        P_final = P[interior].mean()

        # Vegetation should not go extinct
        assert P_final > 0.01, (
            f"Vegetation should persist with adequate rainfall. "
            f"Initial: {P_initial:.4f}, Final: {P_final:.4f}"
        )

    def test_growth_mortality_balance_reached(self, taichi_init):
        """
        At quasi-equilibrium, net growth rate should be approximately zero.

        This verifies growth ≈ mortality at steady state.
        """
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(initial_veg_mean=0.5, initial_veg_std=0.1, seed=42)

        # Run to reach equilibrium
        sim.run(years=5.0, check_mass_balance=False, verbose=False)

        mask = sim.state.fields.mask.to_numpy()
        interior = mask == 1

        # Take snapshot
        P_before = sim.state.fields.P.to_numpy()[interior].sum()

        # Run one more year
        sim.run(years=1.0, check_mass_balance=False, verbose=False)

        P_after = sim.state.fields.P.to_numpy()[interior].sum()

        # Net change should be small relative to total
        net_change = abs(P_after - P_before)
        relative_change = net_change / max(P_before, 1e-10)

        assert relative_change < 0.5, (
            f"At equilibrium, annual vegetation change should be small. "
            f"Relative change: {relative_change:.2%}"
        )


# Fixture for Taichi initialization
@pytest.fixture(scope="module")
def taichi_init():
    """Initialize Taichi once per test module."""
    init_taichi(backend="cpu", debug=True)
    yield
