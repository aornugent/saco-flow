"""
Tests for infiltration kernel.
"""

import numpy as np

from src.kernels.infiltration import infiltration_step
from src.fields import fill_field, copy_field
from src.diagnostics import compute_total


class TestInfiltration:
    """Test vegetation-enhanced infiltration."""

    def test_conservation_h_to_M(self, grid_factory, tilted_plane):
        """Water lost from surface equals water gained in soil: Δh = -ΔM."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Initial conditions
        fill_field(fields.h, 0.05)  # 5cm surface water
        fill_field(fields.M, 0.1)   # 10cm soil moisture
        fill_field(fields.P, 0.5)   # Some vegetation

        h_before = compute_total(fields.h, fields.mask)
        M_before = compute_total(fields.M, fields.mask)

        # Infiltration parameters
        alpha = 0.1    # infiltration rate
        k_P = 1.0      # vegetation half-saturation
        W_0 = 0.1      # bare soil fraction
        M_sat = 0.4    # saturation capacity
        dt = 0.1

        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha, k_P, W_0, M_sat, dt
        )

        h_after = compute_total(fields.h, fields.mask)
        M_after = compute_total(fields.M, fields.mask)

        # Conservation: Δh + ΔM = 0 (with f32 tolerance for atomic operations)
        delta_h = h_after - h_before
        delta_M = M_after - M_before
        assert abs(delta_h + delta_M) < 1e-3, f"Δh={delta_h}, ΔM={delta_M}"

    def test_no_infiltration_when_saturated(self, grid_factory, tilted_plane):
        """No infiltration occurs when soil is saturated."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        M_sat = 0.4
        fill_field(fields.h, 0.1)
        fill_field(fields.M, M_sat)  # Fully saturated
        fill_field(fields.P, 0.5)

        h_before = compute_total(fields.h, fields.mask)

        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha=0.1, k_P=1.0, W_0=0.1, M_sat=M_sat, dt=0.1
        )

        h_after = compute_total(fields.h, fields.mask)
        assert abs(h_after - h_before) < 1e-8, "Water infiltrated into saturated soil"

    def test_no_infiltration_when_dry_surface(self, grid_factory, tilted_plane):
        """No infiltration occurs when surface is dry."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.h, 0.0)   # No surface water
        fill_field(fields.M, 0.1)
        fill_field(fields.P, 0.5)

        M_before = compute_total(fields.M, fields.mask)

        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha=0.1, k_P=1.0, W_0=0.1, M_sat=0.4, dt=0.1
        )

        M_after = compute_total(fields.M, fields.mask)
        assert abs(M_after - M_before) < 1e-8, "Moisture changed without surface water"

    def test_vegetation_enhances_infiltration(self, grid_factory, tilted_plane):
        """Higher vegetation leads to more infiltration."""
        n = 16

        # Run with low vegetation
        fields_low = grid_factory(n=n)
        tilted_plane(fields_low)
        fill_field(fields_low.h, 0.1)
        fill_field(fields_low.M, 0.1)
        fill_field(fields_low.P, 0.0)  # No vegetation

        h_before_low = compute_total(fields_low.h, fields_low.mask)
        infiltration_step(
            fields_low.h, fields_low.M, fields_low.P, fields_low.mask,
            alpha=0.1, k_P=1.0, W_0=0.1, M_sat=0.4, dt=0.1
        )
        h_after_low = compute_total(fields_low.h, fields_low.mask)
        infiltration_low = h_before_low - h_after_low

        # Run with high vegetation
        fields_high = grid_factory(n=n)
        tilted_plane(fields_high)
        fill_field(fields_high.h, 0.1)
        fill_field(fields_high.M, 0.1)
        fill_field(fields_high.P, 10.0)  # Dense vegetation

        h_before_high = compute_total(fields_high.h, fields_high.mask)
        infiltration_step(
            fields_high.h, fields_high.M, fields_high.P, fields_high.mask,
            alpha=0.1, k_P=1.0, W_0=0.1, M_sat=0.4, dt=0.1
        )
        h_after_high = compute_total(fields_high.h, fields_high.mask)
        infiltration_high = h_before_high - h_after_high

        assert infiltration_high > infiltration_low, \
            f"Veg: low={infiltration_low}, high={infiltration_high}"

    def test_infiltration_limited_by_available_water(self, grid_factory, tilted_plane):
        """Infiltration cannot exceed available surface water."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Very little surface water
        fill_field(fields.h, 0.001)
        fill_field(fields.M, 0.0)
        fill_field(fields.P, 1.0)

        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha=10.0,  # High rate that would exceed available
            k_P=1.0, W_0=0.1, M_sat=0.4, dt=1.0
        )

        h_np = fields.h.to_numpy()
        assert np.all(h_np >= 0), "Negative surface water"

    def test_infiltration_limited_by_capacity(self, grid_factory, tilted_plane):
        """Infiltration cannot exceed remaining soil capacity."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        M_sat = 0.4
        fill_field(fields.h, 0.5)       # Lots of water
        fill_field(fields.M, 0.39)      # Almost saturated
        fill_field(fields.P, 1.0)

        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha=10.0, k_P=1.0, W_0=0.1, M_sat=M_sat, dt=1.0
        )

        M_np = fields.M.to_numpy()
        assert np.all(M_np <= M_sat + 1e-6), "Soil moisture exceeded saturation"

    def test_returns_total_infiltration(self, grid_factory, tilted_plane):
        """Kernel returns correct total infiltration volume."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        fill_field(fields.h, 0.1)
        fill_field(fields.M, 0.1)
        fill_field(fields.P, 0.5)

        h_before = compute_total(fields.h, fields.mask)

        total_inf = infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            alpha=0.1, k_P=1.0, W_0=0.1, M_sat=0.4, dt=0.1
        )

        h_after = compute_total(fields.h, fields.mask)
        expected = h_before - h_after

        # Relaxed tolerance for f32 atomic accumulation
        assert abs(total_inf - expected) < 1e-4, f"Returned {total_inf}, expected {expected}"
