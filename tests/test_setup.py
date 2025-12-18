"""
Phase 0 verification tests.

Tests that the development infrastructure is working:
- Taichi initialization
- Field creation
- Basic kernel execution
- Conservation fixtures
"""

import numpy as np
import pytest
import taichi as ti

from src.config import DTYPE, DefaultParams, get_backend
from src.fields import fill_field, copy_field, add_uniform, clamp_field
from src.diagnostics import compute_total


class TestTaichiInit:
    """Test Taichi initialization and configuration."""

    def test_taichi_initialized(self, taichi_init):
        """Verify Taichi is initialized."""
        # If we get here without error, Taichi is initialized
        # Create a simple field to verify
        test_field = ti.field(dtype=ti.f32, shape=(4, 4))
        test_field[0, 0] = 1.0
        assert test_field[0, 0] == 1.0

    def test_backend_detection(self):
        """Test backend auto-detection returns valid backend."""
        backend = get_backend()
        assert backend in ("cuda", "vulkan", "cpu")

    def test_default_dtype(self):
        """Verify default dtype is f32."""
        assert DTYPE == ti.f32


class TestGridFactory:
    """Test grid factory fixture."""

    def test_creates_fields(self, grid_factory):
        """Grid factory creates all required fields."""
        fields = grid_factory(n=16)

        assert hasattr(fields, "h")
        assert hasattr(fields, "M")
        assert hasattr(fields, "P")
        assert hasattr(fields, "Z")
        assert hasattr(fields, "mask")
        assert hasattr(fields, "flow_frac")
        assert hasattr(fields, "n")

    def test_field_shapes(self, grid_factory):
        """Fields have correct shapes."""
        n = 32
        fields = grid_factory(n=n)

        assert fields.h.shape == (n, n)
        assert fields.M.shape == (n, n)
        assert fields.P.shape == (n, n)
        assert fields.Z.shape == (n, n)
        assert fields.mask.shape == (n, n)
        assert fields.flow_frac.shape == (n, n, 8)

    def test_double_buffers_exist(self, grid_factory):
        """Double buffer fields are created."""
        fields = grid_factory(n=16)

        assert hasattr(fields, "h_new")
        assert hasattr(fields, "M_new")
        assert hasattr(fields, "P_new")


class TestSyntheticTerrain:
    """Test synthetic terrain generators."""

    def test_tilted_plane_south(self, grid_factory, tilted_plane):
        """Tilted plane decreases toward south."""
        fields = grid_factory(n=16)
        tilted_plane(fields, slope=0.1, direction="south")

        Z_np = fields.Z.to_numpy()
        # Row 0 should be higher than row -1
        assert Z_np[0, 8] > Z_np[-1, 8]

    def test_tilted_plane_east(self, grid_factory, tilted_plane):
        """Tilted plane decreases toward east."""
        fields = grid_factory(n=16)
        tilted_plane(fields, slope=0.1, direction="east")

        Z_np = fields.Z.to_numpy()
        # Column 0 should be higher than column -1
        assert Z_np[8, 0] > Z_np[8, -1]

    def test_valley_converges_to_center(self, grid_factory, valley_terrain):
        """Valley terrain lowest at center column."""
        n = 32
        fields = grid_factory(n=n)
        valley_terrain(fields, slope=0.01, valley_depth=0.5)

        Z_np = fields.Z.to_numpy()
        center = n // 2
        row = n // 2  # Middle row
        # Center should be lower than edges at same row
        assert Z_np[row, center] < Z_np[row, 0]
        assert Z_np[row, center] < Z_np[row, -1]

    def test_flat_terrain(self, grid_factory, flat_terrain):
        """Flat terrain has zero elevation."""
        fields = grid_factory(n=16)
        flat_terrain(fields)

        Z_np = fields.Z.to_numpy()
        assert np.allclose(Z_np, 0.0)

    def test_mask_boundaries(self, grid_factory, tilted_plane):
        """Terrain generators set boundary mask correctly."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        mask_np = fields.mask.to_numpy()

        # Boundaries should be 0
        assert np.all(mask_np[0, :] == 0)
        assert np.all(mask_np[-1, :] == 0)
        assert np.all(mask_np[:, 0] == 0)
        assert np.all(mask_np[:, -1] == 0)

        # Interior should be 1
        assert np.all(mask_np[1:-1, 1:-1] == 1)


class TestUtilityKernels:
    """Test utility kernels execute correctly."""

    def test_fill_field(self, grid_factory):
        """fill_field sets all values."""
        fields = grid_factory(n=16)
        fill_field(fields.h, 0.5)

        h_np = fields.h.to_numpy()
        assert np.allclose(h_np, 0.5)

    def test_copy_field(self, grid_factory):
        """copy_field duplicates values."""
        fields = grid_factory(n=16)
        fill_field(fields.h, 1.23)
        fill_field(fields.h_new, 0.0)

        copy_field(fields.h, fields.h_new)

        h_new_np = fields.h_new.to_numpy()
        assert np.allclose(h_new_np, 1.23)

    def test_compute_total_masked(self, grid_factory, tilted_plane):
        """compute_total respects mask."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)

        # Fill with 1.0
        fill_field(fields.h, 1.0)

        total = compute_total(fields.h, fields.mask)

        # Interior cells: (n-2) x (n-2)
        expected = (n - 2) * (n - 2)
        assert abs(total - expected) < 1e-5

    def test_add_uniform(self, grid_factory, tilted_plane):
        """add_uniform adds to masked cells only."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)
        fill_field(fields.h, 0.0)

        add_uniform(fields.h, fields.mask, 0.1)

        h_np = fields.h.to_numpy()
        mask_np = fields.mask.to_numpy()

        # Masked cells should have 0.1
        assert np.allclose(h_np[mask_np == 1], 0.1)
        # Boundary cells should still be 0
        assert np.allclose(h_np[mask_np == 0], 0.0)

    def test_clamp_field(self, grid_factory):
        """clamp_field respects bounds."""
        fields = grid_factory(n=16)

        # Set values outside range
        h_np = np.linspace(-1, 2, 16 * 16).reshape(16, 16).astype(np.float32)
        fields.h.from_numpy(h_np)

        clamp_field(fields.h, 0.0, 1.0)

        h_clamped = fields.h.to_numpy()
        assert h_clamped.min() >= 0.0
        assert h_clamped.max() <= 1.0


class TestConservationFixtures:
    """Test mass conservation assertion helpers."""

    def test_conserved_passes(self, assert_mass_conserved):
        """Conservation check passes when mass is conserved."""
        # No exception should be raised
        assert_mass_conserved(100.0, 100.0)

    def test_conserved_with_flux(self, assert_mass_conserved):
        """Conservation check accounts for fluxes."""
        initial = 100.0
        outflow = 10.0
        final = 90.0
        # Should pass: 100 - 10 = 90
        assert_mass_conserved(initial, final, fluxes={"outflow": outflow})

    def test_not_conserved_fails(self, assert_mass_conserved):
        """Conservation check fails when mass lost."""
        with pytest.raises(AssertionError, match="Mass not conserved"):
            assert_mass_conserved(100.0, 50.0)

    def test_compute_total_mass(self, grid_factory, tilted_plane, compute_total_mass):
        """compute_total_mass works correctly."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields)
        fill_field(fields.h, 1.0)

        total = compute_total_mass(fields.h, fields.mask)
        expected = (n - 2) * (n - 2)
        assert abs(total - expected) < 1e-5


class TestDefaultParameters:
    """Test default parameter values are reasonable."""

    def test_parameters_exist(self):
        """All expected parameters are defined."""
        p = DefaultParams

        # Grid
        assert hasattr(p, "DX")

        # Rainfall
        assert hasattr(p, "R_MEAN")
        assert hasattr(p, "STORM_DURATION")
        assert hasattr(p, "INTERSTORM")

        # Infiltration
        assert hasattr(p, "K_SAT")
        assert hasattr(p, "ALPHA_I")

        # Soil
        assert hasattr(p, "M_SAT")
        assert hasattr(p, "ET_MAX")
        assert hasattr(p, "LEAKAGE")
        assert hasattr(p, "D_SOIL")

        # Vegetation
        assert hasattr(p, "G_MAX")
        assert hasattr(p, "K_G")
        assert hasattr(p, "MORTALITY")
        assert hasattr(p, "D_VEG")

        # Routing
        assert hasattr(p, "MANNING_N")
        assert hasattr(p, "MIN_SLOPE")

    def test_parameters_positive(self):
        """Physical parameters are positive."""
        p = DefaultParams

        assert p.DX > 0
        assert p.R_MEAN > 0
        assert p.K_SAT > 0
        assert p.M_SAT > 0
        assert p.ET_MAX > 0
