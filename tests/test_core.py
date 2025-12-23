"""Tests for core modules: geometry, params, fields."""

import numpy as np
import pytest
import taichi as ti

from src.fields import (
    add_uniform,
    allocate,
    copy_field,
    fill_field,
    swap_buffers,
)
from src.geometry import (
    DTYPE,
    NEIGHBOR_DI,
    NEIGHBOR_DIST,
    NEIGHBOR_DJ,
    NUM_NEIGHBORS,
    get_neighbor,
    is_interior,
)
from src.initialization import (
    initialize_mask,
    initialize_tilted_plane,
    initialize_vegetation,
)
from src.params import SimulationParams


class TestGeometry:
    """Tests for geometry module."""

    def test_dtype_is_f32(self):
        """DTYPE should be ti.f32."""
        assert DTYPE == ti.f32

    def test_num_neighbors(self):
        """Should have 8 neighbors."""
        assert NUM_NEIGHBORS == 8

    def test_neighbor_vectors_shape(self):
        """Neighbor vectors should have 8 elements."""
        assert NEIGHBOR_DI.n == 8
        assert NEIGHBOR_DJ.n == 8
        assert NEIGHBOR_DIST.n == 8

    def test_is_interior(self, taichi_init):
        """Test is_interior function."""
        @ti.kernel
        def check() -> ti.i32:
            result = 0
            # Interior should be True
            if is_interior(5, 5, 10):
                result += 1
            # Boundary should be False
            if not is_interior(0, 5, 10):
                result += 1
            if not is_interior(9, 5, 10):
                result += 1
            return result

        assert check() == 3

    def test_get_neighbor(self, taichi_init):
        """Test get_neighbor returns correct coords."""
        @ti.kernel
        def check_east() -> ti.i32:
            nb = get_neighbor(5, 5, 0)  # East
            return nb[0] * 100 + nb[1]

        # East neighbor of (5,5) should be (5,6)
        assert check_east() == 506


class TestParams:
    """Tests for params module."""

    def test_default_creation(self):
        """Default params should be valid."""
        params = SimulationParams()
        assert params.n == 64
        assert params.dx == 1.0
        assert params.M_sat == 0.4

    def test_custom_values(self):
        """Custom params should be stored."""
        params = SimulationParams(n=128, dx=2.0, M_sat=0.5)
        assert params.n == 128
        assert params.dx == 2.0
        assert params.M_sat == 0.5

    def test_validation_n_too_small(self):
        """n < 3 should fail validation."""
        with pytest.raises(ValueError, match="n must be >= 3"):
            SimulationParams(n=2)

    def test_validation_negative_dx(self):
        """Negative dx should fail validation."""
        with pytest.raises(ValueError, match="must be positive"):
            SimulationParams(dx=-1.0)

    def test_validation_W_0_fraction(self):
        """W_0 must be in [0, 1]."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            SimulationParams(W_0=1.5)

    def test_cell_area_property(self):
        """Cell area should be dx^2."""
        params = SimulationParams(dx=5.0)
        assert params.cell_area == 25.0

    def test_events_per_year(self):
        """Events per year calculation."""
        params = SimulationParams(interstorm=18.25)  # 365/20
        assert abs(params.events_per_year - 20.0) < 0.1


class TestFields:
    """Tests for fields module."""

    def test_allocate(self, taichi_init):
        """allocate should create all fields."""
        fields = allocate(32)
        assert fields.n == 32
        assert fields.h.shape == (32, 32)
        assert fields.M.shape == (32, 32)
        assert fields.P.shape == (32, 32)
        assert fields.Z.shape == (32, 32)
        assert fields.mask.shape == (32, 32)
        assert fields.flow_frac.shape == (32, 32, 8)

    def test_double_buffers(self, taichi_init):
        """Should have double buffers for state fields."""
        fields = allocate(16)
        assert fields.h_new.shape == (16, 16)
        assert fields.M_new.shape == (16, 16)
        assert fields.P_new.shape == (16, 16)

    def test_swap_buffers(self, taichi_init):
        """swap_buffers should exchange field references."""
        fields = allocate(8)
        fields.h.fill(1.0)
        fields.h_new.fill(2.0)

        # After swap, h should have value 2.0
        swap_buffers(fields, "h")

        assert np.allclose(fields.h.to_numpy(), 2.0)
        assert np.allclose(fields.h_new.to_numpy(), 1.0)

    def test_initialize_mask(self, taichi_init):
        """initialize_mask should set boundaries to 0."""
        fields = allocate(10)
        initialize_mask(fields)

        mask_np = fields.mask.to_numpy()
        # Boundaries should be 0
        assert mask_np[0, 5] == 0
        assert mask_np[9, 5] == 0
        assert mask_np[5, 0] == 0
        assert mask_np[5, 9] == 0
        # Interior should be 1
        assert mask_np[5, 5] == 1

    def test_initialize_tilted_plane(self, taichi_init):
        """Tilted plane should slope in correct direction."""
        fields = allocate(16)
        initialize_tilted_plane(fields, slope=0.1, direction="south")

        Z_np = fields.Z.to_numpy()
        # North row should be higher than south row
        assert Z_np[1, 8] > Z_np[14, 8]

    def test_initialize_vegetation(self, taichi_init):
        """Vegetation should be initialized with random values."""
        fields = allocate(32)
        initialize_tilted_plane(fields)
        initialize_vegetation(fields, mean=1.0, std=0.2, seed=42)

        P_np = fields.P.to_numpy()
        assert np.mean(P_np) > 0.5
        assert np.std(P_np) > 0.1
        assert np.all(P_np >= 0)

    def test_fill_field(self, taichi_init):
        """fill_field should set all values."""
        fields = allocate(8)
        fill_field(fields.h, 5.0)
        assert np.allclose(fields.h.to_numpy(), 5.0)

    def test_copy_field(self, taichi_init):
        """copy_field should copy values."""
        fields = allocate(8)
        fill_field(fields.h, 3.0)
        copy_field(fields.h, fields.h_new)
        assert np.allclose(fields.h_new.to_numpy(), 3.0)

    def test_add_uniform(self, taichi_init):
        """add_uniform should add to masked cells only."""
        fields = allocate(8)
        initialize_mask(fields)
        fill_field(fields.h, 1.0)
        add_uniform(fields.h, fields.mask, 2.0)

        h_np = fields.h.to_numpy()
        # mask_np = fields.mask.to_numpy()
        # Interior should be 3.0
        assert h_np[4, 4] == 3.0
        # Boundary should still be 1.0 (not modified)
        assert h_np[0, 4] == 1.0


# Note: taichi_init fixture is provided by conftest.py (session-scoped)
