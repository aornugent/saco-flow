"""Tests for field management module.

Tests FieldSpec, FieldContainer, double-buffering, and convenience wrappers.
"""

import numpy as np
import pytest
import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import GridGeometry, NUM_NEIGHBORS
from src.fields import (
    FieldContainer,
    FieldRole,
    FieldSpec,
    ScratchFields,
    StateFields,
    StaticFields,
    create_all_specs,
    create_derived_specs,
    create_scratch_container,
    create_scratch_specs,
    create_simulation_container,
    create_state_container,
    create_state_specs,
    create_static_container,
    create_static_specs,
)


class TestFieldSpec:
    """Tests for FieldSpec dataclass."""

    def test_basic_creation(self):
        """Test basic FieldSpec creation."""
        spec = FieldSpec(
            name="h",
            dtype=DTYPE,
            role=FieldRole.STATE,
        )
        assert spec.name == "h"
        assert spec.dtype == DTYPE
        assert spec.role == FieldRole.STATE
        assert spec.double_buffer is False
        assert spec.extra_dims == ()

    def test_with_double_buffer(self):
        """Test FieldSpec with double buffering."""
        spec = FieldSpec(
            name="m",
            dtype=DTYPE,
            role=FieldRole.STATE,
            double_buffer=True,
        )
        assert spec.double_buffer is True
        assert spec.buffer_name() == "m_new"

    def test_with_extra_dims(self):
        """Test FieldSpec with extra dimensions."""
        spec = FieldSpec(
            name="flow_frac",
            dtype=DTYPE,
            role=FieldRole.STATIC,
            extra_dims=(8,),
        )
        assert spec.extra_dims == (8,)

    def test_immutability(self):
        """Test that FieldSpec is frozen."""
        spec = FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE)
        with pytest.raises(Exception):
            spec.name = "different"

    def test_validation_empty_name(self):
        """Test validation rejects empty name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            FieldSpec(name="", dtype=DTYPE, role=FieldRole.STATE)

    def test_validation_non_snake_case(self):
        """Test validation rejects non-snake_case names."""
        with pytest.raises(ValueError, match="must be snake_case"):
            FieldSpec(name="SurfaceWater", dtype=DTYPE, role=FieldRole.STATE)
        with pytest.raises(ValueError, match="must be snake_case"):
            FieldSpec(name="surface-water", dtype=DTYPE, role=FieldRole.STATE)

    def test_validation_static_double_buffer(self):
        """Test validation rejects double-buffered static fields."""
        with pytest.raises(ValueError, match="cannot be double-buffered"):
            FieldSpec(
                name="z",
                dtype=DTYPE,
                role=FieldRole.STATIC,
                double_buffer=True,
            )


class TestFieldRole:
    """Tests for FieldRole enum."""

    def test_roles_exist(self):
        """Test all expected roles exist."""
        assert FieldRole.STATE is not None
        assert FieldRole.STATIC is not None
        assert FieldRole.DERIVED is not None
        assert FieldRole.SCRATCH is not None

    def test_roles_distinct(self):
        """Test roles have distinct values."""
        roles = [FieldRole.STATE, FieldRole.STATIC, FieldRole.DERIVED, FieldRole.SCRATCH]
        assert len(set(roles)) == 4


class TestFieldContainer:
    """Tests for FieldContainer class."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return GridGeometry(nx=10, ny=10, dx=1.0)

    def test_creation(self, geometry):
        """Test basic container creation."""
        container = FieldContainer(geometry)
        assert container.geometry == geometry
        assert not container.allocated
        assert len(container) == 0

    def test_register_single(self, geometry):
        """Test registering a single field."""
        container = FieldContainer(geometry)
        spec = FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE)
        container.register(spec)
        assert "h" in container
        assert len(container) == 1

    def test_register_many(self, geometry):
        """Test registering multiple fields."""
        container = FieldContainer(geometry)
        specs = [
            FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE),
            FieldSpec(name="m", dtype=DTYPE, role=FieldRole.STATE),
        ]
        container.register_many(specs)
        assert len(container) == 2
        assert "h" in container
        assert "m" in container

    def test_duplicate_registration_fails(self, geometry):
        """Test that duplicate registration raises error."""
        container = FieldContainer(geometry)
        spec = FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE)
        container.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            container.register(spec)

    def test_allocate(self, geometry):
        """Test field allocation."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.allocate()
        assert container.allocated
        field = container["h"]
        assert field.shape == (10, 10)

    def test_allocate_with_extra_dims(self, geometry):
        """Test allocation with extra dimensions."""
        container = FieldContainer(geometry)
        container.register(
            FieldSpec(
                name="flow_frac",
                dtype=DTYPE,
                role=FieldRole.STATIC,
                extra_dims=(8,),
            )
        )
        container.allocate()
        field = container["flow_frac"]
        assert field.shape == (10, 10, 8)

    def test_allocate_double_buffer(self, geometry):
        """Test allocation creates double buffer."""
        container = FieldContainer(geometry)
        container.register(
            FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE, double_buffer=True)
        )
        container.allocate()
        h = container["h"]
        h_new = container.get_buffer("h")
        assert h.shape == h_new.shape

    def test_get_before_allocate_fails(self, geometry):
        """Test that accessing fields before allocation raises error."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        with pytest.raises(RuntimeError, match="not yet allocated"):
            container["h"]

    def test_get_nonexistent_fails(self, geometry):
        """Test that accessing nonexistent field raises error."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.allocate()
        with pytest.raises(KeyError):
            container["nonexistent"]

    def test_get_buffer_non_double_buffered_fails(self, geometry):
        """Test that get_buffer on non-double-buffered field fails."""
        container = FieldContainer(geometry)
        container.register(
            FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE, double_buffer=False)
        )
        container.allocate()
        with pytest.raises(ValueError, match="not double-buffered"):
            container.get_buffer("h")

    def test_register_after_allocate_fails(self, geometry):
        """Test that registration after allocation fails."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.allocate()
        with pytest.raises(RuntimeError, match="after allocation"):
            container.register(FieldSpec(name="m", dtype=DTYPE, role=FieldRole.STATE))

    def test_double_allocate_fails(self, geometry):
        """Test that allocating twice fails."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.allocate()
        with pytest.raises(RuntimeError, match="already allocated"):
            container.allocate()

    def test_field_names(self, geometry):
        """Test field_names property."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.register(FieldSpec(name="m", dtype=DTYPE, role=FieldRole.STATE))
        names = container.field_names
        assert "h" in names
        assert "m" in names
        assert len(names) == 2

    def test_fields_by_role(self, geometry):
        """Test filtering fields by role."""
        container = FieldContainer(geometry)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.register(FieldSpec(name="z", dtype=DTYPE, role=FieldRole.STATIC))
        container.register(FieldSpec(name="m", dtype=DTYPE, role=FieldRole.STATE))

        state_fields = container.fields_by_role(FieldRole.STATE)
        assert "h" in state_fields
        assert "m" in state_fields
        assert "z" not in state_fields

        static_fields = container.fields_by_role(FieldRole.STATIC)
        assert "z" in static_fields
        assert len(static_fields) == 1

    def test_get_spec(self, geometry):
        """Test retrieving field specification."""
        container = FieldContainer(geometry)
        spec = FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE, double_buffer=True)
        container.register(spec)
        retrieved = container.get_spec("h")
        assert retrieved == spec

    def test_memory_estimation(self, geometry):
        """Test memory estimation."""
        container = FieldContainer(geometry)
        # 10x10 field of f32 = 400 bytes
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        container.allocate()
        assert container.memory_bytes == 400

    def test_memory_estimation_double_buffer(self, geometry):
        """Test memory estimation with double buffer."""
        container = FieldContainer(geometry)
        # 10x10 field of f32 x2 = 800 bytes
        container.register(
            FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE, double_buffer=True)
        )
        container.allocate()
        assert container.memory_bytes == 800


class TestDoubleBufferSwap:
    """Tests for double-buffer swap operation."""

    @pytest.fixture
    def container(self):
        """Create container with double-buffered field."""
        geom = GridGeometry(nx=5, ny=5, dx=1.0)
        container = FieldContainer(geom)
        container.register(
            FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE, double_buffer=True)
        )
        container.allocate()
        return container

    def test_swap_exchanges_references(self, container):
        """Test that swap exchanges field references."""
        h = container["h"]
        h_new = container.get_buffer("h")

        # Store original references
        orig_h = h
        orig_h_new = h_new

        container.swap("h")

        # After swap, references should be exchanged
        assert container["h"] is orig_h_new
        assert container.get_buffer("h") is orig_h

    def test_swap_with_data(self, container):
        """Test swap preserves data correctly."""
        h = container["h"]
        h_new = container.get_buffer("h")

        # Set different values in each field
        h.fill(1.0)
        h_new.fill(2.0)

        # Verify initial state
        assert np.allclose(h.to_numpy(), 1.0)
        assert np.allclose(h_new.to_numpy(), 2.0)

        container.swap("h")

        # After swap, values should be exchanged
        h_after = container["h"]
        h_new_after = container.get_buffer("h")

        assert np.allclose(h_after.to_numpy(), 2.0)
        assert np.allclose(h_new_after.to_numpy(), 1.0)

    def test_swap_twice_restores_original(self, container):
        """Test that swapping twice restores original state."""
        h = container["h"]
        h.fill(42.0)

        orig_ref = container["h"]
        container.swap("h")
        container.swap("h")

        assert container["h"] is orig_ref

    def test_swap_non_double_buffered_fails(self):
        """Test swap on non-double-buffered field fails."""
        geom = GridGeometry(nx=5, ny=5, dx=1.0)
        container = FieldContainer(geom)
        container.register(
            FieldSpec(name="z", dtype=DTYPE, role=FieldRole.STATIC)
        )
        container.allocate()

        with pytest.raises(ValueError, match="not double-buffered"):
            container.swap("z")

    def test_stencil_operation_pattern(self, container):
        """Test typical stencil operation pattern using swap."""
        # Simulate: compute h_new from h neighbors, then swap

        @ti.kernel
        def stencil_step(h: ti.template(), h_new: ti.template()):
            """Simple averaging stencil (no neighbors, just demo)."""
            for i, j in h:
                h_new[i, j] = h[i, j] * 2.0

        h = container["h"]
        h_new = container.get_buffer("h")

        # Initialize
        h.fill(5.0)
        h_new.fill(0.0)

        # Run stencil
        stencil_step(h, h_new)

        # Swap to make result the new primary
        container.swap("h")

        # Verify result is now in primary field
        h_result = container["h"]
        assert np.allclose(h_result.to_numpy(), 10.0)


class TestStateFields:
    """Tests for StateFields convenience wrapper."""

    @pytest.fixture
    def container(self):
        """Create container with state fields."""
        return create_state_container(GridGeometry(nx=10, ny=10, dx=1.0))

    def test_access_h(self, container):
        """Test accessing h field."""
        state = StateFields(container)
        h = state.h
        assert h.shape == (10, 10)

    def test_access_h_new(self, container):
        """Test accessing h buffer."""
        state = StateFields(container)
        h_new = state.h_new
        assert h_new.shape == (10, 10)

    def test_access_all_state_fields(self, container):
        """Test accessing all state fields."""
        state = StateFields(container)
        assert state.h.shape == (10, 10)
        assert state.m.shape == (10, 10)
        assert state.p.shape == (10, 10)
        assert state.h_new.shape == (10, 10)
        assert state.m_new.shape == (10, 10)
        assert state.p_new.shape == (10, 10)

    def test_swap_h(self, container):
        """Test swap_h operation."""
        state = StateFields(container)
        state.h.fill(1.0)
        state.h_new.fill(2.0)

        state.swap_h()

        assert np.allclose(state.h.to_numpy(), 2.0)
        assert np.allclose(state.h_new.to_numpy(), 1.0)

    def test_swap_m(self, container):
        """Test swap_m operation."""
        state = StateFields(container)
        state.m.fill(3.0)
        state.m_new.fill(4.0)

        state.swap_m()

        assert np.allclose(state.m.to_numpy(), 4.0)

    def test_swap_p(self, container):
        """Test swap_p operation."""
        state = StateFields(container)
        state.p.fill(5.0)
        state.p_new.fill(6.0)

        state.swap_p()

        assert np.allclose(state.p.to_numpy(), 6.0)


class TestStaticFields:
    """Tests for StaticFields convenience wrapper."""

    @pytest.fixture
    def container(self):
        """Create container with static fields."""
        return create_static_container(GridGeometry(nx=10, ny=10, dx=1.0))

    def test_access_z(self, container):
        """Test accessing z field."""
        static = StaticFields(container)
        z = static.z
        assert z.shape == (10, 10)

    def test_access_mask(self, container):
        """Test accessing mask field."""
        static = StaticFields(container)
        mask = static.mask
        assert mask.shape == (10, 10)

    def test_access_flow_frac(self, container):
        """Test accessing flow_frac field."""
        static = StaticFields(container)
        ff = static.flow_frac
        assert ff.shape == (10, 10, NUM_NEIGHBORS)

    def test_initialize_tilted_plane_south(self, container):
        """Test tilted plane initialization (south)."""
        static = StaticFields(container)
        static.initialize_tilted_plane(slope=0.1, direction="south")

        z = static.z.to_numpy()
        mask = static.mask.to_numpy()

        # South-sloping: row 0 should be highest, row 9 lowest
        assert z[0, 5] > z[9, 5]

        # Boundary mask: edges should be 0
        assert mask[0, 5] == 0
        assert mask[9, 5] == 0
        assert mask[5, 0] == 0
        assert mask[5, 9] == 0

        # Interior should be 1
        assert mask[5, 5] == 1

    def test_initialize_tilted_plane_east(self, container):
        """Test tilted plane initialization (east)."""
        static = StaticFields(container)
        static.initialize_tilted_plane(slope=0.1, direction="east")

        z = static.z.to_numpy()
        # East-sloping: column 0 should be highest, column 9 lowest
        assert z[5, 0] > z[5, 9]

    def test_initialize_tilted_plane_invalid_direction(self, container):
        """Test tilted plane initialization with invalid direction."""
        static = StaticFields(container)
        with pytest.raises(ValueError, match="Unknown direction"):
            static.initialize_tilted_plane(direction="diagonal")

    def test_geometry_access(self, container):
        """Test accessing geometry through StaticFields."""
        static = StaticFields(container)
        assert static.geometry.nx == 10
        assert static.geometry.ny == 10


class TestScratchFields:
    """Tests for ScratchFields convenience wrapper."""

    @pytest.fixture
    def container(self):
        """Create container with scratch fields."""
        return create_scratch_container(GridGeometry(nx=10, ny=10, dx=1.0))

    def test_access_local_source(self, container):
        """Test accessing local_source field."""
        scratch = ScratchFields(container)
        ls = scratch.local_source
        assert ls.shape == (10, 10)

    def test_access_flow_acc(self, container):
        """Test accessing flow_acc field."""
        scratch = ScratchFields(container)
        fa = scratch.flow_acc
        assert fa.shape == (10, 10)

    def test_access_q_out(self, container):
        """Test accessing q_out field."""
        scratch = ScratchFields(container)
        qo = scratch.q_out
        assert qo.shape == (10, 10)

    def test_swap_flow_acc(self, container):
        """Test swap_flow_acc operation."""
        scratch = ScratchFields(container)
        scratch.flow_acc.fill(1.0)
        scratch.flow_acc_new.fill(2.0)

        scratch.swap_flow_acc()

        assert np.allclose(scratch.flow_acc.to_numpy(), 2.0)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_state_specs(self):
        """Test state spec factory."""
        specs = create_state_specs()
        names = [s.name for s in specs]
        assert "h" in names
        assert "m" in names
        assert "p" in names
        assert all(s.double_buffer for s in specs)
        assert all(s.role == FieldRole.STATE for s in specs)

    def test_create_static_specs(self):
        """Test static spec factory."""
        specs = create_static_specs()
        names = [s.name for s in specs]
        assert "z" in names
        assert "mask" in names
        assert "flow_frac" in names
        assert all(s.role == FieldRole.STATIC for s in specs)

    def test_create_derived_specs(self):
        """Test derived spec factory."""
        specs = create_derived_specs()
        names = [s.name for s in specs]
        assert "flow_acc" in names
        assert "q_out" in names

    def test_create_scratch_specs(self):
        """Test scratch spec factory."""
        specs = create_scratch_specs()
        names = [s.name for s in specs]
        assert "local_source" in names

    def test_create_all_specs(self):
        """Test combined spec factory."""
        specs = create_all_specs()
        names = [s.name for s in specs]
        # Should have all fields from all categories
        assert "h" in names
        assert "z" in names
        assert "flow_acc" in names
        assert "local_source" in names

    def test_create_simulation_container(self):
        """Test simulation container factory."""
        geom = GridGeometry(nx=20, ny=20, dx=2.0)
        container = create_simulation_container(geom)

        assert container.allocated
        assert container.geometry == geom

        # Should have all standard fields
        assert "h" in container
        assert "m" in container
        assert "p" in container
        assert "z" in container
        assert "mask" in container
        assert "flow_frac" in container

    def test_create_state_container(self):
        """Test state-only container factory."""
        geom = GridGeometry(nx=10, ny=10, dx=1.0)
        container = create_state_container(geom)

        assert container.allocated
        assert "h" in container
        assert "m" in container
        assert "p" in container
        assert "z" not in container


class TestMemoryTracking:
    """Tests for memory estimation."""

    def test_memory_scales_with_grid_size(self):
        """Test memory scales with grid dimensions."""
        small = create_state_container(GridGeometry(nx=10, ny=10, dx=1.0))
        large = create_state_container(GridGeometry(nx=20, ny=20, dx=1.0))

        # 4x more cells = 4x more memory
        ratio = large.memory_bytes / small.memory_bytes
        assert abs(ratio - 4.0) < 0.01

    def test_memory_mb_conversion(self):
        """Test MB conversion."""
        container = create_state_container(GridGeometry(nx=1000, ny=1000, dx=1.0))
        # 3 state fields x 2 (double buffer) x 1M cells x 4 bytes = 24,000,000 bytes
        # = 24,000,000 / (1024 * 1024) = ~22.89 MiB
        expected_mb = 24_000_000 / (1024 * 1024)
        assert abs(container.memory_mb - expected_mb) < 0.1

    def test_memory_before_allocate(self):
        """Test memory is 0 before allocation."""
        geom = GridGeometry(nx=10, ny=10, dx=1.0)
        container = FieldContainer(geom)
        container.register(FieldSpec(name="h", dtype=DTYPE, role=FieldRole.STATE))
        assert container.memory_bytes == 0
