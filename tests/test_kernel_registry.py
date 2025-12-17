"""Tests for the kernel registry and protocol system.

Tests cover:
- Registry instantiation and kernel retrieval
- Protocol compliance of wrapper classes
- Variant registration and selection
- Error handling for missing variants
"""

import pytest
import taichi as ti
import numpy as np

from src.kernels import (
    KernelRegistry,
    KernelVariant,
    get_registry,
    # Protocol types
    SoilKernel,
    VegetationKernel,
    InfiltrationKernel,
    FlowKernel,
    FlowDirectionKernel,
    # Result types
    SoilFluxes,
    VegetationFluxes,
    InfiltrationFluxes,
    RoutingFluxes,
    # Naive implementations
    NaiveSoilKernel,
    NaiveVegetationKernel,
    NaiveInfiltrationKernel,
    NaiveFlowKernel,
    NaiveFlowDirectionKernel,
)
from src.core.geometry import GridGeometry
from src.core.dtypes import DTYPE
from src.fields.base import FieldContainer
from src.fields.state import create_state_specs, StateFields
from src.fields.static import create_static_specs, StaticFields
from src.fields.scratch import create_scratch_specs, create_derived_specs, ScratchFields
from src.params.schema import (
    SimulationConfig,
    SoilParams,
    VegetationParams,
    InfiltrationParams,
    RoutingParams,
)


# --- Fixtures ---


@pytest.fixture
def geometry():
    """Create a small test grid geometry."""
    return GridGeometry(nx=16, ny=16, dx=1.0)


@pytest.fixture
def field_container(geometry):
    """Create a fully populated field container."""
    container = FieldContainer(geometry)
    container.register_many(create_state_specs())
    container.register_many(create_static_specs())
    container.register_many(create_scratch_specs())
    container.register_many(create_derived_specs())
    container.allocate()
    return container


@pytest.fixture
def state_fields(field_container):
    """Create StateFields wrapper."""
    return StateFields(field_container)


@pytest.fixture
def static_fields(field_container):
    """Create StaticFields wrapper."""
    return StaticFields(field_container)


@pytest.fixture
def scratch_fields(field_container):
    """Create ScratchFields wrapper."""
    return ScratchFields(field_container)


@pytest.fixture
def config():
    """Create default simulation config."""
    return SimulationConfig()


# --- Registry Tests ---


class TestKernelRegistryBasics:
    """Test basic registry functionality."""

    def test_create_registry(self):
        """Registry can be instantiated."""
        registry = KernelRegistry()
        assert registry is not None

    def test_get_default_registry(self):
        """Global registry is accessible."""
        registry = get_registry()
        assert registry is not None
        assert isinstance(registry, KernelRegistry)

    def test_get_soil_naive(self):
        """Can retrieve naive soil kernel."""
        registry = KernelRegistry()
        kernel = registry.get_soil(KernelVariant.NAIVE)
        assert kernel is not None
        assert isinstance(kernel, NaiveSoilKernel)

    def test_get_vegetation_naive(self):
        """Can retrieve naive vegetation kernel."""
        registry = KernelRegistry()
        kernel = registry.get_vegetation(KernelVariant.NAIVE)
        assert kernel is not None
        assert isinstance(kernel, NaiveVegetationKernel)

    def test_get_infiltration_naive(self):
        """Can retrieve naive infiltration kernel."""
        registry = KernelRegistry()
        kernel = registry.get_infiltration(KernelVariant.NAIVE)
        assert kernel is not None
        assert isinstance(kernel, NaiveInfiltrationKernel)

    def test_get_flow_naive(self):
        """Can retrieve naive flow kernel."""
        registry = KernelRegistry()
        kernel = registry.get_flow(KernelVariant.NAIVE)
        assert kernel is not None
        assert isinstance(kernel, NaiveFlowKernel)

    def test_get_flow_direction_naive(self):
        """Can retrieve naive flow direction kernel."""
        registry = KernelRegistry()
        kernel = registry.get_flow_direction(KernelVariant.NAIVE)
        assert kernel is not None
        assert isinstance(kernel, NaiveFlowDirectionKernel)

    def test_default_variant_is_naive(self):
        """Default variant is NAIVE when not specified."""
        registry = KernelRegistry()
        soil = registry.get_soil()
        veg = registry.get_vegetation()
        inf = registry.get_infiltration()
        flow = registry.get_flow()

        assert isinstance(soil, NaiveSoilKernel)
        assert isinstance(veg, NaiveVegetationKernel)
        assert isinstance(inf, NaiveInfiltrationKernel)
        assert isinstance(flow, NaiveFlowKernel)


class TestKernelRegistryErrors:
    """Test registry error handling."""

    def test_missing_variant_soil(self):
        """Raises KeyError for unregistered soil variant."""
        registry = KernelRegistry()
        with pytest.raises(KeyError, match="soil"):
            registry.get_soil(KernelVariant.FUSED)

    def test_missing_variant_vegetation(self):
        """Raises KeyError for unregistered vegetation variant."""
        registry = KernelRegistry()
        with pytest.raises(KeyError, match="vegetation"):
            registry.get_vegetation(KernelVariant.TEMPORAL)

    def test_available_variants_soil(self):
        """Can list available variants for soil."""
        registry = KernelRegistry()
        variants = registry.available_variants("soil")
        assert KernelVariant.NAIVE in variants

    def test_available_variants_unknown_type(self):
        """Raises ValueError for unknown kernel type."""
        registry = KernelRegistry()
        with pytest.raises(ValueError, match="Unknown kernel type"):
            registry.available_variants("unknown")


class TestKernelRegistration:
    """Test custom kernel registration."""

    def test_register_soil_variant(self):
        """Can register custom soil variant."""
        registry = KernelRegistry()

        # Create a mock kernel class
        class MockSoilKernel:
            def step(self, state, static, params, dx, dt):
                return SoilFluxes(total_et=0.0, total_leakage=0.0)

            @property
            def fields_read(self):
                return set()

            @property
            def fields_written(self):
                return set()

        registry.register_soil(KernelVariant.FUSED, MockSoilKernel)
        kernel = registry.get_soil(KernelVariant.FUSED)
        assert isinstance(kernel, MockSoilKernel)

    def test_registration_doesnt_affect_other_registries(self):
        """Registration on one registry doesn't affect others."""
        registry1 = KernelRegistry()
        registry2 = KernelRegistry()

        class MockSoilKernel:
            pass

        registry1.register_soil(KernelVariant.FUSED, MockSoilKernel)

        # registry2 should not have the FUSED variant
        with pytest.raises(KeyError):
            registry2.get_soil(KernelVariant.FUSED)


# --- Protocol Compliance Tests ---


class TestNaiveSoilKernelProtocol:
    """Test NaiveSoilKernel implements SoilKernel protocol."""

    def test_has_step_method(self):
        """Kernel has step method."""
        kernel = NaiveSoilKernel()
        assert hasattr(kernel, "step")
        assert callable(kernel.step)

    def test_has_fields_read(self):
        """Kernel has fields_read property."""
        kernel = NaiveSoilKernel()
        fields = kernel.fields_read
        assert isinstance(fields, set)
        assert "m" in fields
        assert "p" in fields
        assert "mask" in fields

    def test_has_fields_written(self):
        """Kernel has fields_written property."""
        kernel = NaiveSoilKernel()
        fields = kernel.fields_written
        assert isinstance(fields, set)
        assert "m" in fields

    def test_step_returns_soil_fluxes(
        self, state_fields, static_fields, config, geometry
    ):
        """Kernel step returns SoilFluxes."""
        # Initialize fields
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        kernel = NaiveSoilKernel()
        result = kernel.step(state_fields, static_fields, config, geometry.dx, 1.0)

        assert isinstance(result, SoilFluxes)
        assert hasattr(result, "total_et")
        assert hasattr(result, "total_leakage")


class TestNaiveVegetationKernelProtocol:
    """Test NaiveVegetationKernel implements VegetationKernel protocol."""

    def test_has_step_method(self):
        """Kernel has step method."""
        kernel = NaiveVegetationKernel()
        assert hasattr(kernel, "step")
        assert callable(kernel.step)

    def test_has_fields_read(self):
        """Kernel has fields_read property."""
        kernel = NaiveVegetationKernel()
        fields = kernel.fields_read
        assert isinstance(fields, set)
        assert "m" in fields
        assert "p" in fields
        assert "mask" in fields

    def test_has_fields_written(self):
        """Kernel has fields_written property."""
        kernel = NaiveVegetationKernel()
        fields = kernel.fields_written
        assert isinstance(fields, set)
        assert "p" in fields

    def test_step_returns_vegetation_fluxes(
        self, state_fields, static_fields, config, geometry
    ):
        """Kernel step returns VegetationFluxes."""
        # Initialize fields
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        kernel = NaiveVegetationKernel()
        result = kernel.step(state_fields, static_fields, config, geometry.dx, 1.0)

        assert isinstance(result, VegetationFluxes)
        assert hasattr(result, "total_growth")
        assert hasattr(result, "total_mortality")


class TestNaiveInfiltrationKernelProtocol:
    """Test NaiveInfiltrationKernel implements InfiltrationKernel protocol."""

    def test_has_step_method(self):
        """Kernel has step method."""
        kernel = NaiveInfiltrationKernel()
        assert hasattr(kernel, "step")
        assert callable(kernel.step)

    def test_has_fields_read(self):
        """Kernel has fields_read property."""
        kernel = NaiveInfiltrationKernel()
        fields = kernel.fields_read
        assert isinstance(fields, set)
        assert "h" in fields
        assert "m" in fields
        assert "p" in fields
        assert "mask" in fields

    def test_has_fields_written(self):
        """Kernel has fields_written property."""
        kernel = NaiveInfiltrationKernel()
        fields = kernel.fields_written
        assert isinstance(fields, set)
        assert "h" in fields
        assert "m" in fields

    def test_step_returns_infiltration_fluxes(
        self, state_fields, static_fields, config
    ):
        """Kernel step returns InfiltrationFluxes."""
        # Initialize fields
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.h.from_numpy(np.full((16, 16), 0.01, dtype=np.float32))
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        kernel = NaiveInfiltrationKernel()
        result = kernel.step(state_fields, static_fields, config, 0.1)

        assert isinstance(result, InfiltrationFluxes)
        assert hasattr(result, "total_infiltration")


class TestNaiveFlowKernelProtocol:
    """Test NaiveFlowKernel implements FlowKernel protocol."""

    def test_has_step_method(self):
        """Kernel has step method."""
        kernel = NaiveFlowKernel()
        assert hasattr(kernel, "step")
        assert callable(kernel.step)

    def test_has_fields_read(self):
        """Kernel has fields_read property."""
        kernel = NaiveFlowKernel()
        fields = kernel.fields_read
        assert isinstance(fields, set)
        assert "h" in fields
        assert "z" in fields
        assert "mask" in fields
        assert "flow_frac" in fields

    def test_has_fields_written(self):
        """Kernel has fields_written property."""
        kernel = NaiveFlowKernel()
        fields = kernel.fields_written
        assert isinstance(fields, set)
        assert "h" in fields
        assert "q_out" in fields

    def test_step_returns_routing_fluxes(
        self, state_fields, static_fields, scratch_fields, config, geometry
    ):
        """Kernel step returns RoutingFluxes."""
        # Initialize fields
        static_fields.initialize_tilted_plane(slope=0.01)

        # Compute flow directions first
        from src.kernels.naive.flow import compute_flow_directions, FLOW_EXPONENT

        compute_flow_directions(
            static_fields.z,
            static_fields.mask,
            static_fields.flow_frac,
            geometry.dx,
            FLOW_EXPONENT,
        )

        state_fields.h.from_numpy(np.full((16, 16), 0.01, dtype=np.float32))

        kernel = NaiveFlowKernel()
        result = kernel.step(
            state_fields, static_fields, scratch_fields, config, geometry.dx, 0.01
        )

        assert isinstance(result, RoutingFluxes)
        assert hasattr(result, "boundary_outflow")


class TestNaiveFlowDirectionKernelProtocol:
    """Test NaiveFlowDirectionKernel implements FlowDirectionKernel protocol."""

    def test_has_compute_method(self):
        """Kernel has compute method."""
        kernel = NaiveFlowDirectionKernel()
        assert hasattr(kernel, "compute")
        assert callable(kernel.compute)

    def test_has_fields_read(self):
        """Kernel has fields_read property."""
        kernel = NaiveFlowDirectionKernel()
        fields = kernel.fields_read
        assert isinstance(fields, set)
        assert "z" in fields
        assert "mask" in fields

    def test_has_fields_written(self):
        """Kernel has fields_written property."""
        kernel = NaiveFlowDirectionKernel()
        fields = kernel.fields_written
        assert isinstance(fields, set)
        assert "flow_frac" in fields

    def test_compute_sets_flow_fractions(self, static_fields, geometry):
        """Kernel compute method sets flow_frac values."""
        static_fields.initialize_tilted_plane(slope=0.01)

        kernel = NaiveFlowDirectionKernel()
        kernel.compute(static_fields, geometry.dx)

        # Check that flow_frac has been populated
        flow_frac_np = static_fields.flow_frac.to_numpy()
        # Interior cells should have non-zero flow fractions
        assert np.any(flow_frac_np[1:-1, 1:-1, :] != 0)


# --- Integration Tests ---


class TestRegistryIntegration:
    """Integration tests for kernel registry usage."""

    def test_full_workflow_with_registry(
        self, state_fields, static_fields, scratch_fields, config, geometry
    ):
        """Can run a complete step using registry-provided kernels."""
        registry = KernelRegistry()

        # Initialize
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.h.from_numpy(np.full((16, 16), 0.01, dtype=np.float32))
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        # Get kernels from registry
        flow_dir = registry.get_flow_direction()
        infiltration = registry.get_infiltration()
        soil = registry.get_soil()
        vegetation = registry.get_vegetation()
        flow = registry.get_flow()

        # Run flow direction computation
        flow_dir.compute(static_fields, geometry.dx)

        # Run infiltration
        inf_result = infiltration.step(state_fields, static_fields, config, 0.1)
        assert inf_result.total_infiltration >= 0

        # Run soil
        soil_result = soil.step(state_fields, static_fields, config, geometry.dx, 1.0)
        assert soil_result.total_et >= 0
        assert soil_result.total_leakage >= 0

        # Run vegetation
        veg_result = vegetation.step(
            state_fields, static_fields, config, geometry.dx, 1.0
        )
        assert veg_result.total_growth >= 0
        assert veg_result.total_mortality >= 0

        # Run flow routing
        flow_result = flow.step(
            state_fields, static_fields, scratch_fields, config, geometry.dx, 0.01
        )
        assert flow_result.boundary_outflow >= 0


class TestKernelWithDirectParams:
    """Test kernels work with direct parameter objects."""

    def test_soil_with_soil_params(
        self, state_fields, static_fields, geometry
    ):
        """Soil kernel works with SoilParams directly."""
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        params = SoilParams()
        kernel = NaiveSoilKernel()
        result = kernel.step(state_fields, static_fields, params, geometry.dx, 1.0)

        assert isinstance(result, SoilFluxes)

    def test_vegetation_with_vegetation_params(
        self, state_fields, static_fields, geometry
    ):
        """Vegetation kernel works with VegetationParams directly."""
        static_fields.initialize_tilted_plane(slope=0.01)
        state_fields.m.from_numpy(np.full((16, 16), 0.1, dtype=np.float32))
        state_fields.p.from_numpy(np.full((16, 16), 0.5, dtype=np.float32))

        params = VegetationParams()
        kernel = NaiveVegetationKernel()
        result = kernel.step(state_fields, static_fields, params, geometry.dx, 1.0)

        assert isinstance(result, VegetationFluxes)
