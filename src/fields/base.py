"""Base field container and specification classes.

This module provides the foundation for declarative field management:
- FieldSpec: Describes a field's name, dtype, shape, and role
- FieldRole: Enum categorizing field usage patterns
- FieldContainer: Manages field lifecycle, allocation, and double-buffering

Usage:
    container = FieldContainer(geometry)
    container.register(FieldSpec("h", DTYPE, FieldRole.STATE, double_buffer=True))
    container.allocate()
    # Access fields
    h = container.get("h")
    h_new = container.get_buffer("h")
    container.swap("h")  # Swap h <-> h_new
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import GridGeometry, NUM_NEIGHBORS


class FieldRole(Enum):
    """Categorizes field usage patterns for documentation and validation.

    STATE: Primary simulation state (h, M, P) - typically double-buffered
    STATIC: Read-only during simulation (Z, mask, flow_frac)
    DERIVED: Computed from state each step (flow_acc)
    SCRATCH: Temporary workspace for intermediate computations
    """

    STATE = auto()
    STATIC = auto()
    DERIVED = auto()
    SCRATCH = auto()


@dataclass(frozen=True)
class FieldSpec:
    """Immutable specification for a Taichi field.

    Attributes:
        name: Field identifier (snake_case)
        dtype: Taichi data type (ti.f32, ti.i8, etc.)
        role: Field usage category
        double_buffer: If True, allocate a companion buffer for stencil ops
        extra_dims: Additional dimensions beyond (nx, ny), e.g., (8,) for flow_frac
        description: Human-readable description with units

    The field shape is (nx, ny) + extra_dims, where (nx, ny) comes from
    the GridGeometry passed to the FieldContainer.
    """

    name: str
    dtype: Any  # Taichi dtype
    role: FieldRole
    double_buffer: bool = False
    extra_dims: tuple[int, ...] = ()
    description: str = ""

    def __post_init__(self):
        """Validate field specification."""
        if not self.name:
            raise ValueError("Field name cannot be empty")
        if not self.name.islower() or not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Field name must be snake_case, got: {self.name}"
            )
        if self.double_buffer and self.role == FieldRole.STATIC:
            raise ValueError(
                f"Static field '{self.name}' cannot be double-buffered"
            )

    def buffer_name(self) -> str:
        """Get the name for this field's double buffer."""
        return f"{self.name}_new"


class FieldContainer:
    """Manages Taichi field lifecycle with declarative specifications.

    A FieldContainer holds a collection of Taichi fields associated with
    a specific grid geometry. Fields are registered via FieldSpec, then
    allocated together. Double-buffered fields support efficient stencil
    operations via swap().

    Attributes:
        geometry: Grid dimensions and cell size
        specs: Registered field specifications (name -> FieldSpec)
        fields: Allocated Taichi fields (name -> field)
        allocated: Whether fields have been allocated

    Example:
        container = FieldContainer(GridGeometry(100, 100, dx=1.0))
        container.register(FieldSpec("h", DTYPE, FieldRole.STATE, double_buffer=True))
        container.register(FieldSpec("mask", ti.i8, FieldRole.STATIC))
        container.allocate()

        h = container["h"]
        container.swap("h")  # h <-> h_new
    """

    def __init__(self, geometry: GridGeometry):
        """Initialize container with grid geometry.

        Args:
            geometry: Grid dimensions and cell size
        """
        self._geometry = geometry
        self._specs: dict[str, FieldSpec] = {}
        self._fields: dict[str, Any] = {}
        self._allocated = False

    @property
    def geometry(self) -> GridGeometry:
        """Get the grid geometry."""
        return self._geometry

    @property
    def allocated(self) -> bool:
        """Check if fields have been allocated."""
        return self._allocated

    @property
    def field_names(self) -> list[str]:
        """Get list of registered field names (excluding buffers)."""
        return list(self._specs.keys())

    def register(self, spec: FieldSpec) -> None:
        """Register a field specification.

        Args:
            spec: Field specification to register

        Raises:
            ValueError: If name already registered or fields already allocated
        """
        if self._allocated:
            raise RuntimeError("Cannot register fields after allocation")
        if spec.name in self._specs:
            raise ValueError(f"Field '{spec.name}' already registered")
        if spec.double_buffer and spec.buffer_name() in self._specs:
            raise ValueError(
                f"Buffer name '{spec.buffer_name()}' conflicts with existing field"
            )
        self._specs[spec.name] = spec

    def register_many(self, specs: list[FieldSpec]) -> None:
        """Register multiple field specifications.

        Args:
            specs: List of field specifications to register
        """
        for spec in specs:
            self.register(spec)

    def allocate(self) -> None:
        """Allocate all registered fields.

        Creates Taichi fields according to specifications. Double-buffered
        fields get a companion field with suffix "_new".

        Raises:
            RuntimeError: If already allocated or no fields registered
        """
        if self._allocated:
            raise RuntimeError("Fields already allocated")
        if not self._specs:
            raise RuntimeError("No fields registered")

        nx, ny = self._geometry.nx, self._geometry.ny

        for name, spec in self._specs.items():
            shape = (nx, ny) + spec.extra_dims
            self._fields[name] = ti.field(dtype=spec.dtype, shape=shape)

            if spec.double_buffer:
                self._fields[spec.buffer_name()] = ti.field(
                    dtype=spec.dtype, shape=shape
                )

        self._allocated = True

    def get(self, name: str) -> Any:
        """Get a field by name.

        Args:
            name: Field name

        Returns:
            The Taichi field

        Raises:
            KeyError: If field not found
            RuntimeError: If fields not allocated
        """
        if not self._allocated:
            raise RuntimeError("Fields not yet allocated")
        if name not in self._fields:
            raise KeyError(f"Field '{name}' not found")
        return self._fields[name]

    def __getitem__(self, name: str) -> Any:
        """Get a field by name using bracket notation."""
        return self.get(name)

    def get_buffer(self, name: str) -> Any:
        """Get the double buffer for a field.

        Args:
            name: Primary field name

        Returns:
            The buffer field (name_new)

        Raises:
            ValueError: If field is not double-buffered
        """
        if name not in self._specs:
            raise KeyError(f"Field '{name}' not registered")
        spec = self._specs[name]
        if not spec.double_buffer:
            raise ValueError(f"Field '{name}' is not double-buffered")
        return self.get(spec.buffer_name())

    def swap(self, name: str) -> None:
        """Swap a field with its double buffer.

        After swap, the primary field contains what was in the buffer,
        and vice versa. This is O(1) - just pointer swaps.

        Args:
            name: Field name to swap

        Raises:
            ValueError: If field is not double-buffered
        """
        if name not in self._specs:
            raise KeyError(f"Field '{name}' not registered")
        spec = self._specs[name]
        if not spec.double_buffer:
            raise ValueError(f"Field '{name}' is not double-buffered")

        buffer_name = spec.buffer_name()
        # Swap the field references
        self._fields[name], self._fields[buffer_name] = (
            self._fields[buffer_name],
            self._fields[name],
        )

    def get_spec(self, name: str) -> FieldSpec:
        """Get the specification for a field.

        Args:
            name: Field name

        Returns:
            The FieldSpec for this field
        """
        if name not in self._specs:
            raise KeyError(f"Field '{name}' not registered")
        return self._specs[name]

    def fields_by_role(self, role: FieldRole) -> list[str]:
        """Get field names filtered by role.

        Args:
            role: FieldRole to filter by

        Returns:
            List of field names with the specified role
        """
        return [name for name, spec in self._specs.items() if spec.role == role]

    @property
    def memory_bytes(self) -> int:
        """Estimate total memory usage in bytes.

        Returns:
            Approximate memory usage for all allocated fields
        """
        if not self._allocated:
            return 0

        total = 0
        nx, ny = self._geometry.nx, self._geometry.ny

        for name, spec in self._specs.items():
            # Base shape
            shape = (nx, ny) + spec.extra_dims
            n_elements = 1
            for dim in shape:
                n_elements *= dim

            # Dtype size (approximate)
            dtype_size = 4  # Default to f32/i32
            if spec.dtype == ti.f64 or spec.dtype == ti.i64:
                dtype_size = 8
            elif spec.dtype == ti.i8:
                dtype_size = 1
            elif spec.dtype == ti.i16:
                dtype_size = 2

            field_bytes = n_elements * dtype_size
            total += field_bytes

            # Double buffer
            if spec.double_buffer:
                total += field_bytes

        return total

    @property
    def memory_mb(self) -> float:
        """Estimate total memory usage in megabytes."""
        return self.memory_bytes / (1024 * 1024)

    def __contains__(self, name: str) -> bool:
        """Check if a field is registered."""
        return name in self._specs

    def __len__(self) -> int:
        """Number of registered fields (not counting buffers)."""
        return len(self._specs)


# =============================================================================
# Standard field specification factories
# =============================================================================


def create_state_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for primary state fields.

    State fields are double-buffered for stencil operations:
    - h: Surface water depth [m]
    - M: Soil moisture [m water equivalent]
    - P: Vegetation biomass [kg/m²]

    Args:
        dtype: Floating-point type (default: DTYPE from dtypes.py)

    Returns:
        List of FieldSpec for state fields
    """
    return [
        FieldSpec(
            name="h",
            dtype=dtype,
            role=FieldRole.STATE,
            double_buffer=True,
            description="Surface water depth [m]",
        ),
        FieldSpec(
            name="m",
            dtype=dtype,
            role=FieldRole.STATE,
            double_buffer=True,
            description="Soil moisture [m water equivalent]",
        ),
        FieldSpec(
            name="p",
            dtype=dtype,
            role=FieldRole.STATE,
            double_buffer=True,
            description="Vegetation biomass [kg/m²]",
        ),
    ]


def create_static_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for static (read-only) fields.

    Static fields are set during initialization and don't change:
    - z: Terrain elevation [m]
    - mask: Domain mask (1=interior, 0=boundary/outside)
    - flow_frac: MFD flow fractions to 8 neighbors [-]

    Args:
        dtype: Floating-point type for z and flow_frac

    Returns:
        List of FieldSpec for static fields
    """
    return [
        FieldSpec(
            name="z",
            dtype=dtype,
            role=FieldRole.STATIC,
            description="Terrain elevation [m]",
        ),
        FieldSpec(
            name="mask",
            dtype=ti.i8,
            role=FieldRole.STATIC,
            description="Domain mask (1=interior, 0=boundary)",
        ),
        FieldSpec(
            name="flow_frac",
            dtype=dtype,
            role=FieldRole.STATIC,
            extra_dims=(NUM_NEIGHBORS,),
            description="MFD flow fractions to 8 neighbors [-]",
        ),
    ]


def create_derived_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for derived (computed) fields.

    Derived fields are recomputed each timestep:
    - flow_acc: Flow accumulation for routing
    - q_out: Outflow rate from each cell [m/day]

    Args:
        dtype: Floating-point type

    Returns:
        List of FieldSpec for derived fields
    """
    return [
        FieldSpec(
            name="flow_acc",
            dtype=dtype,
            role=FieldRole.DERIVED,
            double_buffer=True,  # Iterative accumulation needs buffer
            description="Flow accumulation [cells or m³]",
        ),
        FieldSpec(
            name="q_out",
            dtype=dtype,
            role=FieldRole.DERIVED,
            description="Outflow rate [m/day]",
        ),
    ]


def create_scratch_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for scratch (temporary) fields.

    Scratch fields are workspace for intermediate computations.
    Add more as needed for specific algorithms.

    Args:
        dtype: Floating-point type

    Returns:
        List of FieldSpec for scratch fields
    """
    return [
        FieldSpec(
            name="local_source",
            dtype=dtype,
            role=FieldRole.SCRATCH,
            description="Local source term for flow accumulation",
        ),
    ]


def create_all_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create all standard field specifications.

    Combines state, static, derived, and scratch field specs.

    Args:
        dtype: Floating-point type

    Returns:
        Complete list of FieldSpec for simulation
    """
    return (
        create_state_specs(dtype)
        + create_static_specs(dtype)
        + create_derived_specs(dtype)
        + create_scratch_specs(dtype)
    )


def create_simulation_container(geometry: GridGeometry) -> FieldContainer:
    """Create a fully-configured field container for simulation.

    This is a convenience function that creates a FieldContainer with
    all standard fields registered and allocated.

    Args:
        geometry: Grid dimensions and cell size

    Returns:
        Allocated FieldContainer ready for simulation
    """
    container = FieldContainer(geometry)
    container.register_many(create_all_specs())
    container.allocate()
    return container
