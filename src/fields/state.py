"""State field specifications and factory.

State fields represent the primary simulation variables that evolve over time:
- h: Surface water depth [m]
- M: Soil moisture [m water equivalent]
- P: Vegetation biomass [kg/m²]

All state fields are double-buffered to support stencil operations where
the new value depends on neighboring cells' current values.
"""

from typing import Any

import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import GridGeometry
from src.fields.base import FieldContainer, FieldRole, FieldSpec


def create_state_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for primary state fields.

    State fields are double-buffered for stencil operations:
    - h: Surface water depth [m]
    - m: Soil moisture [m water equivalent]
    - p: Vegetation biomass [kg/m²]

    Note: Field names use lowercase to match snake_case convention.
    The existing simulation uses uppercase (H, M, P) but we normalize
    to lowercase here. Compatibility adapters handle the mapping.

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


class StateFields:
    """Convenience wrapper for accessing state fields.

    Provides typed access to state fields and their buffers with
    explicit swap operations.

    Example:
        state = StateFields(container)
        h_field = state.h
        state.swap_h()  # After stencil operation
    """

    def __init__(self, container: FieldContainer):
        """Initialize with field container.

        Args:
            container: Allocated FieldContainer with state fields
        """
        self._container = container

    @property
    def h(self) -> Any:
        """Surface water depth field [m]."""
        return self._container["h"]

    @property
    def h_new(self) -> Any:
        """Surface water depth buffer [m]."""
        return self._container.get_buffer("h")

    @property
    def m(self) -> Any:
        """Soil moisture field [m]."""
        return self._container["m"]

    @property
    def m_new(self) -> Any:
        """Soil moisture buffer [m]."""
        return self._container.get_buffer("m")

    @property
    def p(self) -> Any:
        """Vegetation biomass field [kg/m²]."""
        return self._container["p"]

    @property
    def p_new(self) -> Any:
        """Vegetation biomass buffer [kg/m²]."""
        return self._container.get_buffer("p")

    def swap_h(self) -> None:
        """Swap h with its buffer after stencil operation."""
        self._container.swap("h")

    def swap_m(self) -> None:
        """Swap m with its buffer after stencil operation."""
        self._container.swap("m")

    def swap_p(self) -> None:
        """Swap p with its buffer after stencil operation."""
        self._container.swap("p")


def create_state_container(geometry: GridGeometry) -> FieldContainer:
    """Create a container with only state fields.

    Useful for testing state field behavior in isolation.

    Args:
        geometry: Grid dimensions and cell size

    Returns:
        Allocated FieldContainer with state fields only
    """
    container = FieldContainer(geometry)
    container.register_many(create_state_specs())
    container.allocate()
    return container
