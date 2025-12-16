"""Scratch field specifications and factory.

Scratch fields are temporary workspace for intermediate computations.
They are not preserved between timesteps and don't need double-buffering.

Current scratch fields:
- local_source: Local contribution for flow accumulation
- flow_acc: Flow accumulation (iteratively computed)
- q_out: Outflow rate from each cell [m/day]
"""

from typing import Any

import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import GridGeometry
from src.fields.base import FieldContainer, FieldRole, FieldSpec


def create_scratch_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for scratch (temporary) fields.

    Scratch fields are workspace for intermediate computations.
    These are reused each timestep and don't persist state.

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


def create_derived_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for derived (computed each step) fields.

    Derived fields are recomputed from state each timestep:
    - flow_acc: Flow accumulation for routing (iterative, needs buffer)
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


class ScratchFields:
    """Convenience wrapper for accessing scratch and derived fields.

    Provides typed access to temporary workspace fields used during
    flow routing and other multi-pass computations.

    Example:
        scratch = ScratchFields(container)
        local_src = scratch.local_source
        flow = scratch.flow_acc
    """

    def __init__(self, container: FieldContainer):
        """Initialize with field container.

        Args:
            container: Allocated FieldContainer with scratch fields
        """
        self._container = container

    @property
    def local_source(self) -> Any:
        """Local source term field for flow accumulation."""
        return self._container["local_source"]

    @property
    def flow_acc(self) -> Any:
        """Flow accumulation field."""
        return self._container["flow_acc"]

    @property
    def flow_acc_new(self) -> Any:
        """Flow accumulation buffer for iterative computation."""
        return self._container.get_buffer("flow_acc")

    @property
    def q_out(self) -> Any:
        """Outflow rate field [m/day]."""
        return self._container["q_out"]

    def swap_flow_acc(self) -> None:
        """Swap flow_acc with its buffer after iteration."""
        self._container.swap("flow_acc")


def create_scratch_container(geometry: GridGeometry) -> FieldContainer:
    """Create a container with scratch and derived fields.

    Useful for testing scratch field behavior in isolation.

    Args:
        geometry: Grid dimensions and cell size

    Returns:
        Allocated FieldContainer with scratch and derived fields
    """
    container = FieldContainer(geometry)
    container.register_many(create_scratch_specs())
    container.register_many(create_derived_specs())
    container.allocate()
    return container
