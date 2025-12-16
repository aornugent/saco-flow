"""Field management for SACO-Flow simulation.

This module provides declarative field containers for managing Taichi fields
with support for double-buffering, memory tracking, and typed specifications.

Main classes:
- FieldSpec: Declarative field specification
- FieldRole: Field categorization (STATE, STATIC, DERIVED, SCRATCH)
- FieldContainer: Manages Taichi field lifecycle

Convenience wrappers:
- StateFields: Access to h, m, p with swap operations
- StaticFields: Access to z, mask, flow_frac with initialization helpers
- ScratchFields: Access to temporary workspace fields

Factory functions:
- create_simulation_container: Full simulation field set
- create_state_specs, create_static_specs, etc.: Individual spec factories
"""

from src.fields.base import (
    FieldContainer,
    FieldRole,
    FieldSpec,
    create_all_specs,
    create_simulation_container,
)
from src.fields.scratch import (
    ScratchFields,
    create_derived_specs,
    create_scratch_container,
    create_scratch_specs,
)
from src.fields.state import (
    StateFields,
    create_state_container,
    create_state_specs,
)
from src.fields.static import (
    StaticFields,
    create_static_container,
    create_static_specs,
)

__all__ = [
    # Core classes
    "FieldContainer",
    "FieldRole",
    "FieldSpec",
    # Convenience wrappers
    "StateFields",
    "StaticFields",
    "ScratchFields",
    # Factory functions
    "create_simulation_container",
    "create_all_specs",
    "create_state_specs",
    "create_static_specs",
    "create_derived_specs",
    "create_scratch_specs",
    "create_state_container",
    "create_static_container",
    "create_scratch_container",
]
