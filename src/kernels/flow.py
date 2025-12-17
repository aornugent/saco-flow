"""
Flow direction, accumulation, and surface water routing kernels.

MFD (Multiple Flow Direction) distributes flow to downslope neighbors
proportional to slope^p. Routing uses kinematic wave with CFL limiting.

Note: Primary implementation is in naive/flow.py. This module re-exports
for backwards compatibility with existing code.
"""

# Re-export from naive implementation for backwards compatibility
from src.kernels.naive.flow import (
    FLOW_EXPONENT,
    MIN_SLOPE_SUM,
    MIN_DEPTH,
    compute_flow_directions,
    flow_accumulation_step,
    compute_flow_accumulation,
    compute_outflow,
    apply_fluxes,
    route_surface_water,
    compute_max_velocity,
    compute_cfl_timestep,
    NaiveFlowKernel,
    NaiveFlowDirectionKernel,
)

__all__ = [
    "FLOW_EXPONENT",
    "MIN_SLOPE_SUM",
    "MIN_DEPTH",
    "compute_flow_directions",
    "flow_accumulation_step",
    "compute_flow_accumulation",
    "compute_outflow",
    "apply_fluxes",
    "route_surface_water",
    "compute_max_velocity",
    "compute_cfl_timestep",
    "NaiveFlowKernel",
    "NaiveFlowDirectionKernel",
]
