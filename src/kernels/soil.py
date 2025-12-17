"""
Soil moisture dynamics: ET, deep leakage, and lateral diffusion.

dM/dt = -E(M,P) - L(M) + D_M·nabla²M

(Infiltration handled separately in infiltration.py)

Note: Primary implementation is in naive/soil.py. This module re-exports
for backwards compatibility with existing code.
"""

# Re-export from naive implementation for backwards compatibility
from src.kernels.naive.soil import (
    evapotranspiration_step,
    leakage_step,
    diffusion_step,
    soil_moisture_step,
    compute_diffusion_timestep,
    NaiveSoilKernel,
)

__all__ = [
    "evapotranspiration_step",
    "leakage_step",
    "diffusion_step",
    "soil_moisture_step",
    "compute_diffusion_timestep",
    "NaiveSoilKernel",
]
