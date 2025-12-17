"""
Vegetation dynamics: growth, mortality, and seed dispersal.

dP/dt = G(M)·P - mu·P + D_P·nabla²P

Where:
- G(M) = g_max · M/(M + k_G)  -- Monod growth kinetics
- mu·P                        -- constant mortality
- D_P·nabla²P                 -- seed dispersal diffusion

Note: Primary implementation is in naive/vegetation.py. This module re-exports
for backwards compatibility with existing code.
"""

# Re-export from naive implementation for backwards compatibility
from src.kernels.naive.vegetation import (
    growth_step,
    mortality_step,
    vegetation_diffusion_step,
    vegetation_step,
    compute_equilibrium_moisture,
    compute_vegetation_timestep,
    NaiveVegetationKernel,
)

__all__ = [
    "growth_step",
    "mortality_step",
    "vegetation_diffusion_step",
    "vegetation_step",
    "compute_equilibrium_moisture",
    "compute_vegetation_timestep",
    "NaiveVegetationKernel",
]
