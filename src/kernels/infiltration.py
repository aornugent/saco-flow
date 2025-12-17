"""
Infiltration kernel: surface water to soil moisture transfer.

Implements vegetation-enhanced infiltration following Saco et al. (2013):
I = alpha · h · [(P + k_P·W_0)/(P + k_P)] · (1 - M/M_sat)⁺

Note: Primary implementation is in naive/infiltration.py. This module re-exports
for backwards compatibility with existing code.
"""

# Re-export from naive implementation for backwards compatibility
from src.kernels.naive.infiltration import (
    infiltration_step,
    NaiveInfiltrationKernel,
)

__all__ = [
    "infiltration_step",
    "NaiveInfiltrationKernel",
]
