"""
Naive (reference) kernel implementations.

These kernels prioritize correctness and readability over performance.
They serve as the baseline for equivalence testing of optimized variants.

Wrapper classes implement the kernel protocols from protocol.py.
"""

from src.kernels.naive.soil import NaiveSoilKernel
from src.kernels.naive.vegetation import NaiveVegetationKernel
from src.kernels.naive.infiltration import NaiveInfiltrationKernel
from src.kernels.naive.flow import NaiveFlowKernel, NaiveFlowDirectionKernel

__all__ = [
    "NaiveSoilKernel",
    "NaiveVegetationKernel",
    "NaiveInfiltrationKernel",
    "NaiveFlowKernel",
    "NaiveFlowDirectionKernel",
]
