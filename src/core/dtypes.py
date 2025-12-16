"""Type definitions for SACO-Flow simulation.

This module defines the floating-point precision used throughout the simulation.
Using ti.f32 (single precision) provides good performance on GPU while maintaining
sufficient accuracy for ecohydrological modeling.
"""

import taichi as ti

# Default floating-point type for all fields and computations
# ti.f32: Single precision (32-bit float) - faster on GPU, ~7 significant digits
# ti.f64: Double precision (64-bit float) - more accurate, slower on GPU
DTYPE = ti.f32
