"""Utility kernels and neighbor indexing for SACO-Flow."""

import taichi as ti

from src.config import DTYPE

# Re-export neighbor vectors from geometry for backwards compatibility
from src.core.geometry import NEIGHBOR_DI, NEIGHBOR_DJ, NEIGHBOR_DIST


@ti.kernel
def fill_field(field: ti.template(), value: DTYPE):
    """Set all field values to a constant."""
    for I in ti.grouped(field):
        field[I] = value


@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    """Copy src to dst."""
    for I in ti.grouped(src):
        dst[I] = src[I]


@ti.kernel
def compute_total(field: ti.template(), mask: ti.template()) -> DTYPE:
    """Sum field values where mask == 1."""
    total = ti.cast(0.0, DTYPE)
    for I in ti.grouped(field):
        if mask[I] == 1:
            total += field[I]
    return total


@ti.kernel
def add_uniform(field: ti.template(), mask: ti.template(), value: DTYPE):
    """Add value to field where mask == 1."""
    for I in ti.grouped(field):
        if mask[I] == 1:
            field[I] += value


@ti.kernel
def clamp_field(field: ti.template(), min_val: DTYPE, max_val: DTYPE):
    """Clamp field values to [min_val, max_val]."""
    for I in ti.grouped(field):
        field[I] = ti.max(min_val, ti.min(max_val, field[I]))


@ti.func
def is_interior(i: int, j: int, n: int) -> bool:
    """Check if (i, j) is not on the boundary."""
    return 0 < i < n - 1 and 0 < j < n - 1


@ti.func
def get_neighbor(i: int, j: int, k: int) -> ti.Vector:
    """Get neighbor (ni, nj) in direction k (0-7)."""
    return ti.Vector([i + NEIGHBOR_DI[k], j + NEIGHBOR_DJ[k]])
