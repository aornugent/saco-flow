"""
Utility kernels for SACO-Flow.

Provides basic operations used across the simulation:
- Field initialization
- Mass computation
- Field copying
"""

import taichi as ti

from src.config import DTYPE

# Neighbor offsets for 8-connectivity (clockwise from East)
# Index:  5  6  7
#         4  X  0
#         3  2  1
NEIGHBOR_DI = ti.Vector([0, 1, 1, 1, 0, -1, -1, -1])  # row offset
NEIGHBOR_DJ = ti.Vector([1, 1, 0, -1, -1, -1, 0, 1])  # col offset
NEIGHBOR_DIST = ti.Vector([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414])


@ti.kernel
def fill_field(field: ti.template(), value: DTYPE):
    """Fill entire field with a constant value."""
    for I in ti.grouped(field):
        field[I] = value


@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    """Copy src field to dst field."""
    for I in ti.grouped(src):
        dst[I] = src[I]


@ti.kernel
def compute_total(field: ti.template(), mask: ti.template()) -> DTYPE:
    """
    Compute sum of field values where mask == 1.

    Args:
        field: Field to sum
        mask: Binary mask (1 = include, 0 = exclude)

    Returns:
        Sum of masked field values
    """
    total = ti.cast(0.0, DTYPE)
    for I in ti.grouped(field):
        if mask[I] == 1:
            total += field[I]
    return total


@ti.kernel
def add_uniform(field: ti.template(), mask: ti.template(), value: DTYPE):
    """Add uniform value to field where mask == 1."""
    for I in ti.grouped(field):
        if mask[I] == 1:
            field[I] += value


@ti.kernel
def clamp_field(
    field: ti.template(),
    min_val: DTYPE,
    max_val: DTYPE,
):
    """Clamp field values to [min_val, max_val]."""
    for I in ti.grouped(field):
        field[I] = ti.max(min_val, ti.min(max_val, field[I]))


@ti.kernel
def multiply_field(field: ti.template(), factor: DTYPE):
    """Multiply all field values by factor."""
    for I in ti.grouped(field):
        field[I] *= factor


@ti.func
def is_interior(i: int, j: int, n: int) -> bool:
    """Check if (i, j) is an interior cell (not on boundary)."""
    return i > 0 and i < n - 1 and j > 0 and j < n - 1


@ti.func
def get_neighbor(i: int, j: int, k: int) -> ti.Vector:
    """
    Get neighbor coordinates in direction k.

    Args:
        i, j: Current cell indices
        k: Direction index (0-7, clockwise from East)

    Returns:
        Vector with (ni, nj) neighbor indices
    """
    ni = i + NEIGHBOR_DI[k]
    nj = j + NEIGHBOR_DJ[k]
    return ti.Vector([ni, nj])
