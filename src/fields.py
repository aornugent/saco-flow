"""Taichi field allocation and buffer management."""

from dataclasses import dataclass

import taichi as ti

from src.config import DTYPE


@dataclass
class Fields:
    """Container for simulation fields on a square grid."""

    n: int
    M: ti.Field  # soil moisture [m]
    M_new: ti.Field  # soil moisture write buffer [m]
    mask: ti.Field  # active cell mask (1=active, 0=boundary)


def allocate(n: int) -> Fields:
    """Allocate Taichi fields for an n x n grid.

    Double-buffered: M (read) and M_new (write) for stencil operations.
    """
    M = ti.field(DTYPE, shape=(n, n))
    M_new = ti.field(DTYPE, shape=(n, n))
    mask = ti.field(ti.i32, shape=(n, n))
    return Fields(n=n, M=M, M_new=M_new, mask=mask)


@ti.kernel
def swap_buffers(src: ti.template(), dst: ti.template()):
    """Copy src into dst (used after stencil write to M_new)."""
    for i, j in src:
        dst[i, j] = src[i, j]
