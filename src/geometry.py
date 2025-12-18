"""Grid geometry and neighbor indexing for SACO-Flow.

This module centralizes all spatial indexing logic:
- DTYPE: Taichi floating-point type (ti.f32)
- Neighbor vectors: 8-connectivity offsets for D8/MFD algorithms
- Helper functions: Boundary checks and neighbor coordinate computation

8-Connectivity Layout (clockwise from East):
    Index:  5  6  7
            4  X  0
            3  2  1

    Direction 0: East  (+j)
    Direction 1: SE    (+i, +j)
    Direction 2: South (+i)
    Direction 3: SW    (+i, -j)
    Direction 4: West  (-j)
    Direction 5: NW    (-i, -j)
    Direction 6: North (-i)
    Direction 7: NE    (-i, +j)
"""

from math import sqrt

import taichi as ti

# =============================================================================
# Data type
# =============================================================================

DTYPE = ti.f32

# =============================================================================
# Neighbor indexing (compile-time constants)
# =============================================================================

# Number of neighbors in 8-connectivity
NUM_NEIGHBORS: int = 8

# Diagonal distance factor
SQRT2: float = sqrt(2.0)

# 8-connectivity neighbor offsets
# Row offset (i): positive = South, negative = North
NEIGHBOR_DI = ti.Vector([0, 1, 1, 1, 0, -1, -1, -1])

# Column offset (j): positive = East, negative = West
NEIGHBOR_DJ = ti.Vector([1, 1, 0, -1, -1, -1, 0, 1])

# Distance to each neighbor (in cell units): 1.0 for cardinal, sqrt(2) for diagonal
NEIGHBOR_DIST = ti.Vector([1.0, SQRT2, 1.0, SQRT2, 1.0, SQRT2, 1.0, SQRT2])


# =============================================================================
# Taichi helper functions for use in kernels
# =============================================================================


@ti.func
def is_interior(i: int, j: int, n: int) -> bool:
    """Check if cell (i, j) is in the interior (not on boundary).

    Args:
        i: Row index
        j: Column index
        n: Grid size (assumes square grid)

    Returns:
        True if cell is not on any boundary edge
    """
    return 0 < i < n - 1 and 0 < j < n - 1


@ti.func
def get_neighbor(i: int, j: int, k: int) -> ti.Vector:
    """Get neighbor coordinates in direction k.

    Args:
        i: Current row index
        j: Current column index
        k: Neighbor direction (0-7)

    Returns:
        Vector [ni, nj] of neighbor coordinates
    """
    return ti.Vector([i + NEIGHBOR_DI[k], j + NEIGHBOR_DJ[k]])


@ti.func
def get_neighbor_distance(k: int, dx: ti.f32) -> ti.f32:
    """Get physical distance to neighbor k in meters.

    Args:
        k: Neighbor direction (0-7)
        dx: Cell size in meters

    Returns:
        Distance in meters
    """
    return NEIGHBOR_DIST[k] * dx
