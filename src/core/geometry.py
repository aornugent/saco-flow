"""Grid geometry and neighbor indexing for SACO-Flow.

This module centralizes all spatial indexing logic:
- GridGeometry: Immutable dataclass holding grid dimensions and cell size
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

from dataclasses import dataclass
from math import sqrt

import taichi as ti

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


@dataclass(frozen=True)
class GridGeometry:
    """Immutable grid geometry specification.

    Attributes:
        nx: Number of rows (i dimension)
        ny: Number of columns (j dimension)
        dx: Cell size in meters [m]

    The grid uses row-major indexing where:
        - i (row) increases southward (down)
        - j (column) increases eastward (right)
        - Origin (0,0) is at the northwest corner

    Properties:
        n_cells: Total number of cells (nx * ny)
        n_interior: Number of interior cells ((nx-2) * (ny-2))
        domain_width: Physical width in meters (ny * dx)
        domain_height: Physical height in meters (nx * dx)
    """

    nx: int
    ny: int
    dx: float = 1.0

    def __post_init__(self):
        """Validate grid dimensions."""
        if self.nx < 3:
            raise ValueError(f"nx must be >= 3, got {self.nx}")
        if self.ny < 3:
            raise ValueError(f"ny must be >= 3, got {self.ny}")
        if self.dx <= 0:
            raise ValueError(f"dx must be > 0, got {self.dx}")

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny

    @property
    def n_interior(self) -> int:
        """Number of interior (non-boundary) cells."""
        return (self.nx - 2) * (self.ny - 2)

    @property
    def domain_width(self) -> float:
        """Physical domain width in meters."""
        return self.ny * self.dx

    @property
    def domain_height(self) -> float:
        """Physical domain height in meters."""
        return self.nx * self.dx

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as (nx, ny) tuple."""
        return (self.nx, self.ny)

    def neighbor_distance_m(self, direction: int) -> float:
        """Get physical distance to neighbor in meters.

        Args:
            direction: Neighbor index (0-7)

        Returns:
            Distance in meters (dx for cardinal, dx*sqrt(2) for diagonal)
        """
        if direction < 0 or direction >= NUM_NEIGHBORS:
            raise ValueError(f"direction must be 0-7, got {direction}")
        dist_cells = SQRT2 if direction % 2 == 1 else 1.0
        return dist_cells * self.dx


# =============================================================================
# Taichi helper functions for use in kernels
# =============================================================================


@ti.func
def is_interior(i: int, j: int, nx: int, ny: int) -> bool:
    """Check if cell (i, j) is in the interior (not on boundary).

    Args:
        i: Row index
        j: Column index
        nx: Number of rows
        ny: Number of columns

    Returns:
        True if cell is not on any boundary edge
    """
    return 0 < i < nx - 1 and 0 < j < ny - 1


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
