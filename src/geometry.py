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

DTYPE = ti.f64

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


@ti.kernel
def laplacian_diffusion_step(
    field: ti.template(),
    field_new: ti.template(),
    mask: ti.template(),
    D: DTYPE,
    dx: DTYPE,
    dt: DTYPE,
):
    """
    Apply diffusion using 5-point Laplacian stencil.

    ∂φ/∂t = D·∇²φ

    Uses Neumann (no-flux) boundary conditions: only includes neighbors where mask=1.
    Double buffered: reads from field, writes to field_new.

    Uses ti.block_local() to cache stencil reads in shared memory on GPU,
    reducing global memory traffic for neighbor accesses.

    Args:
        field: Source field (read)
        field_new: Destination field (write)
        mask: Active cell mask (1=active, 0=inactive)
        D: Diffusion coefficient [m²/day]
        dx: Cell size [m]
        dt: Timestep [days]
    """
    # Cache field in shared memory for stencil reads (GPU optimization)
    ti.block_local(field)

    n = field.shape[0]
    coeff = D * dt / (dx * dx)

    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            field_new[i, j] = field[i, j]
            continue

        local_val = field[i, j]

        # 5-point Laplacian with Neumann BC
        laplacian = ti.cast(0.0, DTYPE)
        for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += field[ni, nj] - local_val

        # Apply diffusion with non-negativity constraint
        d_val = coeff * laplacian
        field_new[i, j] = ti.max(0.0, local_val + d_val)
