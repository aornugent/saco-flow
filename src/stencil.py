"""Shared stencil constants and grid traversal utilities for 8-connected operations."""

import taichi as ti

# 8-connected neighbor offsets: NW, N, NE, W, E, SW, S, SE
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Distance multiplier: sqrt(2) for diagonal, 1 for cardinal
DIAG = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

# Opposite direction index: neighbor k's outflow toward (i,j) is at OPP[k]
OPP = [7, 6, 5, 4, 3, 2, 1, 0]

# 4-connected cardinal neighbors for Laplacian stencils
CARD = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@ti.func
def gather_flux(
    field: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    i: ti.i32,
    j: ti.i32,
) -> ti.f32:
    """Sum field values arriving from 8 upslope neighbors weighted by flow fractions."""
    result = ti.cast(0.0, ti.f32)
    for k in ti.static(range(8)):
        di, dj = ti.static(OFFSETS[k])
        ni, nj = i + di, j + dj
        if mask[ni, nj] == 1:
            opp = ti.static(OPP[k])
            result += flow_frac[ni, nj, opp] * field[ni, nj]
    return result


@ti.func
def max_downslope(
    z: ti.template(),
    flow_frac: ti.template(),
    i: ti.i32,
    j: ti.i32,
    dx: ti.f32,
) -> ti.f32:
    """Maximum downslope gradient over active outflow directions, floored at 1e-4."""
    slope_max = ti.cast(1e-4, ti.f32)
    for k in ti.static(range(8)):
        if flow_frac[i, j, k] > 0.0:
            di, dj = ti.static(OFFSETS[k])
            dist = ti.static(DIAG[k]) * dx
            ni, nj = i + di, j + dj
            slope_k = (z[i, j] - z[ni, nj]) / dist
            slope_max = ti.max(slope_max, slope_k)
    return slope_max
