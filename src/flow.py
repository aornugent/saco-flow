"""
Flow routing kernels — MFD flow fractions and gather-based water routing.

Multiple-flow direction (MFD) algorithm distributes outflow proportionally
to downslope gradients raised to exponent p.  Water routing gathers incoming
flow from upslope neighbors using previous-timestep Q_out (explicit).
"""

import taichi as ti

# 8-connected neighbor offsets and diagonal distances
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIAG = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]
# Opposite direction index: neighbor k's outflow toward (i,j) is stored at 7-k
_OPP = [7, 6, 5, 4, 3, 2, 1, 0]


@ti.kernel
def compute_flow_fractions(
    z: ti.template(),
    mask: ti.template(),
    flow_frac: ti.template(),
    dx: ti.f32,
    p: ti.f32,
):
    """Compute MFD flow fractions from each cell to its 8 neighbors.

    F_k = max(0, slope_k)^p / sum(max(0, slope_k)^p)

    Args:
        z: Elevation field [m]
        mask: Active cell mask
        flow_frac: Output fractions (n, n, 8)
        dx: Cell spacing [m]
        p: Convergence exponent (1.0=MFD, large=D8-like)
    """
    n = z.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            for k in ti.static(range(8)):
                flow_frac[i, j, k] = 0.0
            continue

        # First pass: compute slope^p and store in flow_frac, accumulate total
        total = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(_OFFSETS[k])
            dist = ti.static(_DIAG[k]) * dx
            ni, nj = i + di, j + dj
            s_pow = ti.cast(0.0, ti.f32)
            if mask[ni, nj] == 1:
                slope = (z[i, j] - z[ni, nj]) / dist
                if slope > 0.0:
                    s_pow = ti.pow(slope, p)
            flow_frac[i, j, k] = s_pow
            total += s_pow

        # Second pass: normalize to fractions
        for k in ti.static(range(8)):
            if total > 0.0:
                flow_frac[i, j, k] /= total
            else:
                flow_frac[i, j, k] = 0.0


@ti.kernel
def route_water(
    Q_out: ti.template(),
    Q_out_new: ti.template(),
    R: ti.template(),
    I_inf: ti.template(),
    h: ti.template(),
    z: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    n_manning: ti.f32,
    cn: ti.f32,
):
    """Gather-based water routing: compute Q, h, and Q_out per cell.

    1. Gather Q_in from upslope neighbors (previous-timestep Q_out)
    2. Q = Q_in + R * dx^2
    3. Q_out = max(0, Q - I_inf * dx^2)
    4. h from Manning's kinematic wave approximation

    Args:
        Q_out: Previous-timestep outgoing discharge (read) [m^3/day]
        Q_out_new: Updated outgoing discharge (write) [m^3/day]
        R: Rainfall rate [m/day]
        I_inf: Infiltration rate [m/day]
        h: Flow depth (write) [m]
        z: Elevation field [m]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
        n_manning: Manning's roughness coefficient [s/m^(1/3)]
        cn: Constant for kinematic wave [m^(1/3)/s]
    """
    n = Q_out.shape[0]
    cell_area = dx * dx

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            Q_out_new[i, j] = 0.0
            h[i, j] = 0.0
            continue

        # Gather incoming flow from upslope neighbors
        Q_in = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(_OFFSETS[k])
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                opp = ti.static(_OPP[k])
                Q_in += flow_frac[ni, nj, opp] * Q_out[ni, nj]

        Q_total = Q_in + R[i, j] * cell_area
        Q_new = ti.max(0.0, Q_total - I_inf[i, j] * cell_area)
        Q_out_new[i, j] = Q_new

        # Max downslope gradient for Manning's equation
        slope_max = ti.cast(1e-4, ti.f32)
        for k in ti.static(range(8)):
            if flow_frac[i, j, k] > 0.0:
                di, dj = ti.static(_OFFSETS[k])
                dist = ti.static(_DIAG[k]) * dx
                ni, nj = i + di, j + dj
                slope_k = (z[i, j] - z[ni, nj]) / dist
                slope_max = ti.max(slope_max, slope_k)

        # h = (Q_daily * n_manning / (cn * sqrt(slope)))^(3/5)
        if Q_new > 0.0 and cn > 0.0:
            h[i, j] = ti.pow(
                Q_new * n_manning / (cn * ti.sqrt(slope_max)),
                0.6,
            )
        else:
            h[i, j] = 0.0
