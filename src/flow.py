"""
Flow routing kernels — MFD flow fractions and gather-based water routing.

Multiple-flow direction (MFD) algorithm distributes outflow proportionally
to downslope gradients raised to exponent p.  Water routing gathers incoming
flow from upslope neighbors using previous-timestep Q_out (explicit).
Picard iteration resolves the q-h-I-Q_out coupling within each cell.
"""

import taichi as ti

from src.stencil import DIAG, OFFSETS, OPP


@ti.kernel
def accumulate_annual_Q(
    Q_annual: ti.template(),
    Q_daily: ti.template(),
    mask: ti.template(),
):
    """Add daily cell-average discharge into annual accumulator.

    Args:
        Q_annual: Running annual total (read/write) [m^3/yr]
        Q_daily: Daily cell-average discharge [m^3/day]
        mask: Active cell mask
    """
    n = Q_annual.shape[0]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 1:
            Q_annual[i, j] += Q_daily[i, j]


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

        total = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(OFFSETS[k])
            dist = ti.static(DIAG[k]) * dx
            ni, nj = i + di, j + dj
            s_pow = ti.cast(0.0, ti.f32)
            if mask[ni, nj] == 1:
                slope = (z[i, j] - z[ni, nj]) / dist
                if slope > 0.0:
                    s_pow = ti.pow(slope, p)
            flow_frac[i, j, k] = s_pow
            total += s_pow

        for k in ti.static(range(8)):
            if total > 0.0:
                flow_frac[i, j, k] /= total
            else:
                flow_frac[i, j, k] = 0.0


@ti.kernel
def route_water(
    Q_out: ti.template(),
    Q_out_new: ti.template(),
    Q_daily: ti.template(),
    R: ti.template(),
    I_inf: ti.template(),
    h: ti.template(),
    z: ti.template(),
    V: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    n_manning: ti.f32,
    cn: ti.f32,
    alpha: ti.f32,
    k2: ti.f32,
    W0: ti.f32,
):
    """Gather-based water routing with Picard iteration for q-h-I-Q_out.

    1. Gather Q_in from upslope neighbors (previous-timestep Q_out)
    2. Picard: q = (Q_in + Q_out) / (2*dx) [m^2/day]
    3. h = (q * n_manning / (cn * sqrt(slope)))^(3/5)
    4. I = alpha * h * (V + k2*W0) / (V + k2)
    5. Q_out = max(0, Q_in + R*dx^2 - I*dx^2)
    6. Q_daily = (Q_in + Q_out) / 2  (Eq 12)
    7. Writes Q_out_new, Q_daily, h, I_inf

    Args:
        Q_out: Previous-timestep outgoing discharge (read) [m^3/day]
        Q_out_new: Updated outgoing discharge (write) [m^3/day]
        Q_daily: Cell-average discharge (write) [m^3/day]
        R: Rainfall rate [m/day]
        I_inf: Infiltration rate (write) [m/day]
        h: Flow depth (write) [m]
        z: Elevation field [m]
        V: Vegetation density [%]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
        n_manning: Manning's roughness coefficient [s/m^(1/3)]
        cn: Constant for kinematic wave [m^(1/3)/s]
        alpha: Infiltration capacity [1/day]
        k2: Vegetation half-saturation for infiltration [%]
        W0: Bare-soil infiltration fraction [-]
    """
    n = Q_out.shape[0]
    cell_area = dx * dx

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            Q_out_new[i, j] = 0.0
            Q_daily[i, j] = 0.0
            h[i, j] = 0.0
            I_inf[i, j] = 0.0
            continue

        # Gather incoming flow from upslope neighbors
        Q_in = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(OFFSETS[k])
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                opp = ti.static(OPP[k])
                Q_in += flow_frac[ni, nj, opp] * Q_out[ni, nj]

        # Max downslope gradient (Lambda_max, floored at 1e-4)
        slope_max = ti.cast(1e-4, ti.f32)
        for k in ti.static(range(8)):
            if flow_frac[i, j, k] > 0.0:
                di, dj = ti.static(OFFSETS[k])
                dist = ti.static(DIAG[k]) * dx
                ni, nj = i + di, j + dj
                slope_k = (z[i, j] - z[ni, nj]) / dist
                slope_max = ti.max(slope_max, slope_k)

        # Picard iteration: q -> h -> I -> Q_out (5 local iterations)
        v = V[i, j]
        Q_o = Q_out[i, j]  # initial guess from previous global pass
        h_val = ti.cast(0.0, ti.f32)
        I_val = ti.cast(0.0, ti.f32)

        for _ in ti.static(range(5)):
            # §2: q = (Q_in + Q_out) / (2*dx)  [m^2/day]
            q = (Q_in + Q_o) / (2.0 * dx)
            # h = (q * n_manning / (cn * sqrt(slope_max)))^(3/5)
            if q > 0.0 and cn > 0.0:
                h_val = ti.pow(
                    q * n_manning / (cn * ti.sqrt(slope_max)),
                    0.6,
                )
            else:
                h_val = 0.0
            # I = alpha * h * (V + k2*W0) / (V + k2)
            I_val = alpha * h_val * (v + k2 * W0) / (v + k2)
            # Q_out = max(0, Q_in + R*dx^2 - I*dx^2)
            Q_o = ti.max(0.0, Q_in + R[i, j] * cell_area - I_val * cell_area)

        # Final consistent recompute from converged Q_o
        q = (Q_in + Q_o) / (2.0 * dx)
        if q > 0.0 and cn > 0.0:
            h_val = ti.pow(
                q * n_manning / (cn * ti.sqrt(slope_max)),
                0.6,
            )
        else:
            h_val = 0.0
        I_val = alpha * h_val * (v + k2 * W0) / (v + k2)

        Q_out_new[i, j] = Q_o
        Q_daily[i, j] = (Q_in + Q_o) / 2.0  # Eq 12
        h[i, j] = h_val
        I_inf[i, j] = I_val
