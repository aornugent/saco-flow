"""
Vegetation dynamics kernel.

Growth, mortality, isotropic seed dispersal (diffusive Laplacian),
and flow-directed seed dispersal (gather from upslope).
"""

import taichi as ti

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_OPP = [7, 6, 5, 4, 3, 2, 1, 0]

# 4-connected neighbors for isotropic Laplacian
_CARD = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@ti.kernel
def vegetation_step(
    V: ti.template(),
    V_new: ti.template(),
    M: ti.template(),
    Q_out: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    c: ti.f32,
    g_max: ti.f32,
    k1: ti.f32,
    d: ti.f32,
    Dp: ti.f32,
    dx: ti.f32,
    c1: ti.f32,
    c2: ti.f32,
    dt: ti.f32,
):
    """Update vegetation density: growth + mortality + seed dispersal.

    V_new = V + dt * (G - D + D_iso + D_flow)

    Args:
        V: Vegetation density read [%]
        V_new: Vegetation density write [%]
        M: Soil moisture [m]
        Q_out: Outgoing water discharge [m^3/day]
        flow_frac: MFD flow fractions (n, n, 8)
        mask: Active cell mask
        c: Growth scaling factor [-]
        g_max: Maximum growth rate [1/day]
        k1: Half-saturation for moisture [m]
        d: Mortality rate [1/day]
        Dp: Isotropic seed diffusion coefficient [m^2/day]
        dx: Cell spacing [m]
        c1: Flow dispersal coefficient [1/m^2]
        c2: Flow dispersal saturation [1/day]
        dt: Timestep [days]
    """
    n = V.shape[0]
    coeff_iso = Dp * dt / (dx * dx)

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            V_new[i, j] = V[i, j]
            continue

        v = V[i, j]
        m = M[i, j]

        # Growth: G = c * g_max * M/(M+k1) * V
        growth = c * g_max * m / (m + k1) * v

        # Mortality: D = d * V
        mortality = d * v

        # Isotropic seed dispersal: D_iso = Dp * laplacian(V)
        laplacian = ti.cast(0.0, ti.f32)
        for di, dj in ti.static(_CARD):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += V[ni, nj] - v

        # Flow-directed seed dispersal: gather from upslope
        d_flow = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(_OFFSETS[k])
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                opp = ti.static(_OPP[k])
                frac = flow_frac[ni, nj, opp]
                if frac > 0.0:
                    q_seed_full = c1 * Q_out[ni, nj] * V[ni, nj]
                    q_seed_cap = c2 * V[ni, nj]
                    q_seed = ti.min(q_seed_full, q_seed_cap)
                    d_flow += frac * q_seed

        V_new[i, j] = ti.max(
            0.0,
            v + dt * (growth - mortality + d_flow) + coeff_iso * laplacian,
        )
