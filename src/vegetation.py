"""
Vegetation dynamics kernel.

Growth, mortality, isotropic seed dispersal (diffusive Laplacian),
and flow-directed seed dispersal (gather from upslope minus outgoing).
"""

import taichi as ti

from src.stencil import CARD, OFFSETS, OPP


@ti.kernel
def vegetation_step(
    V: ti.template(),
    V_new: ti.template(),
    M: ti.template(),
    Q_daily: ti.template(),
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

    D_flow = seed_in - seed_out  (net flow-directed dispersal)
    Q_seed = min(c1 * q * V, c2 * V)

    Unit discharge q is converted to [mm*m/day] to match the paper,
    so Table I/II parameter values can be used directly.

    Args:
        V: Vegetation density read [%]
        V_new: Vegetation density write [%]
        M: Soil moisture [m]
        Q_daily: Cell-average discharge (Q_in+Q_out)/2 [m^3/day]
        flow_frac: MFD flow fractions (n, n, 8)
        mask: Active cell mask
        c: Growth scaling factor [-]
        g_max: Maximum growth rate [1/day]
        k1: Half-saturation for moisture [m]
        d: Mortality rate [1/day]
        Dp: Isotropic seed diffusion coefficient [m^2/day]
        dx: Cell spacing [m]
        c1: Flow dispersal coefficient [day/(mm*m)]
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
        for di, dj in ti.static(CARD):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += V[ni, nj] - v

        # Flow-directed seed dispersal: D_flow = seed_in - seed_out
        # Gather seed_in from upslope neighbors
        seed_in = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(OFFSETS[k])
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                opp = ti.static(OPP[k])
                frac = flow_frac[ni, nj, opp]
                if frac > 0.0:
                    q_per_w = Q_daily[ni, nj] / dx * 1000.0  # [mm*m/day]
                    q_seed = ti.min(c1 * q_per_w * V[ni, nj], c2 * V[ni, nj])
                    seed_in += frac * q_seed

        # Seed_out from this cell: Q_seed = min(c1 * q * V, c2 * V)
        q_self = Q_daily[i, j] / dx * 1000.0  # [mm*m/day]
        seed_out = ti.min(c1 * q_self * v, c2 * v)

        d_flow = seed_in - seed_out

        V_new[i, j] = ti.max(
            0.0,
            v + dt * (growth - mortality + d_flow / dx) + coeff_iso * laplacian,
        )
