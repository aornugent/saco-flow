"""
Soil moisture dynamics kernel.

Infiltration from overland flow, plant water uptake, and drainage loss.
Point-wise operation (no neighbor reads) — can be applied in-place.
"""

import taichi as ti


@ti.kernel
def infiltration_step(
    I_inf: ti.template(),
    h: ti.template(),
    V: ti.template(),
    mask: ti.template(),
    alpha: ti.f32,
    k2: ti.f32,
    W0: ti.f32,
):
    """Compute infiltration rate from flow depth and vegetation.

    I = alpha * h * (V + k2*W0) / (V + k2)

    Args:
        I_inf: Infiltration rate (write) [m/day]
        h: Flow depth [m]
        V: Vegetation density [%]
        mask: Active cell mask
        alpha: Infiltration capacity [1/day]
        k2: Vegetation half-saturation for infiltration [%]
        W0: Bare-soil infiltration fraction [-]
    """
    for i, j in I_inf:
        if mask[i, j] == 0:
            I_inf[i, j] = 0.0
            continue
        v = V[i, j]
        I_inf[i, j] = alpha * h[i, j] * (v + k2 * W0) / (v + k2)


@ti.kernel
def soil_moisture_step(
    M: ti.template(),
    M_new: ti.template(),
    I_inf: ti.template(),
    V: ti.template(),
    mask: ti.template(),
    g_max: ti.f32,
    k1: ti.f32,
    rw: ti.f32,
    dt: ti.f32,
):
    """Update soil moisture: infiltration, uptake, drainage.

    dM/dt = I - g_max * M/(M+k1) * V - rw * M

    Args:
        M: Soil moisture read [m]
        M_new: Soil moisture write [m]
        I_inf: Infiltration rate [m/day]
        V: Vegetation density [%]
        mask: Active cell mask
        g_max: Maximum growth rate [1/day]
        k1: Half-saturation for moisture [m]
        rw: Soil moisture loss rate [1/day]
        dt: Timestep [days]
    """
    for i, j in M:
        if mask[i, j] == 0:
            M_new[i, j] = M[i, j]
            continue

        m = M[i, j]
        v = V[i, j]

        uptake = g_max * m / (m + k1) * v
        loss = rw * m
        dMdt = I_inf[i, j] - uptake - loss

        M_new[i, j] = ti.max(0.0, m + dt * dMdt)
