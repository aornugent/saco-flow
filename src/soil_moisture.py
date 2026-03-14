"""
Soil moisture dynamics kernel.

Infiltration from overland flow, plant water uptake, and drainage loss.
Point-wise operation (no neighbor reads) — updated in-place.
"""

import taichi as ti


@ti.kernel
def soil_moisture_step(
    M: ti.template(),
    I_inf: ti.template(),
    V: ti.template(),
    mask: ti.template(),
    g_max: ti.f32,
    k1: ti.f32,
    rw: ti.f32,
    dt: ti.f32,
):
    """Update soil moisture in-place: infiltration, uptake, drainage.

    dM/dt = I - g_max * M/(M+k1) * V - rw * M

    Args:
        M: Soil moisture (read/write, in-place) [mm]
        I_inf: Infiltration rate [mm/day]
        V: Vegetation density [g/m^2]
        mask: Active cell mask
        g_max: Maximum growth rate [mm*m^2/(g*day)]
        k1: Half-saturation for moisture [mm]
        rw: Soil moisture loss rate [1/day]
        dt: Timestep [days]
    """
    for i, j in M:
        if mask[i, j] == 0:
            continue

        m = M[i, j]
        v = V[i, j]

        uptake = g_max * m / (m + k1) * v
        loss = rw * m
        dMdt = I_inf[i, j] - uptake - loss

        M[i, j] = ti.max(0.0, m + dt * dMdt)
