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

    Linearised as dM/dt ≈ I - λ·M with λ = g_max·V/(M₀+k1) + rw,
    then integrated exactly: M(dt) = I/λ + (M₀ - I/λ)·exp(-λ·dt).
    Unconditionally stable at any dt — no blow-up when V is large.

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
        I = I_inf[i, j]

        # Effective linear decay rate (frozen Michaelis ratio at m)
        lam = g_max * v / (m + k1) + rw

        if lam * dt > 1e-6:
            M_eq = I / lam
            M[i, j] = ti.max(0.0, M_eq + (m - M_eq) * ti.exp(-lam * dt))
        else:
            M[i, j] = ti.max(0.0, m + dt * I)
