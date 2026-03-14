"""
Sediment transport and elevation update kernels.

Gather-based: each cell collects sediment from upslope neighbors,
computes transport capacity via stream power, and updates elevation
from sediment flux divergence (S_0 - S) / dx.
"""

import taichi as ti

from src.stencil import gather_flux, max_downslope


@ti.func
def _erosion_deposition_coeffs(
    v: ti.f32,
    K_max: ti.f32,
    K_min: ti.f32,
    P_min: ti.f32,
    P_max: ti.f32,
    v_low: ti.f32,
    v_high: ti.f32,
) -> ti.types.vector(2, ti.f32):
    """Compute erosion (K) and deposition (P) coefficients from vegetation.

    V <= v_low:  K=K_max, P=P_min
    V >= v_high: K=K_min, P=P_max
    Between: logarithmic interpolation
    """
    K = K_max
    P = P_min
    if v >= v_high:
        K = K_min
        P = P_max
    elif v > v_low:
        t = ti.log(v / v_low) / ti.log(v_high / v_low)
        K = K_max + (K_min - K_max) * t
        P = P_min + (P_max - P_min) * t
    return ti.Vector([K, P])


@ti.kernel
def sediment_transport(
    S: ti.template(),
    S_new: ti.template(),
    Q: ti.template(),
    z: ti.template(),
    V: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    gamma: ti.f32,
    m_exp: ti.f32,
    n_exp: ti.f32,
    K_max: ti.f32,
    K_min: ti.f32,
    P_min: ti.f32,
    P_max: ti.f32,
    v_low: ti.f32,
    v_high: ti.f32,
):
    """Gather-based sediment transport via stream power.

    1. Gather S_0 from upslope neighbors
    2. Q = Q_annual / dx  [m^2/yr]  (paper's convention)
    3. Transport capacity C = gamma * Q^m * slope^n  (Eq 2)
    4. h_sed = C / (beta * Q * slope)  (Eqs 4-5)
    5. S = C + (S_0 - C) * exp(-dx / h_sed)  (Eq 3)

    Q_annual (m^3/yr) is the accumulated daily volumetric discharge over
    the year.  Dividing by dx gives Q in m^2/yr — the paper's unit-width
    annual discharge used directly in the LAPSUS equations.

    Args:
        S: Sediment flux read [kg/m/yr]
        S_new: Sediment flux write [kg/m/yr]
        Q: Annual cumulative volumetric discharge [m^3/yr]
        z: Elevation [m]
        V: Vegetation density [%]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
        gamma: Transport coefficient [-]
        m_exp: Discharge exponent [-]
        n_exp: Slope exponent [-]
        K_max, K_min: Erosion coefficient range [-]
        P_min, P_max: Deposition coefficient range [-]
        v_low: Vegetation threshold for max erosion [%]
        v_high: Vegetation threshold for min erosion [%]
    """
    n = S.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            S_new[i, j] = 0.0
            continue

        # Gather incoming sediment from upslope
        S_0 = gather_flux(S, flow_frac, mask, i, j)

        # Max downslope gradient
        slope_max = max_downslope(z, flow_frac, i, j, dx)

        # Unit-width discharge [m^2/yr] — paper's Q convention
        q = Q[i, j] / dx

        # Transport capacity C = gamma * q^m * slope^n
        C = gamma * ti.pow(ti.max(q, 0.0), m_exp) * ti.pow(slope_max, n_exp)

        # Adaptation length: h_sed = C / (beta * q * slope_max)
        kp = _erosion_deposition_coeffs(
            V[i, j], K_max, K_min, P_min, P_max, v_low, v_high
        )
        coeff = kp[1]  # deposition by default
        if S_0 < C:
            coeff = kp[0]  # erosion
        h_sed = C / ti.max(coeff * q * slope_max, 1e-10)
        S_new[i, j] = ti.max(0.0, C + (S_0 - C) * ti.exp(-dx / h_sed))


@ti.kernel
def update_elevation(
    z: ti.template(),
    S: ti.template(),
    S_new: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
):
    """Update elevation from sediment flux divergence.

    z += (S_0 - S_new) / dx

    S holds pre-transport flux (gather S_0 from neighbors).
    S_new holds post-transport flux.
    Call before swapping S buffers.

    Args:
        z: Elevation field (read/write) [m]
        S: Pre-transport sediment flux (read) [kg/m/yr]
        S_new: Post-transport sediment flux (read) [kg/m/yr]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
    """
    n = z.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        # Gather incoming sediment S_0 from pre-transport flux
        S_0 = gather_flux(S, flow_frac, mask, i, j)

        z[i, j] += (S_0 - S_new[i, j]) / dx
