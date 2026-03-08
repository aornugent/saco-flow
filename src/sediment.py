"""
Sediment transport and elevation update kernels.

Gather-based: each cell collects sediment from upslope neighbors,
computes transport capacity via stream power, and updates elevation
from net erosion/deposition.
"""

import taichi as ti

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_OPP = [7, 6, 5, 4, 3, 2, 1, 0]
_DIAG = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]


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
    Q_out: ti.template(),
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
    2. Compute transport capacity C = gamma * Q^m * slope^n
    3. h_sed = C / (K*q*slope) for erosion, C / (P*q*slope) for deposition
    4. S = C + (S_0 - C) * exp(-dx/h_sed)

    Args:
        S: Sediment flux read [kg/m/day]
        S_new: Sediment flux write [kg/m/day]
        Q_out: Water discharge [m^3/day]
        z: Elevation [m]
        V: Vegetation density [%]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
        gamma: Transport coefficient [kg*day^(m-1)/m^(3m+n)]
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
        S_0 = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(_OFFSETS[k])
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                opp = ti.static(_OPP[k])
                S_0 += flow_frac[ni, nj, opp] * S[ni, nj]

        # Max downslope gradient
        slope_max = ti.cast(1e-4, ti.f32)
        for k in ti.static(range(8)):
            if flow_frac[i, j, k] > 0.0:
                di, dj = ti.static(_OFFSETS[k])
                dist = ti.static(_DIAG[k]) * dx
                ni, nj = i + di, j + dj
                slope_k = (z[i, j] - z[ni, nj]) / dist
                slope_max = ti.max(slope_max, slope_k)

        # Transport capacity
        q = Q_out[i, j]
        C = gamma * ti.pow(ti.max(q, 0.0), m_exp) * ti.pow(slope_max, n_exp)

        # Adaptation length: h = C / (detachment or deposition capacity)
        kp = _erosion_deposition_coeffs(
            V[i, j], K_max, K_min, P_min, P_max, v_low, v_high
        )
        denom = q * slope_max
        coeff = kp[1]  # deposition regime by default
        if S_0 < C:
            coeff = kp[0]  # erosion regime
        h_sed = ti.max(C / ti.max(coeff * denom, 1e-10), dx)  # [m]
        S_new[i, j] = ti.max(0.0, C + (S_0 - C) * ti.exp(-dx / h_sed))


@ti.kernel
def update_elevation(
    z: ti.template(),
    Q_out: ti.template(),
    V: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    K_max: ti.f32,
    K_min: ti.f32,
    P_min: ti.f32,
    P_max: ti.f32,
    v_low: ti.f32,
    v_high: ti.f32,
    dt: ti.f32,
):
    """Update elevation from erosion and deposition.

    dz/dt = P*Q*slope - K*Q*slope  (net deposition - erosion)

    K, P depend on vegetation density via logarithmic interpolation.

    Args:
        z: Elevation field (read/write, point-wise) [m]
        Q_out: Water discharge [m^3/day]
        V: Vegetation density [%]
        mask: Active cell mask
        dx: Cell spacing [m]
        K_max, K_min: Erosion coefficient range [-]
        P_min, P_max: Deposition coefficient range [-]
        v_low: Vegetation threshold for max erosion [%]
        v_high: Vegetation threshold for min erosion [%]
        dt: Timestep [days]
    """
    n = z.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        kp = _erosion_deposition_coeffs(
            V[i, j], K_max, K_min, P_min, P_max, v_low, v_high
        )
        K = kp[0]
        P = kp[1]

        q = Q_out[i, j]
        # Compute local slope (max downslope)
        slope = ti.cast(1e-4, ti.f32)
        for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                s = (z[i, j] - z[ni, nj]) / dx
                slope = ti.max(slope, s)

        deposition = P * q * slope
        erosion = K * q * slope

        z[i, j] += dt * (deposition - erosion)
