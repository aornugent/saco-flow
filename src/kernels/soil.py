"""
Soil moisture dynamics: ET, deep leakage, and lateral diffusion.

∂M/∂t = -E(M,P) - L(M) + D_M·∇²M

(Infiltration handled separately in infiltration.py)
"""

import taichi as ti

from src.config import DTYPE
from src.kernels.utils import copy_field


@ti.kernel
def evapotranspiration_step(
    M: ti.template(),
    P: ti.template(),
    mask: ti.template(),
    E_max: DTYPE,
    k_M: DTYPE,
    beta_E: DTYPE,
    dt: DTYPE,
) -> DTYPE:
    """
    Compute evapotranspiration loss from soil moisture.

    E = E_max · M/(M + k_M) · (1 + β_E·P)

    Monod kinetics in moisture, enhanced by vegetation.
    Returns total ET for mass balance tracking.
    """
    total_et = ti.cast(0.0, DTYPE)
    n = M.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        M_local = M[i, j]
        P_local = P[i, j]

        if M_local <= 0:
            continue

        # Monod kinetics with vegetation enhancement
        ET = E_max * M_local / (M_local + k_M) * (1.0 + beta_E * P_local) * dt

        # Limit to available moisture
        ET = ti.min(ET, M_local)

        M[i, j] = M_local - ET
        ti.atomic_add(total_et, ET)

    return total_et


@ti.kernel
def leakage_step(
    M: ti.template(),
    mask: ti.template(),
    L_max: DTYPE,
    M_sat: DTYPE,
    dt: DTYPE,
) -> DTYPE:
    """
    Compute deep leakage loss from soil moisture.

    L = L_max · (M/M_sat)²

    Quadratic - negligible when dry, significant near saturation.
    Returns total leakage for mass balance tracking.
    """
    total_leakage = ti.cast(0.0, DTYPE)
    n = M.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        M_local = M[i, j]

        if M_local <= 0:
            continue

        # Quadratic leakage
        sat_ratio = M_local / M_sat
        leakage = L_max * sat_ratio * sat_ratio * dt

        # Limit to available moisture
        leakage = ti.min(leakage, M_local)

        M[i, j] = M_local - leakage
        ti.atomic_add(total_leakage, leakage)

    return total_leakage


@ti.kernel
def diffusion_step(
    M: ti.template(),
    M_new: ti.template(),
    mask: ti.template(),
    D_M: DTYPE,
    dx: DTYPE,
    dt: DTYPE,
):
    """
    Compute lateral soil moisture diffusion using 5-point Laplacian.

    ∇²M ≈ (M_E + M_W + M_N + M_S - 4·M) / dx²

    Neumann (no-flux) boundary conditions: only include neighbors where mask=1.
    Uses double buffering (reads from M, writes to M_new).
    """
    n = M.shape[0]
    coeff = D_M * dt / (dx * dx)

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            M_new[i, j] = M[i, j]
            continue

        M_local = M[i, j]

        # 5-point Laplacian with Neumann BC
        laplacian = ti.cast(0.0, DTYPE)

        # Check 4 cardinal neighbors
        for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += M[ni, nj] - M_local

        # Apply diffusion
        dM = coeff * laplacian
        M_new[i, j] = ti.max(0.0, M_local + dM)


def soil_moisture_step(
    M, M_new, P, mask, E_max, k_M, beta_E, L_max, M_sat, D_M, dx, dt
) -> tuple[float, float]:
    """
    Combined soil moisture update: ET, leakage, diffusion.

    Returns (total_et, total_leakage) for mass balance tracking.
    """
    # ET and leakage modify M in-place
    total_et = evapotranspiration_step(M, P, mask, E_max, k_M, beta_E, dt)
    total_leakage = leakage_step(M, mask, L_max, M_sat, dt)

    # Diffusion uses double buffering
    diffusion_step(M, M_new, mask, D_M, dx, dt)

    # Copy back
    copy_field(M_new, M)

    return float(total_et), float(total_leakage)


def compute_diffusion_timestep(D_M: float, dx: float, cfl: float = 0.25) -> float:
    """
    Compute stable timestep for soil moisture diffusion.

    dt <= cfl * dx² / D_M  (stability requires cfl <= 0.25 for 2D)
    """
    if D_M <= 0:
        return float("inf")
    return cfl * dx * dx / D_M
