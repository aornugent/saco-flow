"""
Soil moisture dynamics: ET, deep leakage, and lateral diffusion.

∂M/∂t = -E(M,P) - L(M) + D_M·∇²M

(Infiltration handled separately in infiltration.py)

Provides both naive (separate) and fused kernels:
- Naive: evapotranspiration_step, leakage_step, diffusion_step
- Fused: et_leakage_step_fused (2x less memory traffic for point-wise ops)

The fused kernel is used by default. Naive kernels are kept for regression testing.
"""

from types import SimpleNamespace

import taichi as ti

from src.fields import swap_buffers
from src.geometry import DTYPE, laplacian_diffusion_step


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
def et_leakage_step_fused(
    M: ti.template(),
    P: ti.template(),
    mask: ti.template(),
    E_max: DTYPE,
    k_M: DTYPE,
    beta_E: DTYPE,
    L_max: DTYPE,
    M_sat: DTYPE,
    dt: DTYPE,
) -> ti.types.vector(2, DTYPE):
    """
    Fused evapotranspiration and leakage step (2x less memory traffic).

    Combines:
        E = E_max · M/(M + k_M) · (1 + β_E·P)   -- Monod kinetics with vegetation
        L = L_max · (M/M_sat)²                   -- Quadratic leakage

    Returns [total_et, total_leakage] for mass balance tracking.
    """
    total_et = ti.cast(0.0, DTYPE)
    total_leakage = ti.cast(0.0, DTYPE)
    n = M.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        M_local = M[i, j]
        if M_local <= 0:
            continue

        P_local = P[i, j]

        # Evapotranspiration: Monod kinetics with vegetation enhancement
        ET = E_max * M_local / (M_local + k_M) * (1.0 + beta_E * P_local) * dt
        ET = ti.min(ET, M_local)

        # Leakage: Quadratic (negligible when dry, significant near saturation)
        M_after_et = M_local - ET
        sat_ratio = M_after_et / M_sat
        leakage = L_max * sat_ratio * sat_ratio * dt
        leakage = ti.min(leakage, M_after_et)

        # Single write to M
        M[i, j] = M_after_et - leakage

        ti.atomic_add(total_et, ET)
        ti.atomic_add(total_leakage, leakage)

    return ti.Vector([total_et, total_leakage])


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
    fields: SimpleNamespace,
    E_max: float,
    k_M: float,
    beta_E: float,
    L_max: float,
    M_sat: float,
    D_M: float,
    dx: float,
    dt: float,
) -> tuple[float, float]:
    """
    Combined soil moisture update: ET, leakage, diffusion.

    Uses fused kernel for ET+leakage (2x less memory traffic) and
    generic laplacian diffusion.

    Uses ping-pong buffering: after this call, fields.M holds the updated
    soil moisture (buffers are swapped, no copy needed).

    Returns (total_et, total_leakage) for mass balance tracking.
    """
    # Fused ET+leakage: single read/write per cell (point-wise, in-place)
    totals = et_leakage_step_fused(
        fields.M, fields.P, fields.mask, E_max, k_M, beta_E, L_max, M_sat, dt
    )

    # Diffusion: reads M, writes M_new (stencil operation)
    laplacian_diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)

    # Swap buffers (O(1) pointer swap, not O(n²) copy)
    swap_buffers(fields, "M")

    return float(totals[0]), float(totals[1])


def soil_moisture_step_naive(
    fields: SimpleNamespace,
    E_max: float,
    k_M: float,
    beta_E: float,
    L_max: float,
    M_sat: float,
    D_M: float,
    dx: float,
    dt: float,
) -> tuple[float, float]:
    """
    Naive (unfused) soil moisture step for regression testing.

    Same physics as soil_moisture_step but uses separate kernels.
    """
    # Separate ET and leakage calls (more memory traffic)
    total_et = evapotranspiration_step(
        fields.M, fields.P, fields.mask, E_max, k_M, beta_E, dt
    )
    total_leakage = leakage_step(fields.M, fields.mask, L_max, M_sat, dt)

    # Use local diffusion_step (identical to generic)
    diffusion_step(fields.M, fields.M_new, fields.mask, D_M, dx, dt)

    swap_buffers(fields, "M")

    return float(total_et), float(total_leakage)


def compute_diffusion_timestep(D_M: float, dx: float, cfl: float = 0.25) -> float:
    """
    Compute stable timestep for soil moisture diffusion.

    dt <= cfl * dx² / D_M  (stability requires cfl <= 0.25 for 2D)
    """
    if D_M <= 0:
        return float("inf")
    return cfl * dx * dx / D_M
