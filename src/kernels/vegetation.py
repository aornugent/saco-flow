"""
Vegetation dynamics: growth, mortality, and seed dispersal.

∂P/∂t = G(M)·P - μ·P + D_P·∇²P

Where:
- G(M) = g_max · M/(M + k_G)  -- Monod growth kinetics
- μ·P                         -- constant mortality
- D_P·∇²P                     -- seed dispersal diffusion

Provides both naive (separate) and fused kernels:
- Naive: growth_step, mortality_step, vegetation_diffusion_step
- Fused: growth_mortality_step_fused (2x less memory traffic for point-wise ops)

The fused kernel is used by default. Naive kernels are kept for regression testing.
"""

from types import SimpleNamespace

import taichi as ti

from src.fields import swap_buffers
from src.geometry import DTYPE, laplacian_diffusion_step


@ti.kernel
def growth_step(
    P: ti.template(),
    M: ti.template(),
    mask: ti.template(),
    g_max: DTYPE,
    k_G: DTYPE,
    dt: DTYPE,
) -> DTYPE:
    """
    Compute vegetation growth using Monod kinetics.

    G(M) = g_max · M / (M + k_G)
    dP = G(M) · P · dt

    Growth rate saturates at high moisture.
    Returns total growth for tracking.
    """
    total_growth = ti.cast(0.0, DTYPE)
    n = P.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        P_local = P[i, j]
        M_local = M[i, j]

        if P_local <= 0 or M_local <= 0:
            continue

        # Monod growth rate
        growth_rate = g_max * M_local / (M_local + k_G)
        growth = growth_rate * P_local * dt

        P[i, j] = P_local + growth
        ti.atomic_add(total_growth, growth)

    return total_growth


@ti.kernel
def mortality_step(
    P: ti.template(),
    mask: ti.template(),
    mu: DTYPE,
    dt: DTYPE,
) -> DTYPE:
    """
    Apply constant mortality rate.

    dP = -μ · P · dt

    Returns total mortality for tracking.
    """
    total_mortality = ti.cast(0.0, DTYPE)
    n = P.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        P_local = P[i, j]

        if P_local <= 0:
            continue

        mortality = mu * P_local * dt
        mortality = ti.min(mortality, P_local)  # Can't lose more than exists

        P[i, j] = P_local - mortality
        ti.atomic_add(total_mortality, mortality)

    return total_mortality


@ti.kernel
def growth_mortality_step_fused(
    P: ti.template(),
    M: ti.template(),
    mask: ti.template(),
    g_max: DTYPE,
    k_G: DTYPE,
    mu: DTYPE,
    dt: DTYPE,
) -> ti.types.vector(2, DTYPE):
    """
    Fused growth and mortality step (2x less memory traffic).

    Applies growth first, then mortality (matching naive sequential behavior):
        1. P_grown = P + G(M)·P·dt  where G(M) = g_max · M / (M + k_G)
        2. P_final = P_grown - μ·P_grown·dt

    Returns [total_growth, total_mortality] for tracking.
    """
    total_growth = ti.cast(0.0, DTYPE)
    total_mortality = ti.cast(0.0, DTYPE)
    n = P.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        P_local = P[i, j]
        if P_local <= 0:
            continue

        M_local = M[i, j]

        # Monod growth rate (0 if no moisture)
        growth_rate = ti.cast(0.0, DTYPE)
        if M_local > 0:
            growth_rate = g_max * M_local / (M_local + k_G)

        # Growth applied first
        growth = growth_rate * P_local * dt
        P_after_growth = P_local + growth

        # Mortality applied to grown value (matches naive sequential behavior)
        mortality = mu * P_after_growth * dt
        mortality = ti.min(mortality, P_after_growth)  # Can't lose more than exists

        # Single write to P
        P[i, j] = P_after_growth - mortality

        ti.atomic_add(total_growth, growth)
        ti.atomic_add(total_mortality, mortality)

    return ti.Vector([total_growth, total_mortality])


@ti.kernel
def vegetation_diffusion_step(
    P: ti.template(),
    P_new: ti.template(),
    mask: ti.template(),
    D_P: DTYPE,
    dx: DTYPE,
    dt: DTYPE,
):
    """
    Compute seed dispersal via diffusion using 5-point Laplacian.

    Uses Neumann (no-flux) boundary conditions.
    Double buffered: reads from P, writes to P_new.
    """
    n = P.shape[0]
    coeff = D_P * dt / (dx * dx)

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            P_new[i, j] = P[i, j]
            continue

        P_local = P[i, j]

        # 5-point Laplacian with Neumann BC
        laplacian = ti.cast(0.0, DTYPE)
        for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += P[ni, nj] - P_local

        dP = coeff * laplacian
        P_new[i, j] = ti.max(0.0, P_local + dP)


def vegetation_step(
    fields: SimpleNamespace,
    g_max: float,
    k_G: float,
    mu: float,
    D_P: float,
    dx: float,
    dt: float,
) -> tuple[float, float]:
    """
    Combined vegetation update: growth, mortality, dispersal.

    Uses fused kernel for growth+mortality (2x less memory traffic) and
    generic laplacian diffusion.

    Uses ping-pong buffering: after this call, fields.P holds the updated
    vegetation biomass (buffers are swapped, no copy needed).

    Returns (total_growth, total_mortality) for tracking.
    """
    # Fused growth+mortality: single read/write per cell (point-wise, in-place)
    totals = growth_mortality_step_fused(
        fields.P, fields.M, fields.mask, g_max, k_G, mu, dt
    )

    # Diffusion: reads P, writes P_new (stencil operation)
    laplacian_diffusion_step(fields.P, fields.P_new, fields.mask, D_P, dx, dt)

    # Swap buffers (O(1) pointer swap, not O(n²) copy)
    swap_buffers(fields, "P")

    return float(totals[0]), float(totals[1])


def vegetation_step_naive(
    fields: SimpleNamespace,
    g_max: float,
    k_G: float,
    mu: float,
    D_P: float,
    dx: float,
    dt: float,
) -> tuple[float, float]:
    """
    Naive (unfused) vegetation step for regression testing.

    Same physics as vegetation_step but uses separate kernels.
    """
    # Separate growth and mortality calls (more memory traffic)
    total_growth = growth_step(fields.P, fields.M, fields.mask, g_max, k_G, dt)
    total_mortality = mortality_step(fields.P, fields.mask, mu, dt)

    # Use local vegetation_diffusion_step (identical to generic)
    vegetation_diffusion_step(fields.P, fields.P_new, fields.mask, D_P, dx, dt)

    swap_buffers(fields, "P")

    return float(total_growth), float(total_mortality)


def compute_equilibrium_moisture(g_max: float, k_G: float, mu: float) -> float:
    """
    Compute the moisture level where growth rate equals mortality rate.

    At equilibrium: G(M) = μ  =>  g_max · M / (M + k_G) = μ

    Solving for M:
        M = μ · k_G / (g_max - μ)

    Returns the equilibrium moisture, or inf if growth never exceeds mortality.
    """
    if g_max <= mu:
        return float("inf")
    return mu * k_G / (g_max - mu)


def compute_vegetation_timestep(D_P: float, dx: float, cfl: float = 0.25) -> float:
    """
    Compute stable timestep for vegetation diffusion.

    dt <= cfl * dx² / D_P
    """
    if D_P <= 0:
        return float("inf")
    return cfl * dx * dx / D_P
