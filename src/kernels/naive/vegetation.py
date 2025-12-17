"""
Naive vegetation dynamics kernels.

Implements growth, mortality, and seed dispersal.
Equation: dP/dt = G(M)·P - mu·P + D_P·nabla²P

Where:
- G(M) = g_max · M/(M + k_G)  -- Monod growth kinetics
- mu·P                        -- constant mortality
- D_P·nabla²P                 -- seed dispersal diffusion
"""

import taichi as ti

from src.core.dtypes import DTYPE
from src.kernels.protocol import VegetationKernel, VegetationFluxes


@ti.kernel
def _copy_field(src: ti.template(), dst: ti.template()):
    """Copy src to dst."""
    for I in ti.grouped(src):
        dst[I] = src[I]


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

    dP = -mu · P · dt

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
    P, P_new, M, mask, g_max, k_G, mu, D_P, dx, dt
) -> tuple[float, float]:
    """
    Combined vegetation update: growth, mortality, dispersal.

    Returns (total_growth, total_mortality) for tracking.
    """
    total_growth = growth_step(P, M, mask, g_max, k_G, dt)
    total_mortality = mortality_step(P, mask, mu, dt)

    # Diffusion uses double buffering
    vegetation_diffusion_step(P, P_new, mask, D_P, dx, dt)
    _copy_field(P_new, P)

    return float(total_growth), float(total_mortality)


def compute_equilibrium_moisture(g_max: float, k_G: float, mu: float) -> float:
    """
    Compute the moisture level where growth rate equals mortality rate.

    At equilibrium: G(M) = mu  =>  g_max · M / (M + k_G) = mu

    Solving for M:
        M = mu · k_G / (g_max - mu)

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


# Protocol-compliant wrapper


class NaiveVegetationKernel:
    """Naive implementation of vegetation dynamics.

    Implements the VegetationKernel protocol using separate passes for
    growth, mortality, and diffusion. Reference implementation for
    equivalence testing against optimized variants.
    """

    def step(self, state, static, params, dx: float, dt: float) -> VegetationFluxes:
        """Execute one vegetation timestep.

        Args:
            state: State fields (m, p, and buffers)
            static: Static fields (mask)
            params: Vegetation parameters or SimulationConfig
            dx: Cell size [m]
            dt: Timestep [days]

        Returns:
            VegetationFluxes with total_growth and total_mortality
        """
        # Extract parameters (handle both VegetationParams and SimulationConfig)
        if hasattr(params, "vegetation"):
            # SimulationConfig
            g_max = params.vegetation.g_max
            k_G = params.vegetation.k_G
            mu = params.vegetation.mu
            D_P = params.vegetation.D_P
        else:
            # Direct VegetationParams
            g_max = params.g_max
            k_G = params.k_G
            mu = params.mu
            D_P = params.D_P

        total_growth, total_mortality = vegetation_step(
            state.p,
            state.p_new,
            state.m,
            static.mask,
            g_max,
            k_G,
            mu,
            D_P,
            dx,
            dt,
        )

        return VegetationFluxes(total_growth=total_growth, total_mortality=total_mortality)

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        return {"m", "p", "mask"}

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        return {"p"}
