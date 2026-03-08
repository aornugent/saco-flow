"""
Soil moisture diffusion kernel.

Implements lateral diffusion of soil moisture using a 5-point Laplacian stencil:

    dM/dt = D * nabla^2(M)

Double-buffered: reads from M, writes to M_new. Caller must swap after.
"""

import taichi as ti


@ti.kernel
def diffusion_step(
    M: ti.template(),
    M_new: ti.template(),
    mask: ti.template(),
    D: ti.f32,
    dx: ti.f32,
    dt: ti.f32,
):
    """
    Apply 5-point Laplacian diffusion with Neumann (no-flux) boundaries.

    dM/dt = D * nabla^2(M)

    Uses ti.block_local() for shared memory caching on GPU.

    Args:
        M: Source moisture field (read) [m]
        M_new: Destination moisture field (write) [m]
        mask: Active cell mask (1=active, 0=boundary)
        D: Diffusion coefficient [m^2/day]
        dx: Cell size [m]
        dt: Timestep [days]
    """
    ti.block_local(M)

    n = M.shape[0]
    coeff = D * dt / (dx * dx)

    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            M_new[i, j] = M[i, j]
            continue

        val = M[i, j]

        # 5-point Laplacian with Neumann BC (only include active neighbors)
        laplacian = ti.cast(0.0, ti.f32)
        for di, dj in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += M[ni, nj] - val

        M_new[i, j] = ti.max(0.0, val + coeff * laplacian)


def compute_stable_dt(D: float, dx: float, cfl: float = 0.2) -> float:
    """Compute CFL-stable timestep for 2D diffusion.

    Stability requires dt <= dx^2 / (4*D) for the 5-point stencil.

    Args:
        D: Diffusion coefficient [m^2/day]
        dx: Cell size [m]
        cfl: CFL safety factor (default 0.2)

    Returns:
        Stable timestep [days]
    """
    return cfl * dx * dx / (4.0 * D)
