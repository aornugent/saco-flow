"""
Flow direction, accumulation, and surface water routing kernels.

MFD (Multiple Flow Direction) distributes flow to downslope neighbors
proportional to slope^p. Routing uses kinematic wave with CFL limiting.
"""

import taichi as ti

from src.fields import copy_field
from src.geometry import DTYPE, NEIGHBOR_DI, NEIGHBOR_DIST, NEIGHBOR_DJ

# Flow exponent: 1.0=diffuse, 1.5=default, >5=approaches D8
FLOW_EXPONENT = 1.5

# Minimum values to avoid division by zero
MIN_SLOPE_SUM = 1e-10
MIN_DEPTH = 1e-8  # [m] threshold for flow computation


@ti.kernel
def compute_flow_directions(
    Z: ti.template(),
    mask: ti.template(),
    flow_frac: ti.template(),
    dx: DTYPE,
    p: DTYPE,
):
    """
    Compute MFD flow fractions for each cell.

    Distributes outflow to lower neighbors proportional to slope^p.
    Flat cells (no downslope neighbors) are flagged with flow_frac[i,j,0] = -1.
    """
    n = Z.shape[0]
    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        z_center = Z[i, j]
        slope_sum = ti.cast(0.0, DTYPE)

        # Compute slope^p to each neighbor, store temporarily
        # Include boundary neighbors to allow outflow from domain
        slopes = ti.Vector([ti.cast(0.0, DTYPE)] * 8)
        for k in ti.static(range(8)):
            ni = i + NEIGHBOR_DI[k]
            nj = j + NEIGHBOR_DJ[k]

            # Compute slope to all neighbors (including boundary cells)
            dz = z_center - Z[ni, nj]
            if dz > 0:  # Downslope
                slope = dz / (NEIGHBOR_DIST[k] * dx)
                slopes[k] = ti.pow(slope, p)
                slope_sum += slopes[k]

        # Normalize to fractions or flag as flat
        if slope_sum > MIN_SLOPE_SUM:
            for k in ti.static(range(8)):
                flow_frac[i, j, k] = slopes[k] / slope_sum
        else:
            # Flat cell or local minimum
            for k in ti.static(range(8)):
                flow_frac[i, j, k] = 0.0
            flow_frac[i, j, 0] = -1.0  # Flag


@ti.kernel
def flow_accumulation_step(
    local_source: ti.template(),
    flow_acc: ti.template(),
    flow_acc_new: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
) -> DTYPE:
    """
    One iteration of parallel flow accumulation.

    Each cell accumulates local source plus contributions from upslope donors.
    Returns maximum change for convergence checking.
    """
    max_change = ti.cast(0.0, DTYPE)
    n = flow_acc.shape[0]

    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        # Start with local contribution
        acc_new = local_source[i, j]

        # Add contributions from 8 neighbors (if they flow to us)
        for k in ti.static(range(8)):
            ni = i + NEIGHBOR_DI[k]
            nj = j + NEIGHBOR_DJ[k]

            if mask[ni, nj] == 1:
                # Neighbor k flows to us via direction (k+4)%8
                donor_dir = (k + 4) % 8
                frac = flow_frac[ni, nj, donor_dir]
                if frac > 0:
                    acc_new += frac * flow_acc[ni, nj]

        change = ti.abs(acc_new - flow_acc[i, j])
        ti.atomic_max(max_change, change)
        flow_acc_new[i, j] = acc_new

    return max_change


def compute_flow_accumulation(
    local_source,
    flow_acc,
    flow_acc_new,
    flow_frac,
    mask,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> int:
    """
    Iteratively compute flow accumulation until convergence.

    Returns number of iterations taken.
    """
    # Initialize accumulation with local source
    copy_field(local_source, flow_acc)

    for iteration in range(max_iters):
        change = flow_accumulation_step(
            local_source, flow_acc, flow_acc_new, flow_frac, mask
        )
        copy_field(flow_acc_new, flow_acc)

        if change < tol:
            return iteration + 1

    return max_iters


@ti.kernel
def compute_outflow(
    h: ti.template(),
    Z: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    q_out: ti.template(),
    dx: DTYPE,
    dt: DTYPE,
    manning_n: DTYPE,
):
    """
    Compute outflow rate for each cell (Pass 1 of routing).

    Uses kinematic wave: v = h^(2/3) * sqrt(S) / n
    CFL-limited: q_out <= h/dt
    """
    n = h.shape[0]
    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            q_out[i, j] = 0.0
            continue

        h_local = h[i, j]
        if h_local <= MIN_DEPTH:
            q_out[i, j] = 0.0
            continue

        # Find maximum slope among flow directions
        S_max = ti.cast(0.0, DTYPE)
        for k in ti.static(range(8)):
            if flow_frac[i, j, k] > 0:
                ni = i + NEIGHBOR_DI[k]
                nj = j + NEIGHBOR_DJ[k]
                S = (Z[i, j] - Z[ni, nj]) / (NEIGHBOR_DIST[k] * dx)
                S_max = ti.max(S_max, S)

        if S_max > 0:
            # Kinematic wave velocity
            v = ti.pow(h_local, 2.0 / 3.0) * ti.sqrt(S_max) / manning_n
            # CFL-limited outflow rate [m/s] * [m] / [m] = [m/s]
            q_out[i, j] = ti.min(h_local / dt, v * h_local / dx)
        else:
            q_out[i, j] = 0.0


@ti.kernel
def apply_fluxes(
    h: ti.template(),
    q_out: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dt: DTYPE,
) -> DTYPE:
    """
    Apply outflow and inflow fluxes to update water depth (Pass 2 of routing).

    Returns total outflow at domain boundaries for mass balance tracking.
    """
    boundary_outflow = ti.cast(0.0, DTYPE)
    n = h.shape[0]

    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        # Outflow from this cell
        delta_h = -q_out[i, j] * dt

        # Inflow from neighbors
        for k in ti.static(range(8)):
            ni = i + NEIGHBOR_DI[k]
            nj = j + NEIGHBOR_DJ[k]

            if mask[ni, nj] == 1:
                # Neighbor donates via direction (k+4)%8
                donor_dir = (k + 4) % 8
                frac = flow_frac[ni, nj, donor_dir]
                if frac > 0:
                    delta_h += frac * q_out[ni, nj] * dt

        h[i, j] = ti.max(0.0, h[i, j] + delta_h)

    # Track boundary outflow (water leaving domain)
    # Check cells adjacent to boundaries
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        for k in ti.static(range(8)):
            ni = i + NEIGHBOR_DI[k]
            nj = j + NEIGHBOR_DJ[k]

            # If neighbor is boundary/outside, count outflow
            if mask[ni, nj] == 0:
                frac = flow_frac[i, j, k]
                if frac > 0:
                    ti.atomic_add(boundary_outflow, frac * q_out[i, j] * dt)

    return boundary_outflow


def route_surface_water(h, Z, flow_frac, mask, q_out, dx, dt, manning_n) -> float:
    """
    Route surface water one timestep using two-pass scheme.

    Returns boundary outflow for mass balance tracking.
    """
    compute_outflow(h, Z, flow_frac, mask, q_out, dx, dt, manning_n)
    return apply_fluxes(h, q_out, flow_frac, mask, dt)


@ti.kernel
def compute_max_velocity(
    h: ti.template(),
    Z: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: DTYPE,
    manning_n: DTYPE,
) -> DTYPE:
    """Compute maximum flow velocity for CFL timestep calculation."""
    v_max = ti.cast(0.0, DTYPE)
    n = h.shape[0]

    ti.loop_config(block_dim=1024)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        h_local = h[i, j]
        if h_local <= MIN_DEPTH:
            continue

        # Find max slope
        S_max = ti.cast(0.0, DTYPE)
        for k in ti.static(range(8)):
            if flow_frac[i, j, k] > 0:
                ni = i + NEIGHBOR_DI[k]
                nj = j + NEIGHBOR_DJ[k]
                S = (Z[i, j] - Z[ni, nj]) / (NEIGHBOR_DIST[k] * dx)
                S_max = ti.max(S_max, S)

        if S_max > 0:
            v = ti.pow(h_local, 2.0 / 3.0) * ti.sqrt(S_max) / manning_n
            ti.atomic_max(v_max, v)

    return v_max


def compute_cfl_timestep(
    h, Z, flow_frac, mask, dx: float, manning_n: float, cfl: float = 0.5
) -> float:
    """
    Compute CFL-limited timestep for surface routing.

    dt <= cfl * dx / v_max
    """
    v_max = compute_max_velocity(h, Z, flow_frac, mask, dx, manning_n)
    if v_max < 1e-10:
        return float("inf")  # No flow, timestep unlimited
    return cfl * dx / v_max
