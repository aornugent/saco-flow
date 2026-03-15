"""
Flow routing kernels — MFD flow fractions and wavefront water routing.

Multiple-flow direction (MFD) algorithm distributes outflow proportionally
to downslope gradients raised to exponent p.  Wavefront routing processes
cells upstream-to-downstream by drainage level so Q_in is exact.

All discharge fields are unit-width, mm-based [mm*m/day].
"""

import numpy as np
import taichi as ti

from src.stencil import DIAG, OFFSETS, OPP, gather_flux, max_downslope


@ti.kernel
def accumulate_annual_Q(
    Q_annual: ti.template(),
    Q_daily: ti.template(),
    mask: ti.template(),
):
    """Add daily cell-average discharge into annual accumulator.

    Args:
        Q_annual: Running annual total (read/write) [mm*m/yr]
        Q_daily: Daily cell-average discharge [mm*m/day]
        mask: Active cell mask
    """
    n = Q_annual.shape[0]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 1:
            Q_annual[i, j] += Q_daily[i, j]


@ti.kernel
def compute_flow_fractions(
    z: ti.template(),
    mask: ti.template(),
    flow_frac: ti.template(),
    dx: ti.f32,
    p: ti.f32,
):
    """Compute MFD flow fractions from each cell to its 8 neighbors.

    F_k = max(0, slope_k)^p / sum(max(0, slope_k)^p)

    Args:
        z: Elevation field [m]
        mask: Active cell mask
        flow_frac: Output fractions (n, n, 8)
        dx: Cell spacing [m]
        p: Convergence exponent (1.0=MFD, large=D8-like)
    """
    n = z.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            for k in ti.static(range(8)):
                flow_frac[i, j, k] = 0.0
            continue

        total = ti.cast(0.0, ti.f32)
        for k in ti.static(range(8)):
            di, dj = ti.static(OFFSETS[k])
            dist = ti.static(DIAG[k]) * dx
            ni, nj = i + di, j + dj
            s_pow = ti.cast(0.0, ti.f32)
            if mask[ni, nj] == 1:
                slope = (z[i, j] - z[ni, nj]) / dist
                if slope > 0.0:
                    s_pow = ti.pow(slope, p)
            flow_frac[i, j, k] = s_pow
            total += s_pow

        for k in ti.static(range(8)):
            if total > 0.0:
                flow_frac[i, j, k] /= total
            else:
                flow_frac[i, j, k] = 0.0


def build_levels(z_np, mask_np, flow_frac_np):
    """BFS topological sort by drainage level.  Runs on CPU (numpy).

    Cells are sorted upstream-to-downstream so that when level L is processed,
    all cells at levels < L already hold today's Q_out.

    Args:
        z_np: Elevation array (n, n) [m]
        mask_np: Active cell mask (n, n)
        flow_frac_np: MFD flow fractions (n, n, 8)

    Returns:
        sorted_idx: Flat cell indices sorted by level (int32)
        level_start: level_start[L] = first index in sorted_idx at level L
        max_level: Maximum drainage level
    """
    n = z_np.shape[0]

    # 1. Compute in-degree: count upstream neighbors sending flow to each cell
    in_degree = np.zeros((n, n), dtype=np.int32)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask_np[i, j] == 0:
                continue
            for k in range(8):
                di, dj = OFFSETS[k]
                ni, nj = i + di, j + dj
                if mask_np[ni, nj] == 1 and flow_frac_np[ni, nj, OPP[k]] > 0:
                    in_degree[i, j] += 1

    # 2. BFS from ridgetops (in_degree == 0)
    level = np.full((n, n), -1, dtype=np.int32)
    queue = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask_np[i, j] == 1 and in_degree[i, j] == 0:
                level[i, j] = 0
                queue.append((i, j))

    max_level = 0
    while queue:
        next_queue = []
        for i, j in queue:
            for k in range(8):
                if flow_frac_np[i, j, k] > 0:
                    di, dj = OFFSETS[k]
                    ni, nj = i + di, j + dj
                    if mask_np[ni, nj] == 1:
                        in_degree[ni, nj] -= 1
                        if in_degree[ni, nj] == 0:
                            level[ni, nj] = level[i, j] + 1
                            if level[ni, nj] > max_level:
                                max_level = level[ni, nj]
                            next_queue.append((ni, nj))
        queue = next_queue

    # 3. Argsort cells by level, build level_start offsets
    active = (mask_np == 1) & (level >= 0)
    rows, cols = np.where(active)
    levels = level[rows, cols]
    order = np.argsort(levels, kind="stable")
    sorted_idx = (rows[order] * n + cols[order]).astype(np.int32)
    sorted_levels = levels[order]

    level_start = list(np.searchsorted(sorted_levels, np.arange(max_level + 2)))

    return sorted_idx, level_start, max_level


def prepare_levels(fields):
    """Build wavefront levels from current flow state.

    Call after compute_flow_fractions.  Stores sorted_idx and n_active
    on fields.
    """
    z_np = fields.z.to_numpy()
    mask_np = fields.mask.to_numpy()
    ff_np = fields.flow_frac.to_numpy()
    sorted_idx_np, _, _ = build_levels(z_np, mask_np, ff_np)

    buf = np.zeros(fields.n * fields.n, dtype=np.int32)
    buf[: len(sorted_idx_np)] = sorted_idx_np
    fields.sorted_idx.from_numpy(buf)
    fields.n_active = len(sorted_idx_np)


@ti.kernel
def route_wavefront(
    sorted_idx: ti.template(),
    n_active: ti.i32,
    Q_out: ti.template(),
    Q_daily: ti.template(),
    R: ti.template(),
    I_inf: ti.template(),
    z: ti.template(),
    V: ti.template(),
    flow_frac: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    n_manning: ti.f32,
    cn: ti.f32,
    alpha: ti.f32,
    k2: ti.f32,
    W0: ti.f32,
):
    """Serial wavefront water routing over all cells in topological order.

    Cells in sorted_idx are ordered upstream-to-downstream.  When cell k
    is processed, all its upslope contributors (at lower indices) already
    hold today's Q_out.  Single kernel launch per day.

    All discharge quantities are unit-width, mm-based [mm*m/day].

    Args:
        sorted_idx: Flat cell indices in topological (upstream-first) order
        n_active: Number of valid entries in sorted_idx
        Q_out: Discharge field (read/write) [mm*m/day]
        Q_daily: Cell-average discharge (write) [mm*m/day]
        R: Rainfall rate [mm/day]
        I_inf: Infiltration rate (write) [mm/day]
        z: Elevation [m]
        V: Vegetation density [g/m^2]
        flow_frac: MFD fractions (n, n, 8)
        mask: Active cell mask
        dx: Cell spacing [m]
        n_manning: Manning's roughness [s/m^(1/3)]
        cn: Manning conversion for q [mm·m/d] → h [mm]
        alpha: Infiltration capacity [1/day]
        k2: Vegetation half-saturation [g/m^2]
        W0: Bare-soil infiltration fraction [-]
    """
    ti.loop_config(serialize=True)
    for k in range(n_active):
        idx = sorted_idx[k]
        i = idx // Q_out.shape[1]
        j = idx % Q_out.shape[1]

        # Gather Q_in — upstream cells already hold today's values
        Q_in = gather_flux(Q_out, flow_frac, mask, i, j)

        # Max downslope gradient (Lambda_max, floored at 1e-4)
        slope_max = max_downslope(z, flow_frac, i, j, dx)

        # Local Picard: 5 iterations for h-I-Q_out coupling
        v = V[i, j]
        Q_o = Q_in
        Q_o_prev = Q_o
        I_val = ti.cast(0.0, ti.f32)

        for _ in ti.static(range(5)):
            Q_o_prev = Q_o
            q = (Q_in + Q_o) / 2.0  # [mm*m/day]
            h_val = ti.cast(0.0, ti.f32)
            if q > 0.0 and cn > 0.0:
                h_val = ti.pow(
                    q * n_manning / (cn * ti.sqrt(slope_max)),
                    0.6,
                )  # [mm]
            I_val = alpha * h_val * (v + k2 * W0) / (v + k2)  # [mm/day]
            Q_o = ti.max(0.0, Q_in + R[i, j] * dx - I_val * dx)

        # Average last two iterates to stabilise period-2 orbits
        # that arise when infiltration demand toggles above/below supply.
        # Converged iterations are unaffected (Q_o ≈ Q_o_prev).
        Q_o = (Q_o + Q_o_prev) / 2.0

        Q_out[i, j] = Q_o  # visible to downstream levels
        Q_daily[i, j] = (Q_in + Q_o) / 2.0  # Eq 12
        # Actual infiltration from mass balance
        I_inf[i, j] = (Q_in + R[i, j] * dx - Q_o) / dx
