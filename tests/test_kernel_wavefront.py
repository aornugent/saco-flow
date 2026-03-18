"""Tests for wavefront flow routing correctness.

Verification tests from the wavefront routing spec:
  - Equivalence to serial (Python) upstream-to-downstream sweep
  - Mass balance: boundary outflow ≈ supply - infiltration
"""

import math

import numpy as np

from src.flow import (
    build_levels,
    compute_flow_fractions,
    prepare_levels,
    route_wavefront,
)
from src.stencil import DIAG, OFFSETS, OPP


def _serial_route(
    z_np,
    mask_np,
    flow_frac_np,
    R_np,
    V_np,
    sorted_idx,
    dx,
    n_manning,
    cn,
    alpha,
    k2,
    W0,
):
    """Pure-Python serial sweep: process cells in sorted_idx order.

    Same gather + Newton-on-quintic as the kernel, but executed one cell
    at a time in strict upstream-to-downstream order.
    Returns Q_out, Q_daily, I_inf.
    """
    n = z_np.shape[0]
    Q_out = np.zeros((n, n), dtype=np.float64)
    Q_daily = np.zeros((n, n), dtype=np.float64)
    I_inf = np.zeros((n, n), dtype=np.float64)

    for idx in sorted_idx:
        i = idx // n
        j = idx % n

        # Gather Q_in from upstream neighbors
        Q_in = 0.0
        for k in range(8):
            di, dj = OFFSETS[k]
            ni, nj = i + di, j + dj
            if mask_np[ni, nj] == 1:
                opp = OPP[k]
                Q_in += flow_frac_np[ni, nj, opp] * Q_out[ni, nj]

        # Max downslope gradient
        slope_max = 1e-4
        for k in range(8):
            if flow_frac_np[i, j, k] > 0.0:
                di, dj = OFFSETS[k]
                dist = DIAG[k] * dx
                ni, nj = i + di, j + dj
                slope_k = (z_np[i, j] - z_np[ni, nj]) / dist
                slope_max = max(slope_max, slope_k)

        # Newton-Raphson on quintic: x^5 + C_I*x^3 - K = 0
        v = float(V_np[i, j])
        Q_max = Q_in + float(R_np[i, j]) * dx
        K = Q_in + Q_max

        C_I = 0.0
        if cn > 0.0 and alpha > 0.0:
            manning_ratio = n_manning / (2.0 * cn * math.sqrt(slope_max))
            C_I = alpha * manning_ratio**0.6 * (v + k2 * W0) / (v + k2) * dx

        if K <= 0.0:
            Q_o = 0.0
        elif C_I < 1e-12:
            Q_o = Q_max
        else:
            x = min(K**0.2, (K / C_I) ** (1.0 / 3.0))
            for _ in range(8):
                x2 = x * x
                x3 = x2 * x
                f = x3 * x2 + C_I * x3 - K
                fp = 5.0 * x2 * x2 + 3.0 * C_I * x2
                x -= f / fp
                x = max(x, 0.0)
            S = x**5
            Q_o = max(0.0, S - Q_in)
            Q_o = min(Q_o, Q_max)

        Q_out[i, j] = Q_o
        Q_daily[i, j] = (Q_in + Q_o) / 2.0
        I_inf[i, j] = (Q_in + float(R_np[i, j]) * dx - Q_o) / dx

    return Q_out, Q_daily, I_inf


def test_wavefront_matches_serial_sweep(grid):
    """Wavefront kernel matches cell-by-cell serial Python sweep.

    On a 20×20 planar slope, run the wavefront kernel and a plain Python
    loop that processes cells in sorted_idx order with the same gather +
    Newton solver.  Q_out must match within float32 tolerance.
    """
    n = 20
    dx = 5.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.07 * dx
    fields.z.from_numpy(z)

    R_mm = 10.0
    R = np.full((n, n), R_mm, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 8.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 2.0)
    prepare_levels(fields)

    n_manning, cn, alpha, k2, W0 = 0.05, 1.0, 8.0, 18.0, 0.05

    # Run wavefront kernel
    route_wavefront(
        fields.sorted_idx,
        fields.n_active,
        fields.Q_out,
        fields.Q_daily,
        fields.R,
        fields.I_inf,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        dx,
        n_manning,
        cn,
        alpha,
        k2,
        W0,
    )

    Q_out_kernel = fields.Q_out.to_numpy()
    Q_daily_kernel = fields.Q_daily.to_numpy()
    I_inf_kernel = fields.I_inf.to_numpy()

    # Run serial Python sweep
    z_np = fields.z.to_numpy()
    mask_np = fields.mask.to_numpy()
    ff_np = fields.flow_frac.to_numpy()
    R_np = fields.R.to_numpy()
    V_np = fields.V.to_numpy()
    sorted_idx_np, _, _ = build_levels(z_np, mask_np, ff_np)

    Q_out_ref, Q_daily_ref, I_inf_ref = _serial_route(
        z_np,
        mask_np,
        ff_np,
        R_np,
        V_np,
        sorted_idx_np,
        dx,
        n_manning,
        cn,
        alpha,
        k2,
        W0,
    )

    # Compare cell-by-cell over interior
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask_np[i, j] == 0:
                continue
            assert abs(Q_out_kernel[i, j] - Q_out_ref[i, j]) < 1e-3, (
                f"Q_out mismatch at ({i},{j}): "
                f"kernel={Q_out_kernel[i, j]:.6f}, ref={Q_out_ref[i, j]:.6f}"
            )
            assert abs(Q_daily_kernel[i, j] - Q_daily_ref[i, j]) < 1e-3, (
                f"Q_daily mismatch at ({i},{j}): "
                f"kernel={Q_daily_kernel[i, j]:.6f}, ref={Q_daily_ref[i, j]:.6f}"
            )
            assert abs(I_inf_kernel[i, j] - I_inf_ref[i, j]) < 1e-3, (
                f"I_inf mismatch at ({i},{j}): "
                f"kernel={I_inf_kernel[i, j]:.6f}, ref={I_inf_ref[i, j]:.6f}"
            )


def test_wavefront_mass_balance(grid):
    """Global mass balance: boundary outflow ≈ supply - infiltration.

    After routing on a 20×20 slope:
    sum(Q_out * flow_frac toward boundary) ≈ sum(R*dx²) - sum(I_inf*dx²)
    Relative error < 1e-4.
    """
    n = 20
    dx = 5.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.07 * dx
    fields.z.from_numpy(z)

    R_mm = 10.0
    R = np.full((n, n), R_mm, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 8.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 2.0)
    prepare_levels(fields)

    n_manning, cn, alpha, k2, W0 = 0.05, 1.0, 8.0, 18.0, 0.05

    route_wavefront(
        fields.sorted_idx,
        fields.n_active,
        fields.Q_out,
        fields.Q_daily,
        fields.R,
        fields.I_inf,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        dx,
        n_manning,
        cn,
        alpha,
        k2,
        W0,
    )

    Q_out = fields.Q_out.to_numpy()
    I_inf = fields.I_inf.to_numpy()
    mask = fields.mask.to_numpy()
    frac = fields.flow_frac.to_numpy()
    interior = mask == 1

    # Boundary outflow: sum of Q_out * flow_frac directed toward boundary cells
    boundary_outflow = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            for k in range(8):
                if frac[i, j, k] > 0:
                    di, dj = OFFSETS[k]
                    ni, nj = i + di, j + dj
                    if mask[ni, nj] == 0:  # boundary
                        boundary_outflow += frac[i, j, k] * Q_out[i, j]

    # Per-cell balance: Q_in + R*dx - I*dx - Q_out = 0 (unit-width)
    # Global: sum(R*dx) - sum(I*dx) = sum(Q_out) - sum(Q_in)
    #       = net outflow through boundary
    total_supply_uw = np.sum(R[interior]) * dx
    total_infiltr_uw = np.sum(I_inf[interior]) * dx

    # Net boundary outflow = sum(Q_out) - sum(Q_in) over interior
    # Reconstruct Q_in
    Q_in_total = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            for k in range(8):
                di, dj = OFFSETS[k]
                ni, nj = i + di, j + dj
                if mask[ni, nj] == 1:
                    opp = OPP[k]
                    Q_in_total += frac[ni, nj, opp] * Q_out[ni, nj]

    net_outflow = np.sum(Q_out[interior]) - Q_in_total
    balance = total_supply_uw - total_infiltr_uw - net_outflow

    rel_error = abs(balance) / total_supply_uw
    assert rel_error < 1e-4, (
        f"Mass balance violated: rel_error={rel_error:.2e}, "
        f"supply={total_supply_uw:.4f}, infil={total_infiltr_uw:.4f}, "
        f"net_outflow={net_outflow:.4f}, balance={balance:.6f}"
    )
