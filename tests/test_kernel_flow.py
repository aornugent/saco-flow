"""Physical consistency tests for flow fractions and water routing.

Sections 1-2 of the test plan: exact values, conservation, and analytical
solutions for compute_flow_fractions and route_water.
"""

import math

import numpy as np
import pytest

from src.flow import compute_flow_fractions, route_water
from src.stencil import DIAG, OFFSETS, OPP


def _reconstruct_Q_in(Q_out, flow_frac, mask):
    """Reconstruct Q_in at each cell from neighbors' Q_out and flow_frac."""
    n = Q_out.shape[0]
    Q_in = np.zeros_like(Q_out)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            for k in range(8):
                di, dj = OFFSETS[k]
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and mask[ni, nj] == 1:
                    opp = OPP[k]
                    Q_in[i, j] += flow_frac[ni, nj, opp] * Q_out[ni, nj]
    return Q_in


# ── Section 1: Flow Fractions ────────────────────────────────────────────────


def test_flow_fractions_single_downslope(grid):
    """1.1: All flow to one cell when only one neighbor is downslope."""
    n = 5
    fields = grid(n)
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[3, 2] = 5.0  # south of [2,2]
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.0)
    frac = fields.flow_frac.to_numpy()

    # Direction 6 = S = offset (1,0)
    assert abs(frac[2, 2, 6] - 1.0) < 1e-6
    for k in range(8):
        if k != 6:
            assert abs(frac[2, 2, k]) < 1e-6


def test_flow_fractions_two_equal_neighbors(grid):
    """1.2: Equal split with p=1.0 for two identical cardinal downslope."""
    n = 5
    fields = grid(n)
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[3, 2] = 8.0  # south (cardinal, dist=dx)
    z[2, 3] = 8.0  # east  (cardinal, dist=dx)
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.0)
    frac = fields.flow_frac.to_numpy()

    assert abs(frac[2, 2, 6] - 0.5) < 1e-6  # south
    assert abs(frac[2, 2, 4] - 0.5) < 1e-6  # east


def test_flow_fractions_cardinal_vs_diagonal(grid):
    """1.3: Distance weighting differentiates cardinal from diagonal (p=1)."""
    n = 5
    fields = grid(n)
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[3, 2] = 9.0  # south (cardinal)
    z[3, 3] = 9.0  # SE    (diagonal)
    fields.z.from_numpy(z)

    dx = 1.0
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)
    frac = fields.flow_frac.to_numpy()

    # slope_S = 1/dx, slope_SE = 1/(DIAG[7]*dx)
    d_se = DIAG[7]  # 1.414
    expected_south = 1.0 / (1.0 + 1.0 / d_se)
    expected_se = (1.0 / d_se) / (1.0 + 1.0 / d_se)

    assert abs(frac[2, 2, 6] - expected_south) < 1e-5
    assert abs(frac[2, 2, 7] - expected_se) < 1e-5


def test_flow_fractions_exponent_amplifies(grid):
    """1.4: Higher p amplifies steepest path — exact ratio check."""
    n = 5
    fields = grid(n)
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[3, 2] = 9.0  # south
    z[3, 3] = 9.0  # SE
    fields.z.from_numpy(z)

    dx, p = 1.0, 5.0
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)
    frac = fields.flow_frac.to_numpy()

    d_se = DIAG[7]  # 1.414
    slope_s = 1.0 / dx
    slope_se = 1.0 / (d_se * dx)
    f_s = slope_s**p
    f_se = slope_se**p
    expected_south = f_s / (f_s + f_se)

    assert abs(frac[2, 2, 6] - expected_south) < 1e-4

    # Ratio frac_south / frac_SE = (diag_dist)^p
    ratio = frac[2, 2, 6] / frac[2, 2, 7]
    expected_ratio = d_se**p
    assert abs(ratio - expected_ratio) < 0.01


def test_flow_fractions_pit_cell(grid):
    """1.5: Pit cell (lower than all neighbors) — all fractions zero."""
    n = 5
    fields = grid(n)
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[2, 2] = 5.0  # pit
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.0)
    frac = fields.flow_frac.to_numpy()

    for k in range(8):
        assert abs(frac[2, 2, k]) < 1e-6


# ── Section 2: Water Routing ─────────────────────────────────────────────────


def _route_once(fields, dx, n_manning=0.03, cn=1.0, alpha=0.0, k2=5.0, W0=0.2):
    """Call route_water once (single global Picard sweep)."""
    route_water(
        fields.Q_out,
        fields.Q_out_new,
        fields.Q_daily,
        fields.R,
        fields.I_inf,
        fields.h,
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


def test_route_water_cell_balance(grid):
    """2.1: Cell-level water balance — I_inf from mass balance ensures residual ≈ 0."""
    n = 8
    dx = 1.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    R = np.full((n, n), 0.01, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 5.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # 20 global Picard sweeps
    for _ in range(20):
        _route_once(fields, dx, alpha=0.5, k2=5.0, W0=0.2)
        fields.swap("Q_out")

    Q_out = fields.Q_out.to_numpy()
    I_inf = fields.I_inf.to_numpy()
    mask = fields.mask.to_numpy()
    frac = fields.flow_frac.to_numpy()

    Q_in = _reconstruct_Q_in(Q_out, frac, mask)
    cell_area = dx * dx

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            residual = abs(
                Q_in[i, j] + R[i, j] * cell_area
                - I_inf[i, j] * cell_area
                - Q_out[i, j]
            )
            assert residual < 1e-4 * R[i, j] * cell_area, (
                f"Mass balance violated at ({i},{j}): residual={residual:.2e}"
            )
            # Where Q_out == 0, infiltration cannot exceed supply
            if Q_out[i, j] == 0.0:
                assert I_inf[i, j] * cell_area <= Q_in[i, j] + R[i, j] * cell_area + 1e-10


def test_route_water_zero_infiltration_analytical(grid):
    """2.2: Single-cell analytical solution with alpha=0 (no infiltration).

    Q_in=0, R=0.01, alpha=0 → I=0.
    Q_out = R*dx², Q_daily = R*dx²/2, I_inf = 0.
    """
    n = 3
    dx = 1.0
    fields = grid(n)

    z = np.array([[2, 2, 2], [2, 10, 2], [2, 9, 2]], dtype=np.float32)
    fields.z.from_numpy(z)

    R = np.full((n, n), 0.01, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # Single route_water call
    _route_once(fields, dx, alpha=0.0)

    cell_area = dx * dx
    Q_out_new = fields.Q_out_new.to_numpy()
    Q_daily = fields.Q_daily.to_numpy()
    I_inf = fields.I_inf.to_numpy()

    expected_Q_out = 0.01 * cell_area
    expected_Q_daily = 0.005 * cell_area
    expected_I_inf = 0.0

    assert abs(Q_out_new[1, 1] - expected_Q_out) < 1e-6, (
        f"Q_out: {Q_out_new[1, 1]} != {expected_Q_out}"
    )
    assert abs(Q_daily[1, 1] - expected_Q_daily) < 1e-6, (
        f"Q_daily: {Q_daily[1, 1]} != {expected_Q_daily}"
    )
    assert abs(I_inf[1, 1] - expected_I_inf) < 1e-6, (
        f"I_inf: {I_inf[1, 1]} != {expected_I_inf}"
    )


def test_route_water_picard_analytical(grid):
    """2.3: Single-cell Picard iteration matches Python reference computation.

    5×5 grid with slope=1/dx at center cell. R=0.01, V=10, alpha=1.0.
    Run single route_water call (5 internal Picard iterations).
    """
    n = 5
    dx = 1.0
    fields = grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    R_val = 0.01
    R = np.full((n, n), R_val, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    _route_once(fields, dx, n_manning=0.03, cn=1.0, alpha=1.0, k2=5.0, W0=0.2)

    # Python reference: compute Picard iteration for cell [2,2]
    # Q_in = 0 (first call, all Q_out start at 0)
    Q_in = 0.0
    V = 10.0
    alpha = 1.0
    k2 = 5.0
    W0 = 0.2
    n_manning = 0.03
    cn = 1.0
    cell_area = dx * dx

    # slope_max at [2,2]: south neighbor [3,2] z=1, slope=(2-1)/dx=1.0
    slope_max = 1.0

    Q_o = 0.0  # initial guess
    for _ in range(5):
        q = (Q_in + Q_o) / (2.0 * dx)
        if q > 0.0 and cn > 0.0:
            h_val = (q * n_manning / (cn * math.sqrt(slope_max))) ** 0.6
        else:
            h_val = 0.0
        I_val = alpha * h_val * (V + k2 * W0) / (V + k2)
        Q_o = max(0.0, Q_in + R_val * cell_area - I_val * cell_area)

    # Final recompute
    q = (Q_in + Q_o) / (2.0 * dx)
    if q > 0.0 and cn > 0.0:
        h_final = (q * n_manning / (cn * math.sqrt(slope_max))) ** 0.6
    else:
        h_final = 0.0

    expected_Q_out = Q_o
    expected_Q_daily = (Q_in + Q_o) / 2.0
    expected_I_inf = (Q_in + R_val * cell_area - Q_o) / cell_area

    Q_out_new = fields.Q_out_new.to_numpy()
    Q_daily = fields.Q_daily.to_numpy()
    I_inf = fields.I_inf.to_numpy()
    h = fields.h.to_numpy()

    assert abs(Q_out_new[2, 2] - expected_Q_out) / max(expected_Q_out, 1e-10) < 1e-4
    assert abs(Q_daily[2, 2] - expected_Q_daily) / max(expected_Q_daily, 1e-10) < 1e-4
    assert abs(I_inf[2, 2] - expected_I_inf) / max(expected_I_inf, 1e-10) < 1e-4
    assert abs(h[2, 2] - h_final) / max(h_final, 1e-10) < 1e-4


def test_route_water_infiltration_bounded(grid):
    """2.4: Infiltration cannot exceed supply (very large alpha, tiny R).

    I_inf * dx² must not exceed Q_in + R * dx² (Bug 3 guard).
    """
    n = 3
    dx = 1.0
    fields = grid(n)

    z = np.array([[2, 2, 2], [2, 10, 2], [2, 9, 2]], dtype=np.float32)
    fields.z.from_numpy(z)

    R = np.full((n, n), 0.001, dtype=np.float32)  # very small
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.zeros((n, n), dtype=np.float32))  # bare soil

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    _route_once(fields, dx, alpha=100.0, k2=5.0, W0=0.2)

    cell_area = dx * dx
    I_inf = fields.I_inf.to_numpy()
    R_arr = 0.001

    # I_inf * dx² ≤ R * dx² (since Q_in = 0)
    assert I_inf[1, 1] * cell_area <= R_arr * cell_area + 1e-10, (
        f"Infiltration exceeds supply: I*dx²={I_inf[1, 1] * cell_area:.6f}, "
        f"R*dx²={R_arr * cell_area:.6f}"
    )


def test_route_water_global_conservation(grid):
    """2.5: Global conservation — supply = infiltration + boundary outflow."""
    n = 16
    dx = 1.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    R_val = 0.01
    R = np.full((n, n), R_val, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 5.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    for _ in range(20):
        _route_once(fields, dx, alpha=0.5, k2=5.0, W0=0.2)
        fields.swap("Q_out")

    Q_out = fields.Q_out.to_numpy()
    I_inf = fields.I_inf.to_numpy()
    mask = fields.mask.to_numpy()
    frac = fields.flow_frac.to_numpy()
    cell_area = dx * dx

    Q_in = _reconstruct_Q_in(Q_out, frac, mask)
    interior = mask == 1

    total_supply = np.sum(R[interior]) * cell_area
    total_infiltr = np.sum(I_inf[interior]) * cell_area
    total_Q_boundary = np.sum(Q_out[interior]) - np.sum(Q_in[interior])

    balance = abs(total_supply - total_infiltr - total_Q_boundary)
    assert balance < 1e-3 * total_supply, (
        f"Global conservation violated: balance={balance:.2e}, "
        f"supply={total_supply:.4f}, infil={total_infiltr:.4f}, "
        f"Q_bndry={total_Q_boundary:.4f}"
    )


def test_route_water_Q_daily_consistency(grid):
    """2.6: Q_daily = (Q_in + Q_out) / 2 for every interior cell."""
    n = 16
    dx = 1.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    R = np.full((n, n), 0.01, dtype=np.float32)
    fields.R.from_numpy(R)
    fields.V.from_numpy(np.full((n, n), 5.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    for _ in range(20):
        _route_once(fields, dx, alpha=0.5, k2=5.0, W0=0.2)
        fields.swap("Q_out")

    Q_out = fields.Q_out.to_numpy()
    Q_daily = fields.Q_daily.to_numpy()
    mask = fields.mask.to_numpy()
    frac = fields.flow_frac.to_numpy()

    Q_in = _reconstruct_Q_in(Q_out, frac, mask)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            expected = (Q_in[i, j] + Q_out[i, j]) / 2.0
            assert abs(Q_daily[i, j] - expected) < 1e-5, (
                f"Q_daily mismatch at ({i},{j}): "
                f"{Q_daily[i, j]:.6f} vs {expected:.6f}"
            )
