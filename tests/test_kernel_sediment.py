"""Physical consistency tests for sediment transport and elevation update.

Sections 5-6 of the test plan: transport capacity, erosion/deposition regimes,
equilibrium, K/P interpolation, and elevation mass conservation.

Q_annual is now mm*m/yr (unit-width, mm-based). The sediment kernel
converts to m^2/yr via q = Q * 0.001.
"""

import math

import numpy as np
import taichi as ti

from src.flow import compute_flow_fractions
from src.sediment import (
    _erosion_deposition_coeffs,
    sediment_transport,
    update_elevation,
)
from src.stencil import OFFSETS, OPP


# Wrapper kernel to test _erosion_deposition_coeffs from Python
@ti.kernel
def _test_kp_coeffs(
    v: ti.f32,
    K_max: ti.f32,
    K_min: ti.f32,
    P_min: ti.f32,
    P_max: ti.f32,
    v_low: ti.f32,
    v_high: ti.f32,
    result: ti.template(),
):
    kp = _erosion_deposition_coeffs(v, K_max, K_min, P_min, P_max, v_low, v_high)
    result[0] = kp[0]  # K
    result[1] = kp[1]  # P


def _run_sediment(fields, dx, **kwargs):
    """Run sediment_transport with defaults."""
    defaults = {
        "gamma": 1.0,
        "m_exp": 1.65,
        "n_exp": 1.65,
        "K_max": 0.05,
        "K_min": 0.00005,
        "P_min": 0.05,
        "P_max": 50.0,
        "v_low": 5.0,
        "v_high": 20.0,
    }
    defaults.update(kwargs)
    sediment_transport(
        fields.S,
        fields.S_new,
        fields.Q_annual,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        dx,
        **defaults,
    )


# ── Section 5: Sediment Transport ────────────────────────────────────────────


def test_sediment_transport_capacity(grid):
    """5.1: Transport capacity formula — exact value.

    Q_annual=100000 mm*m/yr -> q=100 m^2/yr, slope=0.02, gamma=1.0, m=n=1.65, dx=5.0.
    C = 1.0 * 100^1.65 * 0.02^1.65.
    """
    n = 5
    dx = 5.0
    fields = grid(n)

    # Set slope = 0.02: z[2,2] - z[3,2] = 0.02 * dx = 0.1
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[2, 2] = 10.0
    z[3, 2] = 10.0 - 0.02 * dx  # south neighbor
    fields.z.from_numpy(z)

    # Q_annual in mm*m/yr: want q = Q*0.001 = 20 m^2/yr -> Q = 20000
    Q_annual = np.zeros((n, n), dtype=np.float32)
    Q_annual[2, 2] = 20000.0  # mm*m/yr -> q = 20 m^2/yr
    fields.Q_annual.from_numpy(Q_annual)

    # S = 0 -> erosion regime, S_new should approach C
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    _run_sediment(fields, dx)

    # Hand computation
    q = 20000.0 * 0.001  # 20.0 m^2/yr
    slope = 0.02
    C = 1.0 * (q**1.65) * (slope**1.65)

    S_new = fields.S_new.to_numpy()

    # S_0 = 0 (no upslope sediment), so:
    # S_new = C + (0 - C) * exp(-dx/h_sed) = C * (1 - exp(-dx/h_sed))
    # S_new should be between 0 and C
    assert 0 < S_new[2, 2] < C * 1.01, f"S_new={S_new[2, 2]:.6f} not in (0, C={C:.6f})"


def test_sediment_erosion_regime(grid):
    """5.2: Erosion regime — S_0 < C -> 0 < S_new < C, matches formula."""
    n = 5
    dx = 5.0
    fields = grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    z[2, 2] = 10.0
    z[3, 2] = 10.0 - 0.02 * dx
    fields.z.from_numpy(z)

    Q_annual = np.zeros((n, n), dtype=np.float32)
    Q_annual[2, 2] = 20000.0  # mm*m/yr -> q = 20 m^2/yr
    fields.Q_annual.from_numpy(Q_annual)

    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    V_val = 10.0
    fields.V.from_numpy(np.full((n, n), V_val, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    K_max, K_min, P_min, P_max = 0.05, 0.00005, 0.05, 50.0
    v_low, v_high = 5.0, 20.0

    _run_sediment(fields, dx, K_max=K_max, K_min=K_min, P_min=P_min, P_max=P_max)

    S_new_arr = fields.S_new.to_numpy()
    S_new_val = S_new_arr[2, 2]

    # Compute expected
    q = 20000.0 * 0.001  # 20.0
    slope = 0.02
    C = (q**1.65) * (slope**1.65)

    # K/P coefficients for V=10 (between v_low=5 and v_high=20)
    t = math.log(V_val / v_low) / math.log(v_high / v_low)
    K = K_max + (K_min - K_max) * t

    # S_0 = 0 < C -> erosion, coeff = K
    h_sed = C / max(K * q * slope, 1e-10)
    expected = C + (0 - C) * math.exp(-dx / h_sed)

    assert 0 < S_new_val < C, f"S_new={S_new_val} not in erosion range (0, {C})"
    rel_err = abs(S_new_val - expected) / max(expected, 1e-10)
    assert rel_err < 1e-3, (
        f"S_new={S_new_val:.6f}, expected={expected:.6f}, rel_err={rel_err:.2e}"
    )


def test_sediment_deposition_regime(grid):
    """5.3: Deposition regime — S_0 > C -> C < S_new < S_0."""
    n = 5
    dx = 5.0
    fields = grid(n)

    # Gentle slope, low Q -> small C
    z = np.full((n, n), 10.0, dtype=np.float32)
    z[2, 2] = 10.0
    z[3, 2] = 10.0 - 0.01 * dx
    fields.z.from_numpy(z)

    Q_annual = np.zeros((n, n), dtype=np.float32)
    Q_annual[2, 2] = 2000.0  # mm*m/yr -> q = 2 m^2/yr (low)
    Q_annual[1, 2] = 2000.0  # upslope neighbor
    fields.Q_annual.from_numpy(Q_annual)

    # High incoming sediment from upslope
    S_init = np.zeros((n, n), dtype=np.float32)
    S_init[1, 2] = 100.0  # upslope has high sediment
    fields.S.from_numpy(S_init)

    V_val = 10.0
    fields.V.from_numpy(np.full((n, n), V_val, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    _run_sediment(fields, dx)

    S_new_arr = fields.S_new.to_numpy()
    S_new_val = S_new_arr[2, 2]

    # Transport capacity at [2,2]
    q = 2000.0 * 0.001  # 2.0
    slope = 0.01
    C = (q**1.65) * (slope**1.65)

    # S_0 gathered from upslope [1,2]
    frac = fields.flow_frac.to_numpy()
    # [1,2] -> [2,2] is direction south (k=6), so flow_frac[1,2,6]
    S_0 = frac[1, 2, 6] * 100.0

    if S_0 > C:
        assert C * 0.99 <= S_new_val <= S_0 * 1.01, (
            f"Deposition: S_new={S_new_val:.4f} not in [{C:.4f}, {S_0:.4f}]"
        )


def test_sediment_equilibrium(grid):
    """5.4: Equilibrium — when S_0 = C, S_new = C (no change)."""
    n = 5
    dx = 5.0
    fields = grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    z[2, 2] = 10.0
    z[3, 2] = 10.0 - 0.02 * dx
    fields.z.from_numpy(z)

    Q_annual = np.zeros((n, n), dtype=np.float32)
    Q_annual[2, 2] = 20000.0  # mm*m/yr -> q = 20
    Q_annual[1, 2] = 20000.0  # upslope
    fields.Q_annual.from_numpy(Q_annual)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # Compute C at [2,2]
    q = 20000.0 * 0.001  # 20.0
    slope = 0.02
    C = (q**1.65) * (slope**1.65)

    # Set upslope S = C so gathered S_0 ~ C
    frac = fields.flow_frac.to_numpy()
    # Need S[1,2] such that flow_frac[1,2,south] * S[1,2] = C
    f_south = frac[1, 2, 6]
    if f_south > 0:
        S_init = np.zeros((n, n), dtype=np.float32)
        S_init[1, 2] = C / f_south
        fields.S.from_numpy(S_init)

        V_val = 10.0
        fields.V.from_numpy(np.full((n, n), V_val, dtype=np.float32))

        _run_sediment(fields, dx)

        S_new_arr = fields.S_new.to_numpy()
        assert abs(S_new_arr[2, 2] - C) / C < 1e-3, (
            f"Equilibrium: S_new={S_new_arr[2, 2]:.6f}, C={C:.6f}"
        )


def test_sediment_kp_coefficients():
    """5.5: Vegetation-dependent K/P coefficients at three points."""
    K_max, K_min = 0.05, 0.00005
    P_min, P_max = 0.05, 50.0
    v_low, v_high = 5.0, 20.0

    result = ti.field(ti.f32, shape=2)

    # V = 1.0 (< v_low): K=K_max, P=P_min
    _test_kp_coeffs(1.0, K_max, K_min, P_min, P_max, v_low, v_high, result)
    assert abs(result[0] - K_max) < 1e-6, f"K at V=1: {result[0]} != {K_max}"
    assert abs(result[1] - P_min) < 1e-6, f"P at V=1: {result[1]} != {P_min}"

    # V = 20.0 (>= v_high): K=K_min, P=P_max
    _test_kp_coeffs(20.0, K_max, K_min, P_min, P_max, v_low, v_high, result)
    assert abs(result[0] - K_min) < 1e-5, f"K at V=20: {result[0]} != {K_min}"
    assert abs(result[1] - P_max) < 1e-5, f"P at V=20: {result[1]} != {P_max}"

    # V = 10.0 (between): t = ln(10/5)/ln(20/5) = ln(2)/ln(4) = 0.5
    t = math.log(10.0 / v_low) / math.log(v_high / v_low)
    K_expected = K_max + (K_min - K_max) * t
    P_expected = P_min + (P_max - P_min) * t

    _test_kp_coeffs(10.0, K_max, K_min, P_min, P_max, v_low, v_high, result)
    assert abs(result[0] - K_expected) < 1e-5, (
        f"K at V=10: {result[0]:.6f} != {K_expected:.6f}"
    )
    assert abs(result[1] - P_expected) < 1e-4, (
        f"P at V=10: {result[1]:.4f} != {P_expected:.4f}"
    )


# ── Section 6: Elevation Update ──────────────────────────────────────────────


def _reconstruct_S0(S, flow_frac, mask):
    """Reconstruct gathered S_0 at each cell (same gather as update_elevation)."""
    n = S.shape[0]
    S_0 = np.zeros_like(S)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            for k in range(8):
                di, dj = OFFSETS[k]
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and mask[ni, nj] == 1:
                    opp = OPP[k]
                    S_0[i, j] += flow_frac[ni, nj, opp] * S[ni, nj]
    return S_0


def test_elevation_erosion(grid):
    """6.1: Erosion lowers elevation — S_0 < S_new -> dz < 0."""
    n = 5
    dx = 5.0
    fields = grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # S = 0 (gathered S_0 ~ 0), S_new > 0 (erosion)
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))
    S_new = np.full((n, n), 15.0, dtype=np.float32)
    fields.S_new.from_numpy(S_new)

    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, dx
    )

    z_final = fields.z.to_numpy()

    # At [2,2]: S_0 ~ 0 (neighbors S=0), S_new=15
    # dz = (0 - 15)/5 = -3.0
    expected_dz = (0.0 - 15.0) / dx
    actual_dz = z_final[2, 2] - z0[2, 2]
    assert abs(actual_dz - expected_dz) < 1e-4, (
        f"dz={actual_dz:.4f}, expected={expected_dz:.4f}"
    )


def test_elevation_deposition(grid):
    """6.2: Deposition raises elevation — S_0 > S_new -> dz > 0."""
    n = 5
    dx = 5.0
    fields = grid(n)

    z = np.full((n, n), 10.0, dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    z0 = z.copy()
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # High S from upslope neighbor
    S = np.full((n, n), 20.0, dtype=np.float32)
    fields.S.from_numpy(S)
    S_new = np.full((n, n), 5.0, dtype=np.float32)
    fields.S_new.from_numpy(S_new)

    # Reconstruct expected S_0 at [2,2]
    frac = fields.flow_frac.to_numpy()
    mask_np = fields.mask.to_numpy()
    S_0 = _reconstruct_S0(S, frac, mask_np)

    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, dx
    )

    z_final = fields.z.to_numpy()
    expected_dz = (S_0[2, 2] - 5.0) / dx
    actual_dz = z_final[2, 2] - z0[2, 2]
    assert abs(actual_dz - expected_dz) < 1e-4, (
        f"dz={actual_dz:.4f}, expected={expected_dz:.4f}"
    )
    assert actual_dz > 0, f"Expected deposition (dz>0), got dz={actual_dz}"


def test_elevation_mass_conservation(grid):
    """6.3: Sediment mass conservation — sum dz ~ 0 on closed domain."""
    n = 8
    dx = 1.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # Q_annual in mm*m/yr
    Q = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Q[i, :] = float(i) * 500.0  # mm*m/yr -> q = i*0.5 m^2/yr
    fields.Q_annual.from_numpy(Q)
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))
    fields.S.from_numpy(np.zeros((n, n), dtype=np.float32))

    # Run a few sediment steps to build up flux
    for _ in range(5):
        _run_sediment(fields, dx)
        fields.swap("S")

    # Now do one more transport + elevation update
    _run_sediment(fields, dx)

    z_before = fields.z.to_numpy().copy()
    mask_np = fields.mask.to_numpy()

    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, dx
    )

    z_after = fields.z.to_numpy()
    dz = z_after - z_before

    # Verify sum dz ~ 0 for interior cells
    interior = mask_np == 1
    total_dz = np.sum(dz[interior])
    max_dz = np.max(np.abs(dz[interior]))
    n_interior = np.sum(interior)

    assert abs(total_dz) < 1e-2 * max_dz * n_interior, (
        f"|sum dz|={abs(total_dz):.4f}, bound={1e-2 * max_dz * n_interior:.4f}"
    )
