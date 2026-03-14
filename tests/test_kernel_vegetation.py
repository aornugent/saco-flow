"""Physical consistency tests for vegetation dynamics.

Section 4 of the test plan: exact Euler steps, diffusion, flow dispersal,
conservation, steady state, and clamp correctness.

Q_daily is unit-width [mm*m/day] — used directly, no /dx*1000 conversion.
"""

import numpy as np

from src.flow import compute_flow_fractions
from src.soil_moisture import soil_moisture_step
from src.vegetation import vegetation_step


def _flat_veg_step(fields, n, **kwargs):
    """Run vegetation_step with flat terrain (no flow dispersal)."""
    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(
        fields.z, fields.mask, fields.flow_frac, kwargs.get("dx", 1.0), 1.0
    )
    fields.Q_daily.from_numpy(np.zeros((n, n), dtype=np.float32))
    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.Q_annual,
        fields.flow_frac,
        fields.mask,
        **kwargs,
    )


def test_vegetation_growth_mortality_exact(grid):
    """4.1: Single-cell growth/mortality exact Euler step (no dispersal).

    V=10, M=5, c=10, g_max=0.05, k1=5, d=0.13, dt=1.
    growth = 10*0.05*5/(5+5)*10 = 2.5
    mortality = 0.13*10 = 1.3
    V_new = 10 + 1*(2.5 - 1.3) = 11.2
    """
    n = 5
    fields = grid(n)

    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), 5.0, dtype=np.float32))

    _flat_veg_step(
        fields,
        n,
        c=10.0,
        g_max=0.05,
        k1=5.0,
        d=0.13,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )

    V_new = fields.V_new.to_numpy()
    assert abs(V_new[2, 2] - 11.2) < 1e-4, f"V_new={V_new[2, 2]:.6f}, expected=11.2"


def test_vegetation_mortality_only_exact(grid):
    """4.2: Mortality-only exact decay (M=0 -> no growth).

    V=20, d=0.13, dt=1. V_new = 20 - 2.6 = 17.4.
    """
    n = 5
    fields = grid(n)

    fields.V.from_numpy(np.full((n, n), 20.0, dtype=np.float32))
    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))

    _flat_veg_step(
        fields,
        n,
        c=10.0,
        g_max=0.05,
        k1=5.0,
        d=0.13,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )

    V_new = fields.V_new.to_numpy()
    assert abs(V_new[2, 2] - 17.4) < 1e-5, f"V_new={V_new[2, 2]:.6f}, expected=17.4"


def test_vegetation_steady_state(grid):
    """4.3: Vegetation equilibrium — growth = mortality.

    At equilibrium: c*g_max*M*/(M*+k1) = d.
    M* = d*k1/(c*g_max - d) = 0.13*5/(0.5-0.13) = 1.7568 mm.
    """
    n = 5
    fields = grid(n)

    c, g_max, k1, d = 10.0, 0.05, 5.0, 0.13
    rw = 0.19

    M_star = d * k1 / (c * g_max - d)  # ~ 1.7568

    # Compute I_inf that sustains M* in soil moisture:
    # I = g_max * M*/(M*+k1) * V_eq + rw * M*
    V_eq = 10.0
    I_sustain = g_max * M_star / (M_star + k1) * V_eq + rw * M_star

    fields.V.from_numpy(np.full((n, n), V_eq, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), M_star, dtype=np.float32))
    fields.I_inf.from_numpy(np.full((n, n), I_sustain, dtype=np.float32))

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.0)
    fields.Q_daily.from_numpy(np.zeros((n, n), dtype=np.float32))

    # Run coupled soil + vegetation for 2000 steps
    for _ in range(2000):
        # Refill I_inf each step (constant forcing)
        fields.I_inf.from_numpy(np.full((n, n), I_sustain, dtype=np.float32))
        soil_moisture_step(
            fields.M,
            fields.I_inf,
            fields.V,
            fields.mask,
            g_max=g_max,
            k1=k1,
            rw=rw,
            dt=1.0,
        )
        vegetation_step(
            fields.V,
            fields.V_new,
            fields.M,
            fields.Q_daily,
            fields.Q_annual,
            fields.flow_frac,
            fields.mask,
            c=c,
            g_max=g_max,
            k1=k1,
            d=d,
            Dp=0.0,
            dx=1.0,
            c1=0.0,
            c2=1.0,
            dt=1.0,
        )
        fields.swap("V")

    V_final = fields.V.to_numpy()
    V_val = V_final[2, 2]

    # V should be stable (not collapsed or exploded)
    assert V_val > 0.5, f"V collapsed to {V_val:.4f}"
    assert V_val < 200.0, f"V exploded to {V_val:.4f}"

    # Check dV/dt is small: run one more step and check change
    V_before = V_val
    fields.I_inf.from_numpy(np.full((n, n), I_sustain, dtype=np.float32))
    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=g_max,
        k1=k1,
        rw=rw,
        dt=1.0,
    )
    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.Q_annual,
        fields.flow_frac,
        fields.mask,
        c=c,
        g_max=g_max,
        k1=k1,
        d=d,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )
    V_after = fields.V_new.to_numpy()[2, 2]
    dV = abs(V_after - V_before)
    assert dV < 0.01 * V_before, f"|dV|={dV:.6f} > 1% of V={V_before:.4f}"


def test_vegetation_isotropic_diffusion_exact(grid):
    """4.4: Exact Laplacian on known pattern — center cell.

    5x5 grid. V[2,2]=20, cardinal neighbors V=10. Dp=1.0, dx=1.0, dt=0.1.
    laplacian = 4*(10-20) = -40. coeff_iso = 0.1.
    V_new = 20 + 0.1*(-40) = 16.0.
    """
    n = 5
    fields = grid(n)

    V = np.full((n, n), 10.0, dtype=np.float32)
    V[2, 2] = 20.0
    fields.V.from_numpy(V)
    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))  # no growth

    _flat_veg_step(
        fields,
        n,
        c=0.0,
        g_max=0.0,
        k1=5.0,
        d=0.0,
        Dp=1.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=0.1,
    )

    V_new = fields.V_new.to_numpy()
    assert abs(V_new[2, 2] - 16.0) < 1e-5, f"V_new={V_new[2, 2]:.6f}, expected=16.0"


def test_vegetation_isotropic_diffusion_conserves(grid):
    """4.5: Isotropic diffusion conserves total vegetation.

    8x8 grid, random V, c=d=c1=0, Dp=0.5, dt=0.05. 100 steps.
    |sum V_final - sum V_initial| < 1e-3 * sum V_initial.
    """
    n = 8
    fields = grid(n)

    rng = np.random.default_rng(123)
    V0 = rng.uniform(1.0, 20.0, (n, n)).astype(np.float32)
    V0[0, :] = V0[-1, :] = V0[:, 0] = V0[:, -1] = 0.0
    fields.V.from_numpy(V0)
    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))

    z = np.full((n, n), 10.0, dtype=np.float32)
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.0)
    fields.Q_daily.from_numpy(np.zeros((n, n), dtype=np.float32))

    mask = fields.mask.to_numpy()
    total_initial = np.sum(V0[mask == 1])

    for _ in range(100):
        vegetation_step(
            fields.V,
            fields.V_new,
            fields.M,
            fields.Q_daily,
            fields.Q_annual,
            fields.flow_frac,
            fields.mask,
            c=0.0,
            g_max=0.0,
            k1=5.0,
            d=0.0,
            Dp=0.5,
            dx=1.0,
            c1=0.0,
            c2=1.0,
            dt=0.05,
        )
        fields.swap("V")

    V_final = fields.V.to_numpy()
    total_final = np.sum(V_final[mask == 1])

    rel_error = abs(total_final - total_initial) / total_initial
    assert rel_error < 1e-3, (
        f"Diffusion not conserving: initial={total_initial:.4f}, "
        f"final={total_final:.4f}, rel_error={rel_error:.2e}"
    )


def test_vegetation_flow_dispersal_exact(grid):
    """4.6: Flow-directed dispersal single-cell exact value.

    5x5 grid, linear slope. Cell [2,2]: V=5, Q_daily=60 mm*m/day.
    Upslope [1,2]: V=10, Q_daily=100 mm*m/day, flow_frac[1,2,south]=1.0.
    c1=0.005, c2=0.0005, dx=5.0.

    seed_in = 1.0 * min(0.005*100*10, 0.0005*10) = 0.005
    seed_out = min(0.005*60*5, 0.0005*5) = 0.0025
    d_flow = 0.0025
    V_new = 5.0 + (0.0025/5.0) = 5.0005
    """
    n = 5
    dx = 5.0
    fields = grid(n)

    # Set z so [1,2] has exactly one downslope interior neighbor: [2,2] (south)
    z = np.full((n, n), 20.0, dtype=np.float32)
    z[1, 2] = 15.0
    z[2, 2] = 10.0
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    # Verify flow_frac[1,2,south] = 1.0 (only downslope neighbor)
    frac = fields.flow_frac.to_numpy()
    assert frac[1, 2, 6] > 0.99, f"Expected south frac ~ 1.0, got {frac[1, 2, 6]:.4f}"

    V = np.full((n, n), 5.0, dtype=np.float32)
    V[1, 2] = 10.0  # upslope neighbor
    fields.V.from_numpy(V)

    # Q_daily in mm*m/day (unit-width, no conversion needed)
    # Old: Q_daily=0.5 m^3/day -> q=0.5/5*1000=100 mm*m/day
    # New: Q_daily=100 mm*m/day directly
    Q_daily = np.zeros((n, n), dtype=np.float32)
    Q_daily[1, 2] = 100.0  # mm*m/day
    Q_daily[2, 2] = 60.0  # mm*m/day
    fields.Q_daily.from_numpy(Q_daily)

    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))  # no growth

    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.Q_annual,
        fields.flow_frac,
        fields.mask,
        c=0.0,
        g_max=0.0,
        k1=5.0,
        d=0.0,
        Dp=0.0,
        dx=dx,
        c1=0.005,
        c2=0.0005,
        dt=1.0,
    )

    V_new = fields.V_new.to_numpy()

    # Hand computation — Q_daily is already mm*m/day
    q_neighbor = 100.0  # mm*m/day
    seed_in = 1.0 * min(
        0.005 * q_neighbor * 10.0, 0.0005 * 10.0
    )  # min(5.0, 0.005) = 0.005
    q_self = 60.0  # mm*m/day
    seed_out = min(0.005 * q_self * 5.0, 0.0005 * 5.0)  # min(1.5, 0.0025) = 0.0025
    d_flow = seed_in - seed_out  # 0.0025
    expected = 5.0 + 1.0 * (d_flow / dx)  # 5.0 + 0.0005 = 5.0005

    assert abs(V_new[2, 2] - expected) < 1e-5, (
        f"V_new={V_new[2, 2]:.6f}, expected={expected:.6f}"
    )


def test_vegetation_flow_dispersal_conserves(grid):
    """4.7: Flow dispersal conserves vegetation on a closed domain.

    8x8 grid, linear slope, uniform V=10, nonzero Q_daily.
    c=d=0, Dp=0, only flow dispersal active. 50 steps.
    """
    n = 12  # Larger grid to reduce boundary effects
    dx = 1.0
    fields = grid(n)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i)
    fields.z.from_numpy(z)

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 1.0)

    V0 = np.full((n, n), 10.0, dtype=np.float32)
    V0[0, :] = V0[-1, :] = V0[:, 0] = V0[:, -1] = 0.0
    fields.V.from_numpy(V0)
    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))

    # Set Q_daily in mm*m/day (moderate values)
    Q_daily = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n - 1):
        Q_daily[i, :] = float(i) * 10.0  # mm*m/day
    fields.Q_daily.from_numpy(Q_daily)

    mask = fields.mask.to_numpy()
    total_initial = np.sum(V0[mask == 1])

    for _ in range(50):
        vegetation_step(
            fields.V,
            fields.V_new,
            fields.M,
            fields.Q_daily,
            fields.Q_annual,
            fields.flow_frac,
            fields.mask,
            c=0.0,
            g_max=0.0,
            k1=5.0,
            d=0.0,
            Dp=0.0,
            dx=dx,
            c1=0.005,
            c2=0.0005,
            dt=1.0,
        )
        fields.swap("V")

    V_final = fields.V.to_numpy()
    total_final = np.sum(V_final[mask == 1])

    # Allow some leakage at boundaries
    rel_error = abs(total_final - total_initial) / total_initial
    assert rel_error < 5e-2, (
        f"Flow dispersal not conserving: initial={total_initial:.4f}, "
        f"final={total_final:.4f}, rel_error={rel_error:.2e}"
    )


def test_vegetation_no_negative(grid):
    """4.8: Clamp correctness — no negative vegetation.

    V=0.001, M=0, d=10.0, dt=1.0, no dispersal.
    mortality = 10*0.001 = 0.01 > V -> clamped to 0.
    """
    n = 5
    fields = grid(n)

    fields.V.from_numpy(np.full((n, n), 0.001, dtype=np.float32))
    fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))

    _flat_veg_step(
        fields,
        n,
        c=0.0,
        g_max=0.0,
        k1=5.0,
        d=10.0,
        Dp=0.0,
        dx=1.0,
        c1=0.0,
        c2=1.0,
        dt=1.0,
    )

    V_new = fields.V_new.to_numpy()
    assert V_new[2, 2] == 0.0, f"Expected V=0 (clamped), got {V_new[2, 2]:.6f}"
