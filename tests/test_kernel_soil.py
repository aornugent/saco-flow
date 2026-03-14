"""Physical consistency tests for soil moisture dynamics.

Section 3 of the test plan: exact Euler steps, decay, steady state,
mass conservation, and clamp correctness.

All quantities in mm: M [mm], I_inf [mm/day], k1 [mm].
soil_moisture_step updates M in-place (point-wise, no buffer needed).
"""

import math

import numpy as np

from src.soil_moisture import soil_moisture_step


def test_soil_moisture_exact_euler(grid):
    """3.1: Single-cell exact Euler step with known parameters.

    M=0.3, V=10, I_inf=0.5, g_max=0.05, k1=5.0, rw=0.19, dt=1.0.
    uptake = 0.05 * 0.3/(0.3+5.0) * 10.0 = 0.02830
    loss = 0.19 * 0.3 = 0.057
    M_new = 0.3 + 1.0 * (0.5 - 0.02830 - 0.057) = 0.7147
    """
    n = 3
    fields = grid(n)

    fields.M.from_numpy(np.full((n, n), 0.3, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))
    fields.I_inf.from_numpy(np.full((n, n), 0.5, dtype=np.float32))

    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=0.05,
        k1=5.0,
        rw=0.19,
        dt=1.0,
    )

    M = fields.M.to_numpy()

    # Hand computation
    m, v = 0.3, 10.0
    uptake = 0.05 * m / (m + 5.0) * v
    loss = 0.19 * m
    expected = m + 1.0 * (0.5 - uptake - loss)

    assert abs(M[1, 1] - expected) < 1e-4, f"M={M[1, 1]:.6f}, expected={expected:.6f}"


def test_soil_moisture_zero_infiltration_decay(grid):
    """3.2: Zero infiltration, known decay.

    M=1.0, V=0.0 (bare soil -> uptake=0), I_inf=0, rw=0.19, dt=1.0.
    M_new = 1.0 - 0.19 = 0.81 (exact).
    """
    n = 3
    fields = grid(n)

    fields.M.from_numpy(np.full((n, n), 1.0, dtype=np.float32))
    fields.V.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.I_inf.from_numpy(np.zeros((n, n), dtype=np.float32))

    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=0.05,
        k1=5.0,
        rw=0.19,
        dt=1.0,
    )

    M = fields.M.to_numpy()
    assert abs(M[1, 1] - 0.81) < 1e-6, f"M={M[1, 1]:.6f}, expected=0.81"


def test_soil_moisture_steady_state(grid):
    """3.3: Steady-state convergence.

    Constant I_inf=2.0, V=15.0, g_max=0.05, k1=5.0, rw=0.19, dt=1.0.
    At steady state: 2.0 = 0.75 * M*/(M*+5) + 0.19 * M*
    Solve: M* ~ 8.087 mm.
    """
    n = 3
    fields = grid(n)

    V = 15.0
    I_inf = 2.0
    g_max = 0.05
    k1 = 5.0
    rw = 0.19

    fields.V.from_numpy(np.full((n, n), V, dtype=np.float32))
    fields.I_inf.from_numpy(np.full((n, n), I_inf, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    # Analytical steady state:
    # 0.19M^2 - 0.3M - 10 = 0
    disc = 0.3**2 + 4 * 0.19 * 10
    M_star = (0.3 + math.sqrt(disc)) / (2 * 0.19)

    for _ in range(500):
        # Refill I_inf each step (it's overwritten in the real sim, but here
        # we keep it constant to test steady state convergence)
        fields.I_inf.from_numpy(np.full((n, n), I_inf, dtype=np.float32))
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

    M_final = fields.M.to_numpy()
    assert abs(M_final[1, 1] - M_star) < 0.05, (
        f"M={M_final[1, 1]:.4f}, expected M*={M_star:.4f}"
    )


def test_soil_moisture_mass_conservation(grid):
    """3.4: Mass conservation over one step for every interior cell.

    |M_after - M_before - dt*(I_inf - uptake - loss)| < 1e-6
    unless clamp fires (M_after = 0 and M_before + dt*dMdt < 0).
    """
    n = 8
    fields = grid(n)

    rng = np.random.default_rng(42)
    M0 = rng.uniform(0.0, 2.0, (n, n)).astype(np.float32)
    V0 = rng.uniform(0.0, 20.0, (n, n)).astype(np.float32)
    I0 = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)

    fields.M.from_numpy(M0)
    fields.V.from_numpy(V0)
    fields.I_inf.from_numpy(I0)

    g_max, k1, rw, dt = 0.05, 5.0, 0.19, 1.0
    M_before = M0.copy()

    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=g_max,
        k1=k1,
        rw=rw,
        dt=dt,
    )

    M_after = fields.M.to_numpy()
    mask = fields.mask.to_numpy()

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            m = M_before[i, j]
            v = V0[i, j]
            uptake = g_max * m / (m + k1) * v
            loss = rw * m
            dMdt = I0[i, j] - uptake - loss
            unclamped = m + dt * dMdt

            if M_after[i, j] == 0.0:
                # Clamp fired — verify it was needed
                assert unclamped < 0.0, (
                    f"Clamp at ({i},{j}) but unclamped={unclamped:.6f} >= 0"
                )
            else:
                residual = abs(M_after[i, j] - unclamped)
                assert residual < 1e-5, (
                    f"Conservation at ({i},{j}): residual={residual:.2e}"
                )


def test_soil_moisture_no_negative(grid):
    """3.5: Clamp correctness — no negative moisture.

    M=0.001, V=100, I_inf=0, g_max=0.5, k1=0.1, rw=0.5, dt=1.0.
    Unclamped: 0.001 - 0.4955 = -0.4945 -> clamped to 0.
    """
    n = 3
    fields = grid(n)

    fields.M.from_numpy(np.full((n, n), 0.001, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 100.0, dtype=np.float32))
    fields.I_inf.from_numpy(np.zeros((n, n), dtype=np.float32))

    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max=0.5,
        k1=0.1,
        rw=0.5,
        dt=1.0,
    )

    M = fields.M.to_numpy()
    assert M[1, 1] == 0.0, f"Expected M=0 (clamped), got {M[1, 1]:.6f}"
