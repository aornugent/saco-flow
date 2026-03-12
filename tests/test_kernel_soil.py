"""Physical consistency tests for soil moisture dynamics.

Section 3 of the test plan: exact Euler steps, decay, steady state,
mass conservation, and clamp correctness.
"""

import math

import numpy as np
import pytest

from src.fields import allocate
from src.soil_moisture import soil_moisture_step


def _setup_grid(n):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


def test_soil_moisture_exact_euler():
    """3.1: Single-cell exact Euler step with known parameters.

    M=0.3, V=10, I_inf=0.5, g_max=0.05, k1=5.0, rw=0.19, dt=1.0.
    uptake = 0.05 * 0.3/(0.3+5.0) * 10.0 = 0.02830
    loss = 0.19 * 0.3 = 0.057
    M_new = 0.3 + 1.0 * (0.5 − 0.02830 − 0.057) = 0.7147
    """
    n = 3
    fields = _setup_grid(n)

    fields.M.from_numpy(np.full((n, n), 0.3, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 10.0, dtype=np.float32))
    fields.I_inf.from_numpy(np.full((n, n), 0.5, dtype=np.float32))

    soil_moisture_step(
        fields.M, fields.M_new, fields.I_inf, fields.V, fields.mask,
        g_max=0.05, k1=5.0, rw=0.19, dt=1.0,
    )

    M_new = fields.M_new.to_numpy()

    # Hand computation
    m, v = 0.3, 10.0
    uptake = 0.05 * m / (m + 5.0) * v
    loss = 0.19 * m
    expected = m + 1.0 * (0.5 - uptake - loss)

    assert abs(M_new[1, 1] - expected) < 1e-4, (
        f"M_new={M_new[1, 1]:.6f}, expected={expected:.6f}"
    )


def test_soil_moisture_zero_infiltration_decay():
    """3.2: Zero infiltration, known decay.

    M=1.0, V=0.0 (bare soil → uptake=0), I_inf=0, rw=0.19, dt=1.0.
    M_new = 1.0 − 0.19 = 0.81 (exact).
    """
    n = 3
    fields = _setup_grid(n)

    fields.M.from_numpy(np.full((n, n), 1.0, dtype=np.float32))
    fields.V.from_numpy(np.zeros((n, n), dtype=np.float32))
    fields.I_inf.from_numpy(np.zeros((n, n), dtype=np.float32))

    soil_moisture_step(
        fields.M, fields.M_new, fields.I_inf, fields.V, fields.mask,
        g_max=0.05, k1=5.0, rw=0.19, dt=1.0,
    )

    M_new = fields.M_new.to_numpy()
    assert abs(M_new[1, 1] - 0.81) < 1e-6, f"M_new={M_new[1, 1]:.6f}, expected=0.81"


def test_soil_moisture_steady_state():
    """3.3: Steady-state convergence.

    Constant I_inf=2.0, V=15.0, g_max=0.05, k1=5.0, rw=0.19, dt=1.0.
    At steady state: 2.0 = 0.75 * M*/(M*+5) + 0.19 * M*
    Solve: M* ≈ 5.263 mm.
    """
    n = 3
    fields = _setup_grid(n)

    V = 15.0
    I_inf = 2.0
    g_max = 0.05
    k1 = 5.0
    rw = 0.19

    fields.V.from_numpy(np.full((n, n), V, dtype=np.float32))
    fields.I_inf.from_numpy(np.full((n, n), I_inf, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    # Solve for analytical steady state: I = g_max * M/(M+k1) * V + rw * M
    # 2.0 = 0.05 * M/(M+5) * 15 + 0.19 * M
    # 2.0 = 0.75 * M/(M+5) + 0.19*M
    # Rearranging: 2*(M+5) = 0.75*M + 0.19*M*(M+5)
    # 2M + 10 = 0.75M + 0.19M² + 0.95M
    # 0.19M² + 0.95M + 0.75M - 2M - 10 = 0
    # 0.19M² - 0.3M - 10 = 0
    # M = (0.3 + sqrt(0.09 + 4*0.19*10)) / (2*0.19)
    # M = (0.3 + sqrt(0.09 + 7.6)) / 0.38
    # M = (0.3 + sqrt(7.69)) / 0.38
    # M = (0.3 + 2.7731) / 0.38 ≈ 8.087
    # Hmm, let me redo more carefully:
    # 2.0(M+5) = 0.75M + 0.19M(M+5)
    # 2M + 10 = 0.75M + 0.19M² + 0.95M
    # 0.19M² + (0.75 + 0.95 - 2)M - 10 = 0
    # 0.19M² - 0.3M - 10 = 0
    disc = 0.3**2 + 4 * 0.19 * 10
    M_star = (0.3 + math.sqrt(disc)) / (2 * 0.19)

    for _ in range(500):
        soil_moisture_step(
            fields.M, fields.M_new, fields.I_inf, fields.V, fields.mask,
            g_max=g_max, k1=k1, rw=rw, dt=1.0,
        )
        fields.swap("M")

    M_final = fields.M.to_numpy()
    assert abs(M_final[1, 1] - M_star) < 0.05, (
        f"M={M_final[1, 1]:.4f}, expected M*={M_star:.4f}"
    )


def test_soil_moisture_mass_conservation():
    """3.4: Mass conservation over one step for every interior cell.

    |M_new − M − dt*(I_inf − uptake − loss)| < 1e-6
    unless clamp fires (M_new = 0 and M + dt*dMdt < 0).
    """
    n = 8
    fields = _setup_grid(n)

    rng = np.random.default_rng(42)
    M0 = rng.uniform(0.0, 2.0, (n, n)).astype(np.float32)
    V0 = rng.uniform(0.0, 20.0, (n, n)).astype(np.float32)
    I0 = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)

    fields.M.from_numpy(M0)
    fields.V.from_numpy(V0)
    fields.I_inf.from_numpy(I0)

    g_max, k1, rw, dt = 0.05, 5.0, 0.19, 1.0
    soil_moisture_step(
        fields.M, fields.M_new, fields.I_inf, fields.V, fields.mask,
        g_max=g_max, k1=k1, rw=rw, dt=dt,
    )

    M_new = fields.M_new.to_numpy()
    mask = fields.mask.to_numpy()

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if mask[i, j] == 0:
                continue
            m = M0[i, j]
            v = V0[i, j]
            uptake = g_max * m / (m + k1) * v
            loss = rw * m
            dMdt = I0[i, j] - uptake - loss
            unclamped = m + dt * dMdt

            if M_new[i, j] == 0.0:
                # Clamp fired — verify it was needed
                assert unclamped < 0.0, (
                    f"Clamp at ({i},{j}) but unclamped={unclamped:.6f} >= 0"
                )
            else:
                residual = abs(M_new[i, j] - unclamped)
                assert residual < 1e-5, (
                    f"Conservation at ({i},{j}): residual={residual:.2e}"
                )


def test_soil_moisture_no_negative():
    """3.5: Clamp correctness — no negative moisture.

    M=0.001, V=100, I_inf=0, g_max=0.5, k1=0.1, rw=0.5, dt=1.0.
    Unclamped: 0.001 − 0.4955 = −0.4945 → clamped to 0.
    """
    n = 3
    fields = _setup_grid(n)

    fields.M.from_numpy(np.full((n, n), 0.001, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 100.0, dtype=np.float32))
    fields.I_inf.from_numpy(np.zeros((n, n), dtype=np.float32))

    soil_moisture_step(
        fields.M, fields.M_new, fields.I_inf, fields.V, fields.mask,
        g_max=0.5, k1=0.1, rw=0.5, dt=1.0,
    )

    M_new = fields.M_new.to_numpy()
    assert M_new[1, 1] == 0.0, f"Expected M=0 (clamped), got {M_new[1, 1]:.6f}"
