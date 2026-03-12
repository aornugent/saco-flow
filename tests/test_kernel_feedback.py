"""Physical consistency tests for coupled feedback mechanisms.

Section 10 of the test plan: positive/negative feedbacks, runoff-runon,
and pattern instability. These are longer-running (multi-year simulations).
"""

import numpy as np
import pytest

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.simulate import step_day, step_year


def _setup_grid(n):
    """Allocate fields with boundary mask."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)
    return fields


_DAILY = {
    "dx": 5.0,
    "n_manning": 0.05,
    "cn": 1.0,
    "alpha": 8.0,
    "k2": 18.0,
    "W0": 0.05,
    "g_max": 0.05,
    "k1": 5.0,
    "rw": 0.19,
    "c": 10.0,
    "d": 0.13,
    "Dp": 0.0007,
    "c1": 0.005,
    "c2": 0.0005,
    "dt": 1.0,
    "n_picard": 10,
}

_ANNUAL = {
    "p": 2.0,
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


def _setup_slope(n, dx):
    """Create grid with 1.4% linear slope and precomputed flow fractions."""
    fields = _setup_grid(n)
    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - 1 - i) * 0.07 * dx
    fields.z.from_numpy(z)
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, 2.0)
    return fields


def _make_rain(days_per_year, n_wet=70, mean_depth=0.00417):
    """Generate exponentially distributed rain on n_wet random days."""
    rng = np.random.default_rng(42)
    rain = np.zeros(days_per_year, dtype=np.float32)
    wet_days = rng.choice(days_per_year, n_wet, replace=False)
    rain[wet_days] = rng.exponential(mean_depth, n_wet).astype(np.float32)
    return rain


@pytest.mark.slow
def test_positive_feedback_vegetation_sustains():
    """10.1: Positive feedback — initial vegetation is sustained by rainfall.

    Two runs, 5 years. Run A: V=0 (bare). Run B: V=5 (vegetated).
    With rainfall, Run B's vegetation should persist (growth > mortality
    because infiltration feedback provides soil moisture for growth).
    Run A starts bare and should have V ≈ 0 throughout (no seeds).

    This demonstrates the positive feedback: V → enhanced I → more M → more growth.
    """
    n = 16
    dx = _DAILY["dx"]
    days_per_year = 90
    rain = _make_rain(days_per_year, n_wet=30)

    results = {}
    for label, V_init in [("A", 0.0), ("B", 5.0)]:
        fields = _setup_slope(n, dx)
        fields.V.from_numpy(np.full((n, n), V_init, dtype=np.float32))
        fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

        for _ in range(5):
            step_year(
                fields, rain=rain, days_per_year=days_per_year,
                **_DAILY, **_ANNUAL,
            )

        mask = fields.mask.to_numpy()
        V = fields.V.to_numpy()
        results[label] = np.mean(V[mask == 1])

    # Vegetation should persist in Run B (positive feedback sustains it)
    # while Run A stays at 0 (no vegetation to start the feedback loop)
    assert results["B"] > results["A"] + 1.0, (
        f"Vegetated V_B={results['B']:.4f} should exceed bare V_A={results['A']:.4f}"
    )
    assert results["B"] > 1.0, (
        f"Vegetation should persist: mean V_B={results['B']:.4f}"
    )


@pytest.mark.slow
def test_negative_feedback_vegetation_depletes_moisture():
    """10.2: Dense vegetation depletes soil moisture locally.

    16×16 grid, 3 years. High V=50 in one quadrant, V=0 elsewhere.
    After 3 years: M in vegetated quadrant < M in bare quadrant.
    """
    n = 16
    dx = _DAILY["dx"]
    days_per_year = 60
    rain = _make_rain(days_per_year, n_wet=20)

    fields = _setup_slope(n, dx)

    V_init = np.zeros((n, n), dtype=np.float32)
    V_init[1:n // 2, 1:n // 2] = 50.0  # upper-left quadrant
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    for _ in range(3):
        step_year(
            fields, rain=rain, days_per_year=days_per_year,
            **_DAILY, **_ANNUAL,
        )

    mask = fields.mask.to_numpy()
    M = fields.M.to_numpy()

    veg_mask = (V_init > 0) & (mask == 1)
    bare_mask = (V_init == 0) & (mask == 1)

    M_veg = np.mean(M[veg_mask])
    M_bare = np.mean(M[bare_mask])

    assert M_veg < M_bare, (
        f"Vegetated M={M_veg:.4f} should be < bare M={M_bare:.4f}"
    )


@pytest.mark.slow
def test_runoff_runon_moisture_gradient():
    """10.3: Vegetated cells intercept runoff → higher M than bare cells.

    32×32 grid, sparse random vegetation, 3 years.
    Classify cells by final V: vegetated cells should have higher mean M
    than bare cells (vegetation intercepts runoff via enhanced infiltration).
    """
    n = 32
    dx = _DAILY["dx"]
    days_per_year = 60
    rain = _make_rain(days_per_year, n_wet=20)

    fields = _setup_slope(n, dx)

    rng = np.random.default_rng(99)
    V_init = np.zeros((n, n), dtype=np.float32)
    mask = fields.mask.to_numpy()
    interior = np.argwhere(mask == 1)
    chosen = rng.choice(len(interior), min(100, len(interior)), replace=False)
    for idx in chosen:
        i, j = interior[idx]
        V_init[i, j] = 10.0
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    for _ in range(3):
        step_year(
            fields, rain=rain, days_per_year=days_per_year,
            **_DAILY, **_ANNUAL,
        )

    M = fields.M.to_numpy()
    V = fields.V.to_numpy()

    veg_cells = (V > 5.0) & (mask == 1)
    bare_cells = (V <= 5.0) & (mask == 1)

    if np.sum(veg_cells) > 0 and np.sum(bare_cells) > 0:
        M_veg = np.mean(M[veg_cells])
        M_bare = np.mean(M[bare_cells])
        assert M_veg > M_bare, (
            f"Vegetated M={M_veg:.4f} should exceed bare M={M_bare:.4f}"
        )


@pytest.mark.slow
def test_pattern_instability():
    """10.4: Uniform state is unstable — perturbation grows.

    32×32 grid, uniform V=10 ± 0.1, 5 years.
    std(V) should increase (pattern emerges from instability).
    """
    n = 32
    dx = _DAILY["dx"]
    days_per_year = 60
    rain = _make_rain(days_per_year, n_wet=20)

    fields = _setup_slope(n, dx)

    rng = np.random.default_rng(42)
    V_init = (10.0 + rng.uniform(-0.1, 0.1, (n, n))).astype(np.float32)
    V_init[0, :] = V_init[-1, :] = V_init[:, 0] = V_init[:, -1] = 0.0
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    mask = fields.mask.to_numpy()
    interior = mask == 1
    std_initial = np.std(V_init[interior])

    for _ in range(5):
        step_year(
            fields, rain=rain, days_per_year=days_per_year,
            **_DAILY, **_ANNUAL,
        )

    V_final = fields.V.to_numpy()
    std_final = np.std(V_final[interior])

    amplification = std_final / std_initial
    assert amplification > 2.0, (
        f"Pattern did not amplify enough: std_init={std_initial:.4f}, "
        f"std_final={std_final:.4f}, ratio={amplification:.2f}"
    )
