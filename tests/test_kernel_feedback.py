"""Physical consistency tests for coupled feedback mechanisms.

Section 10 of the test plan: positive/negative feedbacks, runoff-runon,
and pattern instability. These are longer-running (multi-year simulations).
"""

from dataclasses import replace

import numpy as np
import pytest

from src.flow import route_wavefront
from src.params import Params
from src.simulate import step_year
from src.soil_moisture import soil_moisture_step

# Shared params for feedback tests (paper Table I/II values, dx=5 m).
_PARAMS = Params()


def _make_rain(days, n_wet=70, mean_depth=0.00417):
    """Generate exponentially distributed rain on n_wet random days.

    Returns rain in m/day (converted to mm/day inside step_year).
    """
    rng = np.random.default_rng(42)
    rain = np.zeros(days, dtype=np.float32)
    wet_days = rng.choice(days, n_wet, replace=False)
    rain[wet_days] = rng.exponential(mean_depth, n_wet).astype(np.float32)
    return rain


@pytest.mark.slow
def test_positive_feedback_vegetation_sustains(slope_grid):
    """10.1: Positive feedback — initial vegetation is sustained by rainfall.

    Two runs, 5 years. Run A: V=0 (bare). Run B: V=5 (vegetated).
    With rainfall, Run B's vegetation should persist (growth > mortality
    because infiltration feedback provides soil moisture for growth).
    Run A starts bare and should have V ~ 0 throughout (no seeds).

    This demonstrates the positive feedback: V -> enhanced I -> more M -> more growth.
    """
    n = 16
    params = _PARAMS
    rain = _make_rain(90, n_wet=30)

    results = {}
    for label, V_init in [("A", 0.0), ("B", 5.0)]:
        fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)
        fields.V.from_numpy(np.full((n, n), V_init, dtype=np.float32))
        fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

        for _ in range(5):
            step_year(fields, params, rain=rain)

        mask = fields.mask.to_numpy()
        V = fields.V.to_numpy()
        results[label] = np.mean(V[mask == 1])

    # Vegetation should persist in Run B (positive feedback sustains it)
    # while Run A stays at 0 (no vegetation to start the feedback loop)
    assert results["B"] > results["A"] + 1.0, (
        f"Vegetated V_B={results['B']:.4f} should exceed bare V_A={results['A']:.4f}"
    )
    assert results["B"] > 1.0, f"Vegetation should persist: mean V_B={results['B']:.4f}"


@pytest.mark.slow
def test_negative_feedback_vegetation_depletes_moisture(slope_grid):
    """10.2: Dense vegetation depletes soil moisture locally.

    16x16 grid, 3 years. High V=50 in one quadrant, V=0 elsewhere.
    After 3 years: M in vegetated quadrant < M in bare quadrant.
    """
    n = 16
    params = _PARAMS
    rain = _make_rain(60, n_wet=20)

    fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)

    V_init = np.zeros((n, n), dtype=np.float32)
    V_init[1 : n // 2, 1 : n // 2] = 50.0  # upper-left quadrant
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    for _ in range(3):
        step_year(fields, params, rain=rain)

    mask = fields.mask.to_numpy()
    M = fields.M.to_numpy()

    veg_mask = (V_init > 0) & (mask == 1)
    bare_mask = (V_init == 0) & (mask == 1)

    M_veg = np.mean(M[veg_mask])
    M_bare = np.mean(M[bare_mask])

    assert M_veg < M_bare, f"Vegetated M={M_veg:.4f} should be < bare M={M_bare:.4f}"


def _run_hydrology_only(fields, n_days, rain_depth, params: Params):
    """Run water routing + soil moisture for n_days (no vegetation dynamics).

    Uses g_max=0 so the soil moisture equation becomes dM/dt = I_inf - rw*M,
    isolating the infiltration signal from vegetation uptake.

    rain_depth is in m/day; converted to mm/day for R field.
    """
    for _ in range(n_days):
        fields.R.fill(float(rain_depth) * 1000.0)  # m/day -> mm/day

        for L in range(fields.max_level + 1):
            begin = fields.level_start[L]
            end = fields.level_start[L + 1]
            route_wavefront(
                fields.sorted_idx,
                begin,
                end,
                fields.Q_out,
                fields.Q_daily,
                fields.R,
                fields.I_inf,
                fields.z,
                fields.V,
                fields.flow_frac,
                fields.mask,
                params.dx,
                params.n_manning,
                params.cn,
                params.alpha,
                params.k2,
                params.W0,
            )

        # I_inf is already in mm/day — no scaling needed
        soil_moisture_step(
            fields.M,
            fields.I_inf,
            fields.V,
            fields.mask,
            0.0,  # g_max=0: no vegetation uptake
            params.k1,
            params.rw,
            1.0,
        )


def test_runoff_runon_moisture_gradient(slope_grid):
    """10.3: Vegetation enhances infiltration -> higher soil moisture.

    Two identical 16x16 slopes, 60 days of constant rainfall.
    Run A: V=0 (bare).  Run B: V=20 (vegetated).
    No vegetation dynamics (g_max=0), so M reflects only the
    infiltration gradient: I = alpha*h*(V + k2*W0)/(V + k2).

    Uses alpha=1.0 so that bare soil is capacity-limited (Q_out > 0)
    while vegetated soil captures nearly all available water.
    Mean M should be higher for the vegetated run.

    Deterministic, runs in seconds.
    """
    n = 16
    params = replace(_PARAMS, alpha=1.0)
    rain_depth = 0.01  # m/day — heavy enough that bare soil generates runoff

    results = {}
    for label, V_val in [("bare", 0.0), ("veg", 20.0)]:
        fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)
        fields.V.from_numpy(np.full((n, n), V_val, dtype=np.float32))
        fields.M.from_numpy(np.zeros((n, n), dtype=np.float32))
        _run_hydrology_only(fields, 60, rain_depth, params)

        mask = fields.mask.to_numpy()
        M = fields.M.to_numpy()
        results[label] = np.mean(M[mask == 1])

    assert results["veg"] > results["bare"], (
        f"Vegetated M={results['veg']:.4f} should exceed bare M={results['bare']:.4f}"
    )


@pytest.mark.slow
def test_pattern_instability(slope_grid):
    """10.4: Uniform state is unstable — perturbation grows.

    32x32 grid, uniform V=10 +/- 0.1, 5 years.
    std(V) should increase (pattern emerges from instability).
    """
    n = 32
    params = _PARAMS
    rain = _make_rain(60, n_wet=20)

    fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)

    rng = np.random.default_rng(42)
    V_init = (10.0 + rng.uniform(-0.1, 0.1, (n, n))).astype(np.float32)
    V_init[0, :] = V_init[-1, :] = V_init[:, 0] = V_init[:, -1] = 0.0
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    mask = fields.mask.to_numpy()
    interior = mask == 1
    std_initial = np.std(V_init[interior])

    for _ in range(5):
        step_year(fields, params, rain=rain)

    V_final = fields.V.to_numpy()
    std_final = np.std(V_final[interior])

    amplification = std_final / std_initial
    assert amplification > 2.0, (
        f"Pattern did not amplify enough: std_init={std_initial:.4f}, "
        f"std_final={std_final:.4f}, ratio={amplification:.2f}"
    )
