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


def _make_rain(days, n_wet=70, mean_depth=0.00417, seed=42):
    """Generate exponentially distributed rain on n_wet random days.

    Returns rain in m/day (converted to mm/day inside step_year).
    """
    rng = np.random.default_rng(seed)
    rain = np.zeros(days, dtype=np.float32)
    wet_days = rng.choice(days, n_wet, replace=False)
    rain[wet_days] = rng.exponential(mean_depth, n_wet).astype(np.float32)
    return rain


@pytest.mark.slow
def test_positive_feedback_vegetation_sustains(slope_grid):
    """10.1: Positive feedback — a vegetation band persists via runon.

    32x32 slope: bare upslope, vegetation band near the bottom.
    Bare soil sheds runoff downslope; the vegetated band captures it via
    enhanced infiltration, sustaining growth > mortality.

    Run A: all bare (V=0). Run B: band at V=20 on bare background.
    After 5 years, the band should persist (mean V in band > 1.0).
    """
    n = 32
    params = _PARAMS
    rain = _make_rain(90, n_wet=30)

    results = {}
    for label, band_v in [("A", 0.0), ("B", 20.0)]:
        fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)
        # Bare everywhere
        V_init = np.zeros((n, n), dtype=np.float32)
        # Place band near the bottom (rows n-4 to n-2, avoiding boundary)
        V_init[n - 4 : n - 1, 1 : n - 1] = band_v
        fields.V.from_numpy(V_init)
        fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

        for _ in range(5):
            step_year(fields, params, rain=rain)

        mask = fields.mask.to_numpy()
        V = fields.V.to_numpy()
        # Measure vegetation in the band region
        band_mask = np.zeros_like(mask)
        band_mask[n - 4 : n - 1, 1 : n - 1] = 1
        band_cells = (band_mask == 1) & (mask == 1)
        results[label] = np.mean(V[band_cells])

    # Band should persist in Run B; Run A stays bare
    assert results["B"] > results["A"] + 1.0, (
        f"Band V_B={results['B']:.4f} should exceed bare V_A={results['A']:.4f}"
    )
    assert results["B"] > 1.0, (
        f"Vegetation band should persist: mean V_B={results['B']:.4f}"
    )


@pytest.mark.slow
def test_negative_feedback_vegetation_depletes_moisture(slope_grid):
    """10.2: Dense vegetation depletes soil moisture via uptake.

    Two identical 32x32 slopes with V=50 (fixed, no veg dynamics).
    Same infiltration in both runs (V enhances it equally).
    Run A: g_max=0 in soil moisture (no uptake).
    Run B: g_max>0 in soil moisture (uptake active).

    M_with_uptake < M_without_uptake, demonstrating the negative feedback
    where vegetation depletes the moisture it depends on.
    """
    n = 32
    params = _PARAMS
    rain_depth = 0.005  # m/day
    V_val = 50.0

    results = {}
    for label, g_max in [("no_uptake", 0.0), ("uptake", params.g_max)]:
        fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)
        fields.V.from_numpy(np.full((n, n), V_val, dtype=np.float32))
        fields.M.from_numpy(np.full((n, n), 0.0, dtype=np.float32))

        for _ in range(60):
            fields.R.fill(float(rain_depth) * 1000.0)
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
                params.dx,
                params.n_manning,
                params.cn,
                params.alpha,
                params.k2,
                params.W0,
            )
            soil_moisture_step(
                fields.M,
                fields.I_inf,
                fields.V,
                fields.mask,
                g_max,
                params.k1,
                params.rw,
                1.0,
            )

        mask = fields.mask.to_numpy()
        M = fields.M.to_numpy()
        results[label] = np.mean(M[mask == 1])

    assert results["uptake"] < results["no_uptake"], (
        f"M with uptake={results['uptake']:.4f} should be < "
        f"M without uptake={results['no_uptake']:.4f}"
    )


def _run_hydrology_only(fields, n_days, rain_depth, params: Params):
    """Run water routing + soil moisture for n_days (no vegetation dynamics).

    Uses g_max=0 so the soil moisture equation becomes dM/dt = I_inf - rw*M,
    isolating the infiltration signal from vegetation uptake.

    rain_depth is in m/day; converted to mm/day for R field.
    """
    for _ in range(n_days):
        fields.R.fill(float(rain_depth) * 1000.0)  # m/day -> mm/day

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
    """10.4: Spatial instability — uniform V differentiates along the slope.

    32x32 slope: uniform V=1 everywhere (just enough to trigger feedbacks).
    Marginal rainfall (30 wet days at 4.17 mm mean) — insufficient to
    sustain vegetation from local rainfall alone, but runon at the
    bottom of the slope concentrates water.

    After 5 years, vegetation should differentiate: it concentrates
    where runon accumulates (downslope) and dies where local rainfall
    is insufficient (upslope). std(V) should increase substantially
    from the near-zero initial std.
    """
    n = 32
    params = _PARAMS
    rain = _make_rain(90, n_wet=30)

    fields = slope_grid(n, params.dx, p=params.p, step=0.07 * params.dx)

    mask_np = fields.mask.to_numpy()

    # Uniform weak vegetation everywhere
    V_init = np.full((n, n), 1.0, dtype=np.float32)
    V_init[0, :] = V_init[-1, :] = V_init[:, 0] = V_init[:, -1] = 0.0
    fields.V.from_numpy(V_init)
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    for _ in range(5):
        step_year(fields, params, rain=rain)

    V_final = fields.V.to_numpy()

    # Vegetation should concentrate downslope (high row indices = low elevation)
    # Compare mean V in lower third vs upper third
    third = n // 3
    upper_mask = np.zeros_like(mask_np)
    upper_mask[1:third, 1 : n - 1] = 1
    upper_cells = (upper_mask == 1) & (mask_np == 1)

    lower_mask = np.zeros_like(mask_np)
    lower_mask[2 * third : n - 1, 1 : n - 1] = 1
    lower_cells = (lower_mask == 1) & (mask_np == 1)

    V_upper = np.mean(V_final[upper_cells])
    V_lower = np.mean(V_final[lower_cells])

    # Downslope should have more vegetation than upslope
    assert V_lower > V_upper + 0.5, (
        f"Spatial differentiation not observed: "
        f"V_lower={V_lower:.4f}, V_upper={V_upper:.4f}"
    )
