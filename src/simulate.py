"""Daily/annual simulation orchestrator per Baartman et al. (2018) section 7."""

from src.fields import Fields
from src.flow import (
    accumulate_annual_Q,
    compute_flow_fractions,
    prepare_levels,
)
from src.params import Params
from src.sediment import sediment_transport, update_elevation
from src.soil_moisture import soil_moisture_step
from src.surface import decay_F_inf, step_storm
from src.vegetation import vegetation_step


def step_day(
    fields: Fields,
    params: Params,
    *,
    rain_mm: float = 0.0,
    dt: float = 1.0,
):
    """Run one daily timestep: surface water, soil moisture, vegetation.

    Args:
        fields: Simulation fields (flow_frac and levels must be precomputed)
        params: Physical constants
        rain_mm: Total rainfall this day [mm]
        dt: Timestep [days]
    """
    # section 2: Surface water — diffusion wave + Green-Ampt infiltration
    step_storm(fields, params, rain_mm)

    # section 3: Soil moisture (no lateral diffusion per paper)
    # I_inf is already in mm/day — no conversion needed.
    # Point-wise: updated in-place, no buffer swap.
    soil_moisture_step(
        fields.M,
        fields.I_inf,
        fields.V,
        fields.mask,
        params.g_max,
        params.k1,
        params.rw,
        dt,
    )

    # Decay wetting front depth between storms, keeping F_inf consistent
    # with M dynamics.  Dry days shrink F so the next storm starts with
    # high suction; wet sequences keep F large and pond immediately.
    decay_F_inf(fields.F_inf, fields.mask, params.rw, dt)

    # section 4: Vegetation
    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.flow_frac,
        fields.mask,
        params.c,
        params.g_max,
        params.k1,
        params.d,
        params.Dp,
        params.dx,
        params.c1,
        params.c2,
        dt,
    )
    fields.swap("V")


def step_year(
    fields: Fields,
    params: Params,
    *,
    rain=None,
    dt: float = 1.0,
):
    """Run one year: daily steps + annual sediment/elevation update.

    The number of daily steps equals len(rain) when rain is supplied, or 365
    when rain is None (fields.R must be preset in mm/day in that case).

    Args:
        fields: Simulation fields (flow_frac and levels must be precomputed)
        params: Physical constants
        rain: Daily rainfall array [m/day]. Length determines days per year.
              Converted to mm/day internally.
              If None, fields.R must be preset in mm/day and 365 steps are run.
        dt: Daily timestep [days]
    """
    days = len(rain) if rain is not None else 365

    # Daily loop — accumulate Q_daily into Q_annual for sediment transport
    fields.Q_annual.fill(0.0)
    for day in range(days):
        rain_mm = float(rain[day]) * 1000.0 if rain is not None else 0.0  # m/day -> mm
        step_day(fields, params, rain_mm=rain_mm, dt=dt)
        accumulate_annual_Q(fields.Q_annual, fields.Q_daily, fields.mask)

    # section 5: Sediment transport (annual, uses cumulative annual Q)
    sediment_transport(
        fields.S,
        fields.S_new,
        fields.Q_annual,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        params.dx,
        params.gamma,
        params.m_exp,
        params.n_exp,
        params.K_max,
        params.K_min,
        params.P_min,
        params.P_max,
        params.v_low,
        params.v_high,
    )

    # section 6: Elevation update (before swapping S buffers)
    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, params.dx
    )
    fields.swap("S")

    # section 1: Recompute flow fractions and wavefront levels
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, params.dx, params.p)
    prepare_levels(fields)
