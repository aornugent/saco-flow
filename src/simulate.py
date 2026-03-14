"""Daily/annual simulation orchestrator per Baartman et al. (2018) section 7."""

from src.fields import Fields
from src.flow import accumulate_annual_Q, compute_flow_fractions, route_water
from src.params import Params
from src.sediment import sediment_transport, update_elevation
from src.soil_moisture import soil_moisture_step
from src.vegetation import vegetation_step


def step_day(
    fields: Fields,
    params: Params,
    *,
    dt: float = 1.0,
    n_picard: int = 20,
):
    """Run one daily timestep: water routing, soil moisture, vegetation.

    Args:
        fields: Simulation fields (flow_frac must be precomputed)
        params: Physical constants
        dt: Timestep [days]
        n_picard: Global Picard iterations for water routing
    """
    # section 2: Water routing (global Picard iteration)
    for _ in range(n_picard):
        route_water(
            fields.Q_out,
            fields.Q_out_new,
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
        fields.swap("Q_out")

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
    n_picard: int = 20,
):
    """Run one year: daily steps + annual sediment/elevation update.

    The number of daily steps equals len(rain) when rain is supplied, or 365
    when rain is None (fields.R must be preset in mm/day in that case).

    Args:
        fields: Simulation fields (flow_frac must be precomputed)
        params: Physical constants
        rain: Daily rainfall array [m/day]. Length determines days per year.
              Converted to mm/day internally.
              If None, fields.R must be preset in mm/day and 365 steps are run.
        dt: Daily timestep [days]
        n_picard: Global Picard iterations for water routing
    """
    days = len(rain) if rain is not None else 365

    # Daily loop — accumulate Q_daily into Q_annual for sediment transport
    fields.Q_annual.fill(0.0)
    for day in range(days):
        if rain is not None:
            fields.R.fill(float(rain[day]) * 1000.0)  # m/day -> mm/day
        step_day(fields, params, dt=dt, n_picard=n_picard)
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

    # section 1: Recompute flow fractions from updated elevation
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, params.dx, params.p)
