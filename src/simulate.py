"""Daily/annual simulation orchestrator per Baartman et al. (2018) section 7."""

from src.fields import Fields
from src.flow import compute_flow_fractions, route_water
from src.sediment import sediment_transport, update_elevation
from src.soil_moisture import soil_moisture_step
from src.vegetation import vegetation_step


def step_day(
    fields: Fields,
    *,
    dx: float,
    n_manning: float,
    cn: float,
    alpha: float,
    k2: float,
    W0: float,
    g_max: float,
    k1: float,
    rw: float,
    c: float,
    d: float,
    Dp: float,
    c1: float,
    c2: float,
    dt: float = 1.0,
    n_picard: int = 20,
):
    """Run one daily timestep: water routing, soil moisture, vegetation.

    Args:
        fields: Simulation fields (flow_frac must be precomputed)
        dx: Cell spacing [m]
        n_manning: Manning's roughness [s/m^(1/3)]
        cn: Kinematic wave constant [m^(1/3)/s]
        alpha: Infiltration capacity [1/day]
        k2: Vegetation half-saturation for infiltration [%]
        W0: Bare-soil infiltration fraction [-]
        g_max: Maximum growth rate [1/day]
        k1: Half-saturation for moisture [m]
        rw: Soil moisture loss rate [1/day]
        c: Vegetation growth scaling [-]
        d: Mortality rate [1/day]
        Dp: Isotropic seed diffusion [m^2/day]
        c1: Flow dispersal coefficient [day/m^2]
        c2: Flow dispersal saturation [1/day]
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
        fields.swap("Q_out")

    # section 3: Soil moisture (no lateral diffusion per paper)
    soil_moisture_step(
        fields.M,
        fields.M_new,
        fields.I_inf,
        fields.V,
        fields.mask,
        g_max,
        k1,
        rw,
        dt,
    )
    fields.swap("M")

    # section 4: Vegetation
    vegetation_step(
        fields.V,
        fields.V_new,
        fields.M,
        fields.Q_daily,
        fields.flow_frac,
        fields.mask,
        c,
        g_max,
        k1,
        d,
        Dp,
        dx,
        c1,
        c2,
        dt,
    )
    fields.swap("V")


def step_year(
    fields: Fields,
    *,
    dx: float,
    p: float,
    n_manning: float,
    cn: float,
    alpha: float,
    k2: float,
    W0: float,
    g_max: float,
    k1: float,
    rw: float,
    c: float,
    d: float,
    Dp: float,
    c1: float,
    c2: float,
    gamma: float,
    m_exp: float,
    n_exp: float,
    K_max: float,
    K_min: float,
    P_min: float,
    P_max: float,
    v_low: float,
    v_high: float,
    days_per_year: int = 365,
    dt: float = 1.0,
    n_picard: int = 20,
):
    """Run one year: daily steps + annual sediment/elevation update.

    Args:
        fields: Simulation fields (flow_frac must be precomputed)
        dx: Cell spacing [m]
        p: MFD convergence exponent
        n_manning: Manning's roughness [s/m^(1/3)]
        cn: Kinematic wave constant [m^(1/3)/s]
        alpha: Infiltration capacity [1/day]
        k2: Vegetation half-saturation for infiltration [%]
        W0: Bare-soil infiltration fraction [-]
        g_max: Maximum growth rate [1/day]
        k1: Half-saturation for moisture [m]
        rw: Soil moisture loss rate [1/day]
        c: Vegetation growth scaling [-]
        d: Mortality rate [1/day]
        Dp: Isotropic seed diffusion [m^2/day]
        c1: Flow dispersal coefficient [day/m^2]
        c2: Flow dispersal saturation [1/day]
        gamma: Sediment transport coefficient
        m_exp: Discharge exponent [-]
        n_exp: Slope exponent [-]
        K_max, K_min: Erosion coefficient range [-]
        P_min, P_max: Deposition coefficient range [-]
        v_low: Vegetation threshold for max erosion [%]
        v_high: Vegetation threshold for min erosion [%]
        days_per_year: Number of daily steps per year
        dt: Daily timestep [days]
        n_picard: Global Picard iterations for water routing
    """
    daily_kwargs = {
        "dx": dx,
        "n_manning": n_manning,
        "cn": cn,
        "alpha": alpha,
        "k2": k2,
        "W0": W0,
        "g_max": g_max,
        "k1": k1,
        "rw": rw,
        "c": c,
        "d": d,
        "Dp": Dp,
        "c1": c1,
        "c2": c2,
        "dt": dt,
        "n_picard": n_picard,
    }

    # Daily loop
    for _ in range(days_per_year):
        step_day(fields, **daily_kwargs)

    # section 5: Sediment transport (annual, uses end-of-year Q_daily)
    sediment_transport(
        fields.S,
        fields.S_new,
        fields.Q_daily,
        fields.z,
        fields.V,
        fields.flow_frac,
        fields.mask,
        dx,
        gamma,
        m_exp,
        n_exp,
        K_max,
        K_min,
        P_min,
        P_max,
        v_low,
        v_high,
    )

    # section 6: Elevation update (before swapping S buffers)
    update_elevation(
        fields.z, fields.S, fields.S_new, fields.flow_frac, fields.mask, dx
    )
    fields.swap("S")

    # section 1: Recompute flow fractions from updated elevation
    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)
