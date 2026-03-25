"""
Surface water solver — diffusion wave routing with Green-Ampt infiltration.

Replaces the steady-state wavefront router with a transient, GPU-parallel
diffusion wave (LISFLOOD-FP style).  Infiltration is operator-split as a
point-wise sink; the two kernels share the surface depth field h.

All depths in mm, times in hours within the storm substep loop.
Internal kernel arithmetic in SI (metres, seconds); conversions at
field boundaries only.  Interface fields (I_inf, Q_daily) are
converted to mm/day and mm*m/day respectively at the end.
"""

import taichi as ti

from src.fields import Fields
from src.params import Params
from src.stencil import CARD


@ti.kernel
def apply_rainfall(
    h: ti.template(),
    mask: ti.template(),
    intensity: ti.f32,
    dt: ti.f32,
):
    """Add rainfall to surface depth.  Point-wise, in-place.

    Args:
        h: Surface water depth [mm]
        mask: Active cell mask
        intensity: Rainfall intensity [mm/hr]
        dt: Substep duration [hr]
    """
    n = h.shape[0]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 1:
            h[i, j] += intensity * dt


@ti.kernel
def infiltration_green_ampt(
    h: ti.template(),
    F_inf: ti.template(),
    V: ti.template(),
    mask: ti.template(),
    K_s: ti.f32,
    psi_f: ti.f32,
    delta_theta: ti.f32,
    k2: ti.f32,
    W0: ti.f32,
    dt: ti.f32,
):
    """Green-Ampt infiltration with Rietkerk vegetation modulation.  Point-wise, in-place.

    Infiltration rate: f = K_eff * (1 + psi_f * delta_theta / F)
    where K_eff = K_s * (V + k2*W0) / (V + k2).

    Args:
        h: Surface water depth (read/write) [mm]
        F_inf: Cumulative infiltration (read/write) [mm]
        V: Vegetation density [g/m^2]
        mask: Active cell mask
        K_s: Saturated hydraulic conductivity [mm/hr]
        psi_f: Wetting front suction head [mm]
        delta_theta: Moisture deficit [-]
        k2: Vegetation half-saturation for infiltration [g/m^2]
        W0: Bare-soil infiltration fraction [-]
        dt: Substep duration [hr]
    """
    n = h.shape[0]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 0 or h[i, j] <= 0.0:
            continue

        v = V[i, j]
        K_eff = K_s * (v + k2 * W0) / (v + k2)  # [mm/hr]

        F_cur = ti.max(F_inf[i, j], 1e-6)  # avoid division by zero
        f_rate = K_eff * (1.0 + psi_f * delta_theta / F_cur)  # [mm/hr]

        I_vol = ti.min(f_rate * dt, h[i, j])  # [mm]
        h[i, j] -= I_vol
        F_inf[i, j] += I_vol


@ti.kernel
def diffusion_wave_step(
    h: ti.template(),
    h_new: ti.template(),
    z: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    dt: ti.f32,
    n_M: ti.f32,
):
    """Explicit diffusion wave update on 4-connected stencil.

    Manning flux: q = h_face^(5/3) / n_M * sqrt(|d_eta|/dx) * sign(d_eta)
    where eta = z + h (water surface elevation).

    All internal arithmetic in SI (metres, seconds).  h is converted
    mm→m on entry and m→mm on exit — one conversion in, one out.

    Stencil op: read h, write h_new — caller swaps after.

    Args:
        h: Surface water depth (read) [mm]
        h_new: Surface water depth (write) [mm]
        z: Elevation [m]
        mask: Active cell mask
        dx: Cell spacing [m]
        dt: Substep duration [hr]
        n_M: Manning's roughness [s/m^(1/3)]
    """
    n = h.shape[0]
    dt_s = dt * 3600.0  # [s]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            h_new[i, j] = 0.0
            continue

        h_m = h[i, j] / 1000.0  # [m]
        eta_here = z[i, j] + h_m  # [m]
        flux_sum = 0.0  # net outward flux [m/s]

        for di, dj in ti.static(CARD):
            ni, nj = i + di, j + dj

            if mask[ni, nj] == 0:
                # Boundary: h=0 outflow sink; only allow outward flux
                eta_there = z[ni, nj]  # [m], h=0
                d_eta = (eta_here - eta_there) / dx  # slope [-]
                if d_eta > 1e-10 and h_m > 1e-6:
                    q = ti.pow(h_m, 5.0 / 3.0) / n_M * ti.sqrt(d_eta)  # [m^2/s]
                    flux_sum += q / dx  # [m/s]
            else:
                # Interior neighbor
                h_nb = h[ni, nj] / 1000.0  # [m]
                eta_there = z[ni, nj] + h_nb  # [m]
                d_eta = (eta_here - eta_there) / dx  # slope [-]

                # Upwind depth: from the cell with higher water surface
                h_face = h_m if eta_here >= eta_there else h_nb  # [m]

                if h_face > 1e-6 and ti.abs(d_eta) > 1e-10:
                    q = (
                        ti.pow(h_face, 5.0 / 3.0) / n_M * ti.sqrt(ti.abs(d_eta))
                    )  # [m^2/s]
                    sign = 1.0 if d_eta > 0.0 else -1.0
                    flux_sum += sign * q / dx  # [m/s]

        h_new[i, j] = ti.max(0.0, h[i, j] - dt_s * flux_sum * 1000.0)  # [mm]


@ti.kernel
def compute_adaptive_dt(
    h: ti.template(),
    z: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    n_M: ti.f32,
    cfl: ti.f32,
    dt_max: ti.f32,
    result: ti.template(),
):
    """CFL-limited global dt from max diffusivity across domain.

    Diffusion coefficient: D = h^(5/3) / (n_M * sqrt(S))  [m^2/s]
    CFL for explicit diffusion: dt <= cfl * dx^2 / (4*D)

    Args:
        h: Surface water depth [mm]
        z: Elevation [m]
        mask: Active cell mask
        dx: Cell spacing [m]
        n_M: Manning's roughness [s/m^(1/3)]
        cfl: CFL safety factor [-]
        dt_max: Maximum allowed substep [hr]
        result: 0-D field to write adaptive dt [hr]
    """
    max_D = 0.0  # [m^2/s]
    n = h.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0 or h[i, j] < 1e-6:
            continue
        h_m = h[i, j] / 1000.0  # [m]
        # Local slope from max neighbor gradient
        S_local = 1e-4  # floor
        for di, dj in ti.static(CARD):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                dz = (z[i, j] - z[ni, nj]) / dx
                S_local = ti.max(S_local, ti.abs(dz))
        D = ti.pow(h_m, 5.0 / 3.0) / (n_M * ti.sqrt(S_local))  # [m^2/s]
        ti.atomic_max(max_D, D)

    if max_D > 0.0:
        dt_cfl = cfl * dx * dx / (4.0 * max_D)  # [s]
        result[None] = ti.min(dt_cfl / 3600.0, dt_max)  # [hr]
    else:
        result[None] = dt_max


@ti.kernel
def accumulate_Q_substep(
    h: ti.template(),
    z: ti.template(),
    Q_daily: ti.template(),
    mask: ti.template(),
    dx: ti.f32,
    n_M: ti.f32,
    dt: ti.f32,
):
    """Accumulate time-weighted discharge magnitude into Q_daily.

    Computes local unit-width discharge from Manning's equation and
    adds q * dt contribution in SI units [m^2/s * hr].  The caller
    converts the final sum to [mm*m/day] once after the storm loop.

    Args:
        h: Surface water depth [mm]
        z: Elevation [m]
        Q_daily: Accumulated discharge (read/write) [m^2/s * hr]
        mask: Active cell mask
        dx: Cell spacing [m]
        n_M: Manning's roughness [s/m^(1/3)]
        dt: Substep duration [hr]
    """
    n = h.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0 or h[i, j] < 1e-6:
            continue

        h_m = h[i, j] / 1000.0  # [m]
        eta_here = z[i, j] + h_m  # [m]

        # Max outward flux magnitude as proxy for local discharge
        q_max = 0.0  # [m^2/s]
        for di, dj in ti.static(CARD):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                h_nb = h[ni, nj] / 1000.0  # [m]
                eta_there = z[ni, nj] + h_nb  # [m]
                d_eta = (eta_here - eta_there) / dx  # [-]
                if d_eta > 1e-10:
                    q = ti.pow(h_m, 5.0 / 3.0) / n_M * ti.sqrt(d_eta)  # [m^2/s]
                    q_max = ti.max(q_max, q)

        Q_daily[i, j] += q_max * dt  # [m^2/s * hr]


def step_storm(fields: Fields, params: Params, rain_mm: float):
    """Run a storm event with sub-hourly adaptive timestepping.

    Composes rainfall, Green-Ampt infiltration, and diffusion wave
    routing over CFL-limited substeps.  Writes I_inf and Q_daily
    for downstream operators.

    Args:
        fields: Simulation fields (h, h_new, F_inf, mask, z, V, ...)
        params: Physical constants
        rain_mm: Total rainfall this day [mm]
    """
    if rain_mm <= 0.0:
        fields.I_inf.fill(0.0)
        fields.Q_daily.fill(0.0)
        return

    # Storm duration from intensity
    duration_hr = rain_mm / params.storm_intensity  # [hr]

    # Reset storm state.  F_inf reset assumes soil fully drains between
    # events (dry-soil initial condition).  Known simplification: multi-day
    # storm sequences on heavy clay would retain the wetting front.
    # Richards equation will carry theta across events naturally.
    fields.h.fill(0.0)
    fields.h_new.fill(0.0)
    fields.F_inf.fill(0.0)
    fields.Q_daily.fill(0.0)

    t_elapsed = 0.0  # [hr]

    # --- Rainfall phase ---
    while t_elapsed < duration_hr:
        compute_adaptive_dt(
            fields.h,
            fields.z,
            fields.mask,
            params.dx,
            params.n_manning,
            params.cfl,
            params.dt_max,
            fields.dt_adapt,
        )
        dt_sub = fields.dt_adapt[None]
        dt_sub = min(dt_sub, duration_hr - t_elapsed)

        apply_rainfall(fields.h, fields.mask, params.storm_intensity, dt_sub)

        infiltration_green_ampt(
            fields.h,
            fields.F_inf,
            fields.V,
            fields.mask,
            params.K_s,
            params.psi_f,
            params.delta_theta,
            params.k2,
            params.W0,
            dt_sub,
        )

        diffusion_wave_step(
            fields.h,
            fields.h_new,
            fields.z,
            fields.mask,
            params.dx,
            dt_sub,
            params.n_manning,
        )
        fields.swap("h")

        accumulate_Q_substep(
            fields.h,
            fields.z,
            fields.Q_daily,
            fields.mask,
            params.dx,
            params.n_manning,
            dt_sub,
        )

        t_elapsed += dt_sub

    # --- Drainage tail (no rainfall) ---
    t_drain = 0.0
    t_drain_max = duration_hr  # drain for at most as long as the storm
    while t_drain < t_drain_max:
        compute_adaptive_dt(
            fields.h,
            fields.z,
            fields.mask,
            params.dx,
            params.n_manning,
            params.cfl,
            params.dt_max,
            fields.dt_adapt,
        )
        dt_sub = fields.dt_adapt[None]
        if dt_sub >= params.dt_max:
            break  # domain is effectively dry

        dt_sub = min(dt_sub, t_drain_max - t_drain)

        infiltration_green_ampt(
            fields.h,
            fields.F_inf,
            fields.V,
            fields.mask,
            params.K_s,
            params.psi_f,
            params.delta_theta,
            params.k2,
            params.W0,
            dt_sub,
        )

        diffusion_wave_step(
            fields.h,
            fields.h_new,
            fields.z,
            fields.mask,
            params.dx,
            dt_sub,
            params.n_manning,
        )
        fields.swap("h")

        accumulate_Q_substep(
            fields.h,
            fields.z,
            fields.Q_daily,
            fields.mask,
            params.dx,
            params.n_manning,
            dt_sub,
        )

        t_drain += dt_sub

    total_hr = t_elapsed + t_drain

    # --- Write interface fields ---
    # I_inf: cumulative infiltration over the day [mm/day]
    # F_inf already holds cumulative mm; I_inf = F_inf (one event per day)
    _copy_F_to_I(fields.F_inf, fields.I_inf, fields.mask)

    # Q_daily: normalise time-integrated discharge to daily average [mm*m/day]
    if total_hr > 0.0:
        _normalise_Q(fields.Q_daily, fields.mask, total_hr)


@ti.kernel
def _copy_F_to_I(
    F_inf: ti.template(),
    I_inf: ti.template(),
    mask: ti.template(),
):
    n = F_inf.shape[0]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 1:
            I_inf[i, j] = F_inf[i, j]
        else:
            I_inf[i, j] = 0.0


@ti.kernel
def _normalise_Q(
    Q_daily: ti.template(),
    mask: ti.template(),
    total_hr: ti.f32,
):
    """Convert time-integrated Q to daily-average discharge [mm*m/day].

    Q_daily holds sum of (q [m^2/s] * dt [hr]).
    Time-average: divide by total_hr → [m^2/s]
    Unit convert: * 1000 [mm/m] * 86400 [s/day] → [mm*m/day]
    """
    n = Q_daily.shape[0]
    scale = 1000.0 * 86400.0 / total_hr  # [mm/m * s/day / hr]
    for i, j in ti.ndrange(n, n):
        if mask[i, j] == 1:
            Q_daily[i, j] *= scale
        else:
            Q_daily[i, j] = 0.0
