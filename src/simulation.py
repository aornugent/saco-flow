"""
Main simulation loop with hierarchical timestepping.

Integrates surface water routing, infiltration, soil moisture dynamics,
and vegetation dynamics with adaptive subcycling for surface processes.

Time units: days throughout.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import taichi as ti

from src.config import DTYPE, DefaultParams
from src.kernels.flow import (
    compute_cfl_timestep,
    compute_flow_directions,
    route_surface_water,
    FLOW_EXPONENT,
)
from src.kernels.infiltration import infiltration_step
from src.kernels.soil import soil_moisture_step
from src.kernels.vegetation import vegetation_step
from src.kernels.utils import add_uniform, compute_total, fill_field


@dataclass
class MassBalance:
    """Tracks cumulative fluxes for mass conservation verification."""

    initial_water: float = 0.0  # h + M at start [m³]
    cumulative_rain: float = 0.0  # total rainfall [m³]
    cumulative_et: float = 0.0  # total evapotranspiration [m³]
    cumulative_leakage: float = 0.0  # total deep leakage [m³]
    cumulative_outflow: float = 0.0  # total boundary outflow [m³]

    def expected_water(self) -> float:
        """Compute expected total water based on fluxes."""
        return (
            self.initial_water
            + self.cumulative_rain
            - self.cumulative_et
            - self.cumulative_leakage
            - self.cumulative_outflow
        )

    def check(self, actual: float, rtol: float = 1e-4, atol: float = 1e-8) -> float:
        """
        Check mass conservation and return relative error.

        Args:
            actual: Current total water (h + M) [m³]
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Relative error

        Raises:
            AssertionError: If mass balance violated beyond tolerance
        """
        expected = self.expected_water()
        error = abs(actual - expected)
        tol = atol + rtol * abs(expected)

        if error > tol:
            raise AssertionError(
                f"Mass conservation violated!\n"
                f"  Expected: {expected:.6e} m³\n"
                f"  Actual:   {actual:.6e} m³\n"
                f"  Error:    {error:.6e} (tolerance: {tol:.6e})\n"
                f"  Rain:     {self.cumulative_rain:.6e}\n"
                f"  ET:       {self.cumulative_et:.6e}\n"
                f"  Leakage:  {self.cumulative_leakage:.6e}\n"
                f"  Outflow:  {self.cumulative_outflow:.6e}"
            )

        return error / max(expected, 1e-10)


@dataclass
class SimulationState:
    """Complete simulation state including fields and tracking variables."""

    fields: SimpleNamespace
    mass_balance: MassBalance = field(default_factory=MassBalance)
    current_day: float = 0.0
    dx: float = 1.0  # cell size [m]

    def total_surface_water(self) -> float:
        """Total surface water volume [m³]."""
        return float(compute_total(self.fields.h, self.fields.mask)) * self.dx * self.dx

    def total_soil_moisture(self) -> float:
        """Total soil moisture volume [m³]."""
        return float(compute_total(self.fields.M, self.fields.mask)) * self.dx * self.dx

    def total_water(self) -> float:
        """Total water in system (h + M) [m³]."""
        return self.total_surface_water() + self.total_soil_moisture()

    def max_surface_water(self) -> float:
        """Maximum surface water depth [m]."""
        h_np = self.fields.h.to_numpy()
        mask_np = self.fields.mask.to_numpy()
        return float(np.max(h_np * (mask_np == 1)))


@dataclass
class SimulationParams:
    """All simulation parameters with defaults from DefaultParams."""

    # Grid
    n: int = 64
    dx: float = DefaultParams.DX

    # Rainfall
    rain_depth: float = DefaultParams.R_MEAN  # [m]
    storm_duration: float = DefaultParams.STORM_DURATION  # [days]
    interstorm: float = DefaultParams.INTERSTORM  # [days]

    # Infiltration
    alpha: float = DefaultParams.K_SAT  # infiltration rate [m/day]
    k_P: float = DefaultParams.K_P  # vegetation half-sat [kg/m²]
    W_0: float = DefaultParams.W_0  # bare soil factor [-]
    M_sat: float = DefaultParams.M_SAT  # saturation [m]

    # Soil moisture
    E_max: float = DefaultParams.ET_MAX  # max ET [m/day]
    k_ET: float = DefaultParams.K_ET  # ET half-sat [m]
    beta_ET: float = DefaultParams.BETA_ET  # veg enhancement [-]
    L_max: float = DefaultParams.LEAKAGE  # leakage [1/day]
    D_M: float = DefaultParams.D_SOIL  # diffusivity [m²/day]

    # Vegetation
    g_max: float = DefaultParams.G_MAX  # growth rate [1/day]
    k_G: float = DefaultParams.K_G  # growth half-sat [m]
    mu: float = DefaultParams.MORTALITY  # mortality [1/day]
    D_P: float = DefaultParams.D_VEG  # dispersal [m²/day]

    # Surface routing
    manning_n: float = DefaultParams.MANNING_N  # roughness [-]

    # Drainage
    h_threshold: float = DefaultParams.H_THRESHOLD  # [m]
    drainage_time: float = DefaultParams.DRAINAGE_TIME  # [days]

    # Timesteps
    dt_veg: float = 7.0  # vegetation timestep [days]
    dt_soil: float = 1.0  # soil timestep [days]


def create_simulation_fields(n: int) -> SimpleNamespace:
    """Create all Taichi fields needed for simulation."""
    fields = SimpleNamespace(n=n)

    # Primary state variables
    fields.h = ti.field(dtype=DTYPE, shape=(n, n))  # Surface water [m]
    fields.M = ti.field(dtype=DTYPE, shape=(n, n))  # Soil moisture [m]
    fields.P = ti.field(dtype=DTYPE, shape=(n, n))  # Vegetation [kg/m²]

    # Static fields
    fields.Z = ti.field(dtype=DTYPE, shape=(n, n))  # Elevation [m]
    fields.mask = ti.field(dtype=ti.i8, shape=(n, n))  # Domain mask

    # Flow directions (8 neighbors)
    fields.flow_frac = ti.field(dtype=DTYPE, shape=(n, n, 8))

    # Double buffers for stencil operations
    fields.h_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.M_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.P_new = ti.field(dtype=DTYPE, shape=(n, n))

    # Flow routing fields
    fields.q_out = ti.field(dtype=DTYPE, shape=(n, n))  # Outflow rate [m/day]
    fields.flow_acc = ti.field(dtype=DTYPE, shape=(n, n))  # Flow accumulation
    fields.flow_acc_new = ti.field(dtype=DTYPE, shape=(n, n))  # Flow acc buffer
    fields.local_source = ti.field(dtype=DTYPE, shape=(n, n))  # Local contribution

    return fields


def initialize_tilted_plane(
    fields: SimpleNamespace,
    slope: float = 0.01,
    direction: str = "south",
) -> None:
    """Initialize elevation as a tilted plane with boundary mask."""
    n = fields.n
    rows = np.arange(n, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float32).reshape(1, -1)

    if direction == "south":
        Z = (n - 1 - rows) * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "north":
        Z = rows * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "east":
        Z = (n - 1 - cols) * slope * np.ones((n, 1), dtype=np.float32)
    elif direction == "west":
        Z = cols * slope * np.ones((n, 1), dtype=np.float32)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    fields.Z.from_numpy(Z.astype(np.float32))

    # Set boundary mask
    mask = np.ones((n, n), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)


def initialize_vegetation(
    fields: SimpleNamespace,
    mean: float = 0.5,
    std: float = 0.1,
    seed: int | None = None,
) -> None:
    """Initialize vegetation with random perturbation."""
    if seed is not None:
        np.random.seed(seed)

    n = fields.n
    P_np = np.random.normal(mean, std, (n, n)).astype(np.float32)
    P_np = np.clip(P_np, 0.0, None)  # Ensure non-negative
    fields.P.from_numpy(P_np)


def initialize_soil_moisture(
    fields: SimpleNamespace,
    value: float = 0.1,
) -> None:
    """Initialize uniform soil moisture."""
    fill_field(fields.M, value)


class Simulation:
    """Main simulation orchestrator with hierarchical timestepping."""

    def __init__(self, params: SimulationParams | None = None):
        self.params = params or SimulationParams()
        self.state: SimulationState | None = None

    def initialize(
        self,
        slope: float = 0.01,
        direction: str = "south",
        initial_moisture: float = 0.1,
        initial_veg_mean: float = 0.5,
        initial_veg_std: float = 0.1,
        seed: int | None = 42,
    ) -> SimulationState:
        """
        Initialize simulation with synthetic terrain and random vegetation.

        Returns the SimulationState for inspection/testing.
        """
        p = self.params

        # Create fields
        fields = create_simulation_fields(p.n)

        # Initialize terrain
        initialize_tilted_plane(fields, slope=slope, direction=direction)

        # Compute flow directions
        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, p.dx, FLOW_EXPONENT
        )

        # Initialize state variables
        fill_field(fields.h, 0.0)
        initialize_soil_moisture(fields, value=initial_moisture)
        initialize_vegetation(fields, mean=initial_veg_mean, std=initial_veg_std, seed=seed)

        # Create state
        self.state = SimulationState(
            fields=fields,
            mass_balance=MassBalance(),
            current_day=0.0,
            dx=p.dx,
        )

        # Record initial water
        self.state.mass_balance.initial_water = self.state.total_water()

        return self.state

    def run_rainfall_event(self, depth: float, duration: float) -> None:
        """
        Run surface water dynamics during a rainfall event.

        Applies rainfall as intensity over duration, routes water,
        and handles infiltration with adaptive CFL timestepping.
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized")

        p = self.params
        fields = self.state.fields
        dx2 = p.dx * p.dx

        intensity = depth / duration  # [m/day]
        t = 0.0
        max_subcycles = 10000  # Safety limit

        n_interior = int(np.sum(fields.mask.to_numpy() == 1))

        for _ in range(max_subcycles):
            # Check termination
            if t >= duration + p.drainage_time:
                break
            if t >= duration and self.state.max_surface_water() < p.h_threshold:
                break

            # Compute CFL timestep
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                p.dx, p.manning_n, cfl=0.5
            )

            # Limit timestep
            if dt == float("inf"):
                dt = 0.01  # Small step if no flow yet
            dt = min(dt, duration + p.drainage_time - t, 0.1)  # Cap at 0.1 days

            # Apply rainfall if still in event
            if t < duration:
                rain_this_step = intensity * dt
                add_uniform(fields.h, fields.mask, rain_this_step)
                self.state.mass_balance.cumulative_rain += rain_this_step * n_interior * dx2

            # Route surface water
            boundary_outflow = route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, p.dx, dt, p.manning_n
            )
            self.state.mass_balance.cumulative_outflow += boundary_outflow * dx2

            # Infiltration
            total_infiltration = infiltration_step(
                fields.h, fields.M, fields.P, fields.mask,
                p.alpha, p.k_P, p.W_0, p.M_sat, dt
            )
            # Infiltration transfers h to M, no net change in water

            t += dt

    def step_soil(self, dt: float) -> None:
        """
        Update soil moisture: ET, leakage, diffusion.

        Args:
            dt: Timestep [days]
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized")

        p = self.params
        fields = self.state.fields
        dx2 = p.dx * p.dx

        total_et, total_leakage = soil_moisture_step(
            fields.M, fields.M_new, fields.P, fields.mask,
            p.E_max, p.k_ET, p.beta_ET, p.L_max, p.M_sat, p.D_M, p.dx, dt
        )

        self.state.mass_balance.cumulative_et += total_et * dx2
        self.state.mass_balance.cumulative_leakage += total_leakage * dx2

    def step_vegetation(self, dt: float) -> tuple[float, float]:
        """
        Update vegetation: growth, mortality, dispersal.

        Args:
            dt: Timestep [days]

        Returns:
            (total_growth, total_mortality)
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized")

        p = self.params
        fields = self.state.fields

        return vegetation_step(
            fields.P, fields.P_new, fields.M, fields.mask,
            p.g_max, p.k_G, p.mu, p.D_P, p.dx, dt
        )

    def run(
        self,
        years: float,
        check_mass_balance: bool = True,
        verbose: bool = False,
    ) -> SimulationState:
        """
        Run simulation for specified duration.

        Args:
            years: Simulation duration [years]
            check_mass_balance: Verify mass conservation periodically
            verbose: Print progress

        Returns:
            Final SimulationState
        """
        if self.state is None:
            self.initialize()

        p = self.params
        days_total = years * 365.0

        # Event schedule using intensity-based rainfall
        next_event_day = np.random.exponential(p.interstorm)
        events_run = 0

        while self.state.current_day < days_total:
            # Check for rainfall event
            if self.state.current_day >= next_event_day:
                self.run_rainfall_event(p.rain_depth, p.storm_duration)
                next_event_day += np.random.exponential(p.interstorm)
                events_run += 1

            # Daily soil update
            self.step_soil(p.dt_soil)

            # Weekly vegetation update
            if int(self.state.current_day) % 7 == 0:
                self.step_vegetation(p.dt_veg)

            self.state.current_day += p.dt_soil

            # Periodic mass balance check
            if check_mass_balance and int(self.state.current_day) % 30 == 0:
                error = self.state.mass_balance.check(self.state.total_water())
                if verbose:
                    print(
                        f"Day {self.state.current_day:.0f}: "
                        f"mass error = {error:.2e}, events = {events_run}"
                    )

        if verbose:
            print(f"Simulation complete: {years} years, {events_run} rainfall events")

        return self.state

    def check_mass_balance(self) -> float:
        """Check mass conservation and return relative error."""
        if self.state is None:
            raise RuntimeError("Simulation not initialized")
        return self.state.mass_balance.check(self.state.total_water())
