"""Main simulation loop with hierarchical timestepping.

Integrates surface water routing, infiltration, soil moisture dynamics,
and vegetation dynamics with adaptive subcycling for surface processes.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from src.diagnostics import MassBalance, compute_total
from src.fields import (
    add_uniform,
    allocate,
    fill_field,
)
from src.initialization import (
    initialize_tilted_plane,
    initialize_vegetation,
)
from src.kernels.flow import (
    FLOW_EXPONENT,
    compute_cfl_timestep,
    compute_flow_directions,
    route_surface_water,
)
from src.kernels.infiltration import infiltration_step
from src.kernels.soil import soil_moisture_step
from src.kernels.vegetation import vegetation_step
from src.params import SimulationParams


@dataclass
class SimulationState:
    """Complete simulation state including fields and tracking variables."""

    fields: SimpleNamespace
    mass_balance: MassBalance = field(default_factory=MassBalance)
    current_day: float = 0.0
    dx: float = 1.0  # cell size [m]

    def total_surface_water(self) -> float:
        """Total surface water volume [m^3]."""
        return float(compute_total(self.fields.h, self.fields.mask)) * self.dx * self.dx

    def total_soil_moisture(self) -> float:
        """Total soil moisture volume [m^3]."""
        return float(compute_total(self.fields.M, self.fields.mask)) * self.dx * self.dx

    def total_water(self) -> float:
        """Total water in system (h + M) [m^3]."""
        return self.total_surface_water() + self.total_soil_moisture()

    def max_surface_water(self) -> float:
        """Maximum surface water depth [m]."""
        h_np = self.fields.h.to_numpy()
        mask_np = self.fields.mask.to_numpy()
        return float(np.max(h_np * (mask_np == 1)))


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
        """Initialize simulation with synthetic terrain and random vegetation.

        Returns the SimulationState for inspection/testing.
        """
        p = self.params

        # Create fields
        fields = allocate(p.n)

        # Initialize terrain
        initialize_tilted_plane(fields, slope=slope, direction=direction)

        # Compute flow directions
        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, p.dx, FLOW_EXPONENT
        )

        # Initialize state variables
        fill_field(fields.h, 0.0)
        fill_field(fields.M, initial_moisture)
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
        """Run surface water dynamics during a rainfall event.

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
            infiltration_step(
                fields.h, fields.M, fields.P, fields.mask,
                p.alpha, p.k_P, p.W_0, p.M_sat, dt
            )
            # Infiltration transfers h to M, no net change in water

            t += dt

    def step_soil(self, dt: float) -> None:
        """Update soil moisture: ET, leakage, diffusion.

        Args:
            dt: Timestep [days]
        """
        if self.state is None:
            raise RuntimeError("Simulation not initialized")

        p = self.params
        fields = self.state.fields
        dx2 = p.dx * p.dx

        total_et, total_leakage = soil_moisture_step(
            fields, p.E_max, p.k_ET, p.beta_ET, p.L_max, p.M_sat, p.D_M, p.dx, dt
        )

        self.state.mass_balance.cumulative_et += total_et * dx2
        self.state.mass_balance.cumulative_leakage += total_leakage * dx2

    def step_vegetation(self, dt: float) -> tuple[float, float]:
        """Update vegetation: growth, mortality, dispersal.

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
            fields, p.g_max, p.k_G, p.mu, p.D_P, p.dx, dt
        )

    def run(
        self,
        years: float,
        check_mass_balance: bool = True,
        verbose: bool = False,
    ) -> SimulationState:
        """Run simulation for specified duration.

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



