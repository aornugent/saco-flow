"""Main simulation loop with hierarchical timestepping.

Integrates surface routing, infiltration, soil moisture, and vegetation
dynamics with adaptive subcycling for surface processes.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import taichi as ti

from src.config import DTYPE
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
from src.params import SimulationConfig


@dataclass
class MassBalance:
    """Tracks cumulative fluxes for mass conservation verification."""

    initial_water: float = 0.0
    cumulative_rain: float = 0.0
    cumulative_et: float = 0.0
    cumulative_leakage: float = 0.0
    cumulative_outflow: float = 0.0

    def expected_water(self) -> float:
        """Expected total water based on fluxes."""
        return (
            self.initial_water
            + self.cumulative_rain
            - self.cumulative_et
            - self.cumulative_leakage
            - self.cumulative_outflow
        )

    def check(self, actual: float, rtol: float = 1e-4, atol: float = 1e-8) -> float:
        """Check mass conservation and return relative error."""
        expected = self.expected_water()
        error = abs(actual - expected)
        tol = atol + rtol * abs(expected)

        if error > tol:
            raise AssertionError(
                f"Mass conservation violated: expected={expected:.6e}, "
                f"actual={actual:.6e}, error={error:.6e}"
            )

        return error / max(expected, 1e-10)


@dataclass
class SimulationState:
    """Simulation state with fields and tracking."""

    fields: SimpleNamespace
    mass_balance: MassBalance = field(default_factory=MassBalance)
    current_day: float = 0.0
    dx: float = 1.0

    def total_surface_water(self) -> float:
        """Total surface water [m³]."""
        return float(compute_total(self.fields.h, self.fields.mask)) * self.dx * self.dx

    def total_soil_moisture(self) -> float:
        """Total soil moisture [m³]."""
        return float(compute_total(self.fields.M, self.fields.mask)) * self.dx * self.dx

    def total_water(self) -> float:
        """Total water (h + M) [m³]."""
        return self.total_surface_water() + self.total_soil_moisture()

    def max_surface_water(self) -> float:
        """Maximum surface water depth [m]."""
        h_np = self.fields.h.to_numpy()
        mask_np = self.fields.mask.to_numpy()
        return float(np.max(h_np * (mask_np == 1)))


def create_simulation_fields(n: int) -> SimpleNamespace:
    """Create Taichi fields for simulation."""
    fields = SimpleNamespace(n=n)

    # State variables
    fields.h = ti.field(dtype=DTYPE, shape=(n, n))
    fields.M = ti.field(dtype=DTYPE, shape=(n, n))
    fields.P = ti.field(dtype=DTYPE, shape=(n, n))

    # Static fields
    fields.Z = ti.field(dtype=DTYPE, shape=(n, n))
    fields.mask = ti.field(dtype=ti.i8, shape=(n, n))
    fields.flow_frac = ti.field(dtype=DTYPE, shape=(n, n, 8))

    # Buffers
    fields.h_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.M_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.P_new = ti.field(dtype=DTYPE, shape=(n, n))

    # Routing scratch
    fields.q_out = ti.field(dtype=DTYPE, shape=(n, n))
    fields.flow_acc = ti.field(dtype=DTYPE, shape=(n, n))
    fields.flow_acc_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.local_source = ti.field(dtype=DTYPE, shape=(n, n))

    return fields


def initialize_tilted_plane(
    fields: SimpleNamespace, slope: float = 0.01, direction: str = "south"
) -> None:
    """Initialize elevation as tilted plane with boundary mask."""
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

    mask = np.ones((n, n), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)


def initialize_vegetation(
    fields: SimpleNamespace, mean: float = 0.5, std: float = 0.1, seed: int | None = None
) -> None:
    """Initialize vegetation with random perturbation."""
    if seed is not None:
        np.random.seed(seed)

    n = fields.n
    P_np = np.clip(np.random.normal(mean, std, (n, n)), 0.0, None).astype(np.float32)
    fields.P.from_numpy(P_np)


def initialize_soil_moisture(fields: SimpleNamespace, value: float = 0.1) -> None:
    """Initialize uniform soil moisture."""
    fill_field(fields.M, value)


class Simulation:
    """Main simulation orchestrator."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.state: SimulationState | None = None

    @property
    def p(self) -> SimulationConfig:
        """Shorthand for config."""
        return self.config

    def initialize(
        self,
        slope: float = 0.01,
        direction: str = "south",
        initial_moisture: float = 0.1,
        initial_veg_mean: float = 0.5,
        initial_veg_std: float = 0.1,
        seed: int | None = 42,
    ) -> SimulationState:
        """Initialize simulation with synthetic terrain."""
        c = self.config

        fields = create_simulation_fields(c.n)
        initialize_tilted_plane(fields, slope=slope, direction=direction)
        compute_flow_directions(fields.Z, fields.mask, fields.flow_frac, c.dx, FLOW_EXPONENT)

        fill_field(fields.h, 0.0)
        initialize_soil_moisture(fields, value=initial_moisture)
        initialize_vegetation(fields, mean=initial_veg_mean, std=initial_veg_std, seed=seed)

        self.state = SimulationState(
            fields=fields, mass_balance=MassBalance(), current_day=0.0, dx=c.dx
        )
        self.state.mass_balance.initial_water = self.state.total_water()

        return self.state

    def run_rainfall_event(self, depth: float, duration: float) -> None:
        """Run surface dynamics during rainfall event."""
        if self.state is None:
            raise RuntimeError("Not initialized")

        c = self.config
        fields = self.state.fields
        dx2 = c.dx * c.dx

        intensity = depth / duration
        t = 0.0
        n_interior = int(np.sum(fields.mask.to_numpy() == 1))

        for _ in range(10000):
            if t >= duration + c.drainage.drainage_time:
                break
            if t >= duration and self.state.max_surface_water() < c.drainage.h_threshold:
                break

            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                c.dx, c.routing.manning_n, cfl=0.5
            )
            if dt == float("inf"):
                dt = 0.01
            dt = min(dt, duration + c.drainage.drainage_time - t, 0.1)

            if t < duration:
                rain_this_step = intensity * dt
                add_uniform(fields.h, fields.mask, rain_this_step)
                self.state.mass_balance.cumulative_rain += rain_this_step * n_interior * dx2

            boundary_outflow = route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, c.dx, dt, c.routing.manning_n
            )
            self.state.mass_balance.cumulative_outflow += boundary_outflow * dx2

            infiltration_step(
                fields.h, fields.M, fields.P, fields.mask,
                c.infiltration.alpha, c.infiltration.k_P, c.infiltration.W_0,
                c.soil.M_sat, dt
            )

            t += dt

    def step_soil(self, dt: float) -> None:
        """Update soil moisture."""
        if self.state is None:
            raise RuntimeError("Not initialized")

        c = self.config
        fields = self.state.fields
        dx2 = c.dx * c.dx

        total_et, total_leakage = soil_moisture_step(
            fields.M, fields.M_new, fields.P, fields.mask,
            c.soil.E_max, c.soil.k_ET, c.soil.beta_ET,
            c.soil.L_max, c.soil.M_sat, c.soil.D_M, c.dx, dt
        )

        self.state.mass_balance.cumulative_et += total_et * dx2
        self.state.mass_balance.cumulative_leakage += total_leakage * dx2

    def step_vegetation(self, dt: float) -> tuple[float, float]:
        """Update vegetation."""
        if self.state is None:
            raise RuntimeError("Not initialized")

        c = self.config
        fields = self.state.fields

        return vegetation_step(
            fields.P, fields.P_new, fields.M, fields.mask,
            c.vegetation.g_max, c.vegetation.k_G, c.vegetation.mu,
            c.vegetation.D_P, c.dx, dt
        )

    def run(
        self, years: float, check_mass_balance: bool = True, verbose: bool = False
    ) -> SimulationState:
        """Run simulation for specified duration."""
        if self.state is None:
            self.initialize()

        c = self.config
        days_total = years * 365.0

        next_event_day = np.random.exponential(c.rainfall.interstorm)
        events_run = 0

        while self.state.current_day < days_total:
            if self.state.current_day >= next_event_day:
                self.run_rainfall_event(c.rainfall.rain_depth, c.rainfall.storm_duration)
                next_event_day += np.random.exponential(c.rainfall.interstorm)
                events_run += 1

            self.step_soil(c.timestep.dt_soil)

            if int(self.state.current_day) % 7 == 0:
                self.step_vegetation(c.timestep.dt_veg)

            self.state.current_day += c.timestep.dt_soil

            if check_mass_balance and int(self.state.current_day) % 30 == 0:
                error = self.state.mass_balance.check(self.state.total_water())
                if verbose:
                    print(f"Day {self.state.current_day:.0f}: error={error:.2e}")

        if verbose:
            print(f"Complete: {years} years, {events_run} events")

        return self.state

    def check_mass_balance(self) -> float:
        """Check mass conservation."""
        if self.state is None:
            raise RuntimeError("Not initialized")
        return self.state.mass_balance.check(self.state.total_water())
