"""CLI entry point for Saco-Flow simulation.

Orchestrates the simulation loop with optional 3D visualization.
Handles the multi-scale coupling of rainfall, soil moisture, and vegetation.
"""

import argparse
import time
import sys
from pathlib import Path
import tomllib
import numpy as np
import taichi as ti

from src.fields import add_uniform
from src.gui import Visualizer3D
from src.kernels.flow import compute_cfl_timestep, route_surface_water
from src.kernels.infiltration import infiltration_step
from src.params import SimulationParams
from src.simulation import Simulation
from src.output import save_simulation_output


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def update_params_from_config(params: SimulationParams, config: dict, ignore_keys: list[str] = None):
    """Update SimulationParams from config dictionary."""
    ignore_keys = ignore_keys or []
    for key, value in config.items():
        if key in ignore_keys:
            continue
        if hasattr(params, key):
            setattr(params, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}'")


def run_rainfall_event_vis(sim: Simulation, vis: Visualizer3D, depth: float, duration: float):
    """Run rainfall event with visualization updates.
    
    This duplicates logic from Simulation.run_rainfall_event to inject
    visualization calls into the inner loop.
    """
    p = sim.params
    fields = sim.state.fields
    dx2 = p.dx * p.dx

    intensity = depth / duration
    t = 0.0
    max_subcycles = 10000

    # Needed for mass balance tracking
    from src.diagnostics import compute_total
    n_interior = int(compute_total(fields.mask, fields.mask))

    last_v_max = 0.0
    first_step = True

    for _ in range(max_subcycles):
        # Handle GUI events
        if vis.window.is_pressed(ti.ui.ESCAPE):
            vis.window.destroy()
            return
            
        # Check termination
        if t >= duration + p.drainage_time:
            break
        if t >= duration and sim.state.max_surface_water() < p.h_threshold:
            break

        # Compute max global timestep (CFL)
        if first_step:
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                p.dx, p.manning_n, cfl=0.5
            )
            first_step = False
        else:
            if last_v_max < 1e-10:
                dt = float("inf")
            else:
                dt = 0.5 * p.dx / last_v_max

        if dt == float("inf"):
            dt = 0.01
        dt = min(dt, duration + p.drainage_time - t, 0.1)

        # Apply rain
        if t < duration:
            rain_this_step = intensity * dt
            add_uniform(fields.h, fields.mask, rain_this_step)
            sim.state.mass_balance.cumulative_rain += rain_this_step * n_interior * dx2

        # Route water
        boundary_outflow, last_v_max = route_surface_water(
            fields.h, fields.Z, fields.flow_frac, fields.mask,
            fields.q_out, p.dx, dt, p.manning_n
        )
        sim.state.mass_balance.cumulative_outflow += boundary_outflow * dx2

        # Infiltration
        infiltration_step(
            fields.h, fields.M, fields.P, fields.mask,
            p.alpha, p.k_P, p.W_0, p.M_sat, dt
        )

        t += dt
        
        # Update Visualization
        vis.update(sim.state, p)
        vis.render()


def main():
    parser = argparse.ArgumentParser(description="Saco-Flow Simulation")
    parser.add_argument("--config", type=str, help="Path to TOML configuration file")
    parser.add_argument("--gui", action="store_true", help="Enable 3D visualization")
    parser.add_argument("--n", type=int, help="Grid size (NxN). Overrides config.")
    parser.add_argument("--years", type=float, help="Simulation duration in years. Overrides config.")
    parser.add_argument("--slope", type=float, help="Initial terrain slope. Overrides config.")
    parser.add_argument("--output", type=str, help="Output directory (optional)")
    
    args = parser.parse_args()

    # Initialize Taichi
    if args.gui:
        ti.init(arch=ti.gpu)
    else:
        # Fallback order: cuda -> vulkan -> cpu
        try:
            ti.init(arch=ti.cuda)
        except:
            ti.init(arch=ti.vulkan)

    # Setup parameters
    params = SimulationParams()
    
    # 1. Load config if present
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
        update_params_from_config(
            params,
            config,
            ignore_keys=["years", "slope", "snapshot_interval"]
        )
    
    # 2. CLI overrides
    if args.n:
        params.n = args.n
    
    # NOTE: years and slope are not simple params fields yet, but used in init/run.
    # We should respect config for these if possible, but params struct defines defaults.
    # Slope is init param, years is run param.
    # If they are in config, we should probably read them from config dict if not in CLI.
    # Or strict mapping?
    # Usually SimulationParams contains physics/grid params. 
    # Slope and years are setup/runtime args.
    # Let's allow them in config but handle separately if they are not in params.
    
    run_years = 1.0
    init_slope = 0.01
    
    if args.config:
        # Re-read for non-param fields
        # (update_params_from_config only updates attributes of SimulationParams)
        config = load_config(args.config)
        if "years" in config:
            run_years = float(config["years"])
        if "slope" in config:
            init_slope = float(config["slope"])
            
    if args.years is not None:
        run_years = args.years
    if args.slope is not None:
        init_slope = args.slope
        
    # Initialize Simulation
    sim = Simulation(params)
    sim.initialize(slope=init_slope)
    
    vis = None
    if args.gui:
        # Decoupled resolution: use smaller of N or 512
        vis_n = min(params.n, 512)
        vis = Visualizer3D(vis_n=vis_n)
        print(f"Visualization initialized: Phys={params.n}x{params.n}, Vis={vis_n}x{vis_n}")

    # Output setup
    snapshot_dir = None
    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        snapshot_dir = out_path / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        # Save initial state
        save_simulation_output(
            sim.state.fields,
            args.output,
            dx=params.dx,
            day=sim.state.current_day
        )
        # Also save 0th snapshot
        save_simulation_output(
            sim.state.fields,
            snapshot_dir,
            prefix="snap",
            dx=params.dx,
            day=sim.state.current_day
        )

    # Configurable snapshot interval (default 30 days)
    snapshot_interval = 30.0
    if args.config:
        if "snapshot_interval" in config:
            snapshot_interval = float(config["snapshot_interval"])

    print(f"Starting simulation for {run_years} years...")
    start_time = time.time()
    
    days_total = run_years * 365.0
    next_event_day = np.random.exponential(params.interstorm)
    events_run = 0
    last_snapshot_idx = 0
    
    try:
        while sim.state.current_day < days_total:
            # Check GUI exit
            if vis and not vis.window.is_running:
                break
                
            # Rainfall Event
            if sim.state.current_day >= next_event_day:
                if vis:
                    run_rainfall_event_vis(sim, vis, params.rain_depth, params.storm_duration)
                else:
                    sim.run_rainfall_event(params.rain_depth, params.storm_duration)
                    
                next_event_day += np.random.exponential(params.interstorm)
                events_run += 1
                
                if not args.gui:
                    print(f"Event {events_run}: Day {sim.state.current_day:.1f}")

            # Soil step
            sim.step_soil(params.dt_soil)
            
            # Vegetation step (weekly)
            if int(sim.state.current_day) % 7 == 0:
                sim.step_vegetation(params.dt_veg)
                # Update Vis on veg change (if not raining)
                if vis:
                    vis.update(sim.state, params)
                    vis.render()

            sim.state.current_day += params.dt_soil
            
            # Heartbeat Snapshots (Configurable Interval)
            current_snapshot_idx = int(sim.state.current_day / snapshot_interval)
            if current_snapshot_idx > last_snapshot_idx:
                if snapshot_dir:
                    print(f"Saving snapshot {current_snapshot_idx} (Day {sim.state.current_day:.1f})...")
                    save_simulation_output(
                        sim.state.fields,
                        snapshot_dir,
                        prefix="snap",
                        dx=params.dx,
                        day=sim.state.current_day
                    )
                last_snapshot_idx = current_snapshot_idx

            # Periodic Status Check (also monthly, aligns with heartbeat)
            if int(sim.state.current_day) % 30 == 0:
                if not args.gui:
                    try:
                        error = sim.check_mass_balance()
                        print(f"Day {sim.state.current_day:.0f}: Mass Error = {error:.2e}")
                    except AssertionError as e:
                        print(f"Warning: {e}")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    # Save final state
    if args.output:
        save_simulation_output(
            sim.state.fields,
            args.output,
            dx=params.dx,
            day=sim.state.current_day
        )
    
    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")
    print(f"Simulated {sim.state.current_day/365.0:.2f} years")

if __name__ == "__main__":
    main()
