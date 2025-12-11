# Hierarchical Time Stepping

The system spans timescales from fractions of a day (surface flow) to years (vegetation patterns).
All times are in **days**. Use operator splitting with nested subcycling.

## Timestep Hierarchy

```
VEGETATION (Δt_V = 7 days)
└── SOIL (Δt_M = 1 day)
    └── SURFACE (Δt_h = adaptive, fractions of a day)
```

## Main Loop Structure

```
For each vegetation timestep (7 days):
    For each soil timestep (1 day):
        If rainfall event:
            While water present (h > H_THRESHOLD):
                1. Route surface water (CFL-limited dt)
                2. Compute infiltration
                3. Update h, M
        4. Apply ET and leakage to M
        5. Diffuse soil moisture
    6. Update vegetation (growth - mortality)
    7. Diffuse vegetation (seed dispersal)
```

## Stability Constraints

| Process | Constraint | Typical Value (days) |
|---------|------------|----------------------|
| Surface routing | `dt <= dx / v_max` | 0.0001-0.01 |
| Soil diffusion | `dt <= dx² / (4·D_M)` | ~2.5 (not limiting) |
| Vegetation diffusion | `dt <= dx² / (4·D_P)` | ~25 (not limiting) |

Surface routing is the only constraint that matters in practice.

## Adaptive Subcycling

Surface water routing needs small timesteps only during active flow:

```python
def run_rainfall_event(depth: float, duration: float):
    """
    Run surface water dynamics during a rainfall event.

    Args:
        depth: Total rainfall depth [m]
        duration: Event duration [days]
    """
    intensity = depth / duration  # [m/day]
    t = 0.0
    while t < duration + DRAINAGE_TIME and max(h) > H_THRESHOLD:
        dt = compute_cfl_timestep()
        if t < duration:
            add_rainfall(intensity * dt)
        route_surface_water(dt)
        infiltration_step(dt)
        t += dt
```

## Implementation Reference

See `ecohydro_spec.md:111-134` for timestep constraints.
See `ecohydro_spec.md:486-520` for simulation loop structure.
