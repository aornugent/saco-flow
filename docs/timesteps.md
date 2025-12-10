# Hierarchical Time Stepping

The system spans timescales from minutes to years. Use operator splitting with nested subcycling.

## Timestep Hierarchy

```
VEGETATION (Δt_V = 7-30 days)
└── SOIL (Δt_M = 1 day)
    └── SURFACE (Δt_h = adaptive, ~minutes)
```

## Main Loop Structure

```
For each vegetation timestep:
    For each soil timestep:
        If rainfall event:
            While water present:
                1. Route surface water (CFL-limited dt)
                2. Compute infiltration
                3. Update h, M
        4. Apply ET and leakage to M
        5. Diffuse soil moisture
    6. Update vegetation (growth - mortality)
    7. Diffuse vegetation
```

## Stability Constraints

| Process | Constraint | Typical Value |
|---------|------------|---------------|
| Surface routing | `dt <= dx / v_max` | seconds-minutes |
| Soil diffusion | `dt <= dx² / (4·D_M)` | ~250 days (not limiting) |
| Vegetation diffusion | `dt <= dx² / (4·D_P)` | ~2500 days (not limiting) |

Surface routing is the only constraint that matters in practice.

## Adaptive Subcycling

Surface water routing needs small timesteps only during active flow:

```python
def run_rainfall_event(depth, duration):
    apply_rainfall(depth / duration)
    t = 0
    while t < duration + drainage_time and max(h) > threshold:
        dt = compute_CFL_timestep()
        route_surface_water(dt)
        infiltration_step(dt)
        t += dt
```

## Implementation Reference

See `ecohydro_spec.md:111-134` for timestep constraints.
See `ecohydro_spec.md:486-520` for simulation loop structure.
