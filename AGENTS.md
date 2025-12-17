# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi (B200/H100).

## Principles

1. **Simplicity first** — Naive implementation before optimization
2. **Correctness before speed** — Every kernel needs a mass conservation test
3. **Explicit over implicit** — Physical variable names, units in comments
4. **Taichi-idiomatic** — Leverage `ti.template()`, `ti.static()`, ping-pong buffers

## Before Starting Work

Read the docs relevant to your task:

| Task | Read |
|------|------|
| Understanding the system | `docs/overview.md`, `ecohydro_spec.md` |
| Flow direction/routing | `docs/kernels/flow_directions.md`, `docs/kernels/surface_routing.md` |
| Flow accumulation | `docs/kernels/flow_accumulation.md` |
| Infiltration | `docs/kernels/infiltration.md` |
| Soil moisture | `docs/kernels/soil_moisture.md` |
| Vegetation | `docs/kernels/vegetation.md` |
| Boundary handling | `docs/boundaries.md` |
| Timestep strategy | `docs/timesteps.md` |
| Memory layout | `docs/data_structures.md` |
| Testing/debugging | `docs/mass_conservation.md` |
| Architecture & GPU optimization | `docs/ARCHITECTURE.md` |
| Current priorities | `IMPLEMENTATION_PLAN.md` |

## Conventions

- **Fields:** `snake_case`, document units in comments
- **Kernels:** `snake_case` verb phrases, docstring with equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>`
- **Optimized kernels:** suffix `_fused`, keep naive version for regression testing

## Kernel Pattern

```python
@ti.kernel
def update_soil_moisture(dt: ti.f32):
    """Update soil moisture. Physics: dM/dt = -ET - L + D∇²M"""
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        # ... compute fluxes ...
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M_local + dM))
```

Key points: check mask, clamp to physical bounds, use double buffering for stencils.

## Gotchas

- **Flat cells**: Flag with `flow_frac[i,j,0] = -1.0` — no downslope neighbors
- **Boundaries**: Always check `mask[ni, nj] == 1` before reading neighbors
- **Stability**: CFL for routing `dt ≤ dx/v`, diffusion `dt ≤ dx²/4D`
- **Conservation**: If mass isn't conserved, check boundary fluxes and clamping

## Debugging Mass Errors

1. Test on flat terrain with uniform initial conditions
2. Track each flux term separately (infiltration, ET, leakage, outflow)
3. Check boundary cells — water escaping without being counted?
4. Verify timestep satisfies stability constraints

## Workflow

1. Read relevant doc from table above
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Document with equation and units
