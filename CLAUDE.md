# CLAUDE.md

GPU-accelerated ecohydrological simulation using Taichi (H100/B200).

## Principles

1. **Simplicity first** - Naive implementation before optimization
2. **Correctness before speed** - Every kernel needs a mass conservation test
3. **Explicit over implicit** - Physical variable names, units in comments

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
| Current priorities | `IMPLEMENTATION_PLAN.md` |

## Conventions

- **Fields:** `snake_case`, document units in comments
- **Kernels:** `snake_case` verb phrases, docstring with equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>`

## Workflow

1. Read relevant doc from table above
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Document with equation and units
