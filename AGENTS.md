# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi (B200/H100).

## Principles

1. **Simplicity first** — Naive implementation before optimization
2. **Correctness before speed** — Every kernel needs a mass conservation test
3. **Explicit over implicit** — Physical variable names, units in comments
4. **Taichi-idiomatic** — Use `ti.template()`, `ti.static()`, ping-pong buffers

## Before Starting Work

Read the docs relevant to your task:

| Task | Read |
|------|------|
| Understanding the system | `docs/overview.md` |
| Physics and parameters | `docs/physics.md` |
| Numerical methods | `docs/discretization.md` |
| Code structure / GPU | `docs/architecture.md` |
| Flow/routing kernels | `docs/kernels/flow.md` |
| Soil moisture kernels | `docs/kernels/soil.md` |
| Vegetation kernels | `docs/kernels/vegetation.md` |
| Testing/debugging | `docs/testing.md` |
| Performance data | `docs/BENCHMARKS.md` |

## Development Commands

```bash
# Setup
uv sync && source .venv/bin/activate

# Linting (fix automatically)
ruff check --fix . && ruff format .

# Tests
pytest -m "not slow"          # Fast tests only (~seconds)
pytest                         # All tests including slow (~minutes)
pytest tests/test_flow.py -v  # Specific file, verbose
pytest --cov=src              # With coverage

# Benchmarks & Profiling
python -m benchmarks.run                           # Run all benchmarks
python -m benchmarks.run scaling                   # Run specific benchmark
python -m benchmarks.run --profile                 # Run with Taichi kernel profiler
```

## Code Style

- **Fields:** `snake_case`, units in trailing comment: `h = ti.field(...)  # surface water [m]`
- **Kernels:** `snake_case` verb phrases, docstring with physics equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>` (e.g., `test_diffusion_conserves_mass`)
- **Fused kernels:** suffix `_fused`, keep naive version for regression testing
- **No section separators:** Don't use `# ====` break comments. Use whitespace.
- **Commit messages:** Imperative mood, max 72 chars (e.g., "Add vegetation dispersal kernel")

## Critical Rules

- **Always pass fields as `ti.template()`** — Closure capture bakes fields at JIT time, breaking `swap_buffers()`
- **Check mask before neighbor access** — `if mask[ni, nj] == 1` prevents out-of-domain reads
- **Stencil ops use double buffer** — Read from `field`, write to `field_new`, then swap
- **Point-wise ops can be in-place** — No neighbor reads means no race conditions
- **Clamp to physical bounds** — `ti.max(0.0, ti.min(M_sat, value))`
- **CFL stability** — Routing: `dt ≤ dx/v`, Diffusion: `dt ≤ dx²/4D`

## Debugging Mass Errors

1. Test on flat terrain with uniform initial conditions
2. Track each flux term separately (infiltration, ET, leakage, outflow)
3. Check boundary cells — water escaping without being counted?
4. Verify timestep satisfies stability constraints
5. Run `sim.run(years=1, check_mass_balance=True, verbose=True)` for per-month diagnostics

## Workflow

1. Read relevant doc from table above
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Run `ruff check --fix . && ruff format .` before committing
5. Document kernel with equation and units
