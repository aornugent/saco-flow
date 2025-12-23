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
| Understanding the system | `docs/overview.md` |
| Physics and parameters | `docs/physics.md` |
| Numerical methods | `docs/discretization.md` |
| Code structure / GPU | `docs/architecture.md` |
| Flow/routing kernels | `docs/kernels/flow.md` |
| Soil moisture kernels | `docs/kernels/soil.md` |
| Vegetation kernels | `docs/kernels/vegetation.md` |
| Testing/debugging | `docs/testing.md` |
| Performance data | `docs/BENCHMARKS.md` |

## Development Setup

We use `uv` for dependency management.

```bash
# Install dependencies into .venv
uv pip install -r requirements.txt

# Run tests
source .venv/bin/activate
pytest

# Run benchmarks
source .venv/bin/activate
python -m benchmarks.benchmark
```

## Conventions

- **Fields:** `snake_case`, document units in comments
- **Kernels:** `snake_case` verb phrases, docstring with equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>`
- **Optimized kernels:** suffix `_fused`, keep naive version for regression testing
- **Code organization:** Do NOT use break comments (`====`) for section separators. Use clear function/class organization and whitespace instead.

## Kernel Pattern

```python
@ti.kernel
def update_soil_moisture(
    M: ti.template(),       # Always pass fields as ti.template()
    M_new: ti.template(),
    mask: ti.template(),
    M_sat: ti.f32,
    dt: ti.f32,
):
    """Update soil moisture. Physics: dM/dt = -ET - L + D∇²M"""
    n = M.shape[0]
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue
        # ... compute fluxes ...
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M_local + dM))
```

Key points: pass fields as `ti.template()`, check mask, clamp to physical bounds, use double buffering for stencils.

## Gotchas

- **ti.template() required**: Pass fields as `ti.template()` arguments, never capture in closures. Closure-captured fields are baked in at JIT compile time — `swap_buffers()` won't work!
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

## Running Tests

```bash
# Fast tests only (skip multi-year simulations)
pytest -m "not slow"

# All tests including slow validation tests
pytest

# Specific test file
pytest tests/test_flow.py

# With coverage
pytest --cov=src --cov-report=term-missing
```

**Test markers:**
- `@pytest.mark.slow` — Multi-year simulations (minutes to run). Used for pattern emergence, parameter sensitivity, and long-term stability tests.
- Unmarked tests — Unit tests and short integration tests (seconds to run).
