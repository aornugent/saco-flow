# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi (B200/H100).

## Principles
1. **Correctness before speed** — Every kernel needs a mass conservation test
2. **Explicit over implicit** — Physical variable names, units in comments
3. **Taichi-idiomatic** — Kernels must be GPU only, avoid CPU transfer bottlenecks

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

## 🚫 STRICT BAN ON META-COMMENTARY 
**You are prohibited from narrating your thought process, doubts, or verification steps in code comments**
 - Code comments must describe the `code`, never the `coder`.
 - Use tools to read the file *before* you start writing code.
 - Write comments as established facts. Never use phrases like "Assuming," "Ideally," "But wait," or "For now."
 - Do not leave breadcrumbs of your investigation (e.g., `# Checking geometry.py for offsets`). 
 - The user only wants the result, not the history of how you found it.

## Critical Rules
- **Always pass fields as `ti.template()`** — Closure capture bakes fields at JIT time, breaking `swap_buffers()`
- **Check mask before neighbor access** — `if mask[ni, nj] == 1` prevents out-of-domain reads
- **Stencil ops use double buffer** — Read from `field`, write to `field_new`, then swap_buffers
- **Point-wise ops can be in-place** — No neighbor reads means no race conditions
- **Clamp to physical bounds** — `ti.max(0.0, ti.min(M_sat, value))`
- **CFL stability** — Routing: `dt ≤ dx/v`, Diffusion: `dt ≤ dx²/4D`

## Workflow
1. Read relevant doc from table above
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Run `ruff check --fix . && ruff format .` before committing
5. Document kernel with equation and units
