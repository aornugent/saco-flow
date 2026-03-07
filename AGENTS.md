# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi (B200/H100).

## Principles
1. **Correctness before speed** — Every kernel needs a mass conservation test
2. **Explicit over implicit** — Physical variable names, units in comments
3. **Taichi-idiomatic** — Kernels must be GPU-compatible, avoid CPU transfer bottlenecks

## Development Commands

```bash
# Setup
uv sync && source .venv/bin/activate

# Linting (fix automatically)
ruff check --fix . && ruff format .

# Tests
pytest                         # All tests
pytest tests/test_diffusion.py -v  # Specific file, verbose
pytest --cov=src              # With coverage

# Benchmarks
python -m benchmarks.run                    # Run all benchmarks
python -m benchmarks.run diffusion          # Run specific benchmark
python -m benchmarks.run --profile          # With Taichi kernel profiler
```

## Code Style
- **Fields:** `snake_case`, units in trailing comment: `M = ti.field(...)  # soil moisture [m]`
- **Kernels:** `snake_case` verb phrases, docstring with physics equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>` (e.g., `test_diffusion_conserves_mass`)
- **No section separators:** Don't use `# ====` break comments. Use whitespace.
- **Commit messages:** Imperative mood, max 72 chars

## STRICT BAN ON META-COMMENTARY
**You are prohibited from narrating your thought process, doubts, or verification steps in code comments**
 - Code comments must describe the `code`, never the `coder`.
 - Use tools to read the file *before* you start writing code.
 - Write comments as established facts. Never use phrases like "Assuming," "Ideally," "But wait," or "For now."
 - Do not leave breadcrumbs of your investigation.

## Critical Rules
- **Always pass fields as `ti.template()`** — Closure capture bakes fields at JIT time, breaking `swap_buffers()`
- **Check mask before neighbor access** — `if mask[ni, nj] == 1` prevents out-of-domain reads
- **Stencil ops use double buffer** — Read from `field`, write to `field_new`, then `swap_buffers`
- **Point-wise ops can be in-place** — No neighbor reads means no race conditions
- **Clamp to physical bounds** — `ti.max(0.0, ti.min(upper, value))`
- **CFL stability** — Diffusion: `dt <= dx^2 / (4*D)`

## Workflow
1. Write test first (conservation + edge cases)
2. Implement simply, verify mass balance
3. Run `ruff check --fix . && ruff format .` before committing
4. Document kernel with equation and units
