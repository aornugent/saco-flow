# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi.

## Commands

```bash
uv sync && source .venv/bin/activate
pytest
ruff check --fix . && ruff format .
```

## Taichi Rules

- Pass fields as `ti.template()` — closure capture breaks `swap_buffers()`
- Swap buffers in Python scope (`a, b = b, a`), not with a copy kernel
- Stencil ops: read from `field`, write to `field_new`, caller swaps after
- Point-wise ops can be in-place (no neighbor reads, no race conditions)
- Check `mask[ni, nj] == 1` before reading neighbors
- Clamp to physical bounds: `ti.max(0.0, ...)`
- CFL stability: `dt <= dx^2 / (4*D)` for diffusion; global dt = min across processes
- One kernel per physical process, compose in Python
- `ti.static()` for neighbor stencils, layer loops, and process toggles
- SoA by default: separate `ti.field()` per quantity, no struct packing
- `ti.block_local()` is not always faster — benchmark before keeping
- `ti.loop_config(block_dim=)` — tune per kernel, 1024 is not always optimal
- Never `to_numpy()` in the hot loop; use 0-D fields for running totals
- Allocate all fields in one place; no scattered allocation after `ti.init()`
- Test on CPU with `debug=True`, bench on GPU without
- Each unique `ti.template()` arg triggers a compile — keep variants small and exercise them in warmup

## Numerics

- Constitutive relations drive the iteration; conservation laws close the books
- When a quantity is determined by a conservation law, compute it from the balance, not the constitutive relation

## Style

- Write the physics, not the plumbing
- No file until it has two uses. No package until it has two modules
- Comments describe code, not the coder's thought process
- Variable names carry units: `D  # [m^2/day]`
- Write tests first: `test_<what>_<expected>`
