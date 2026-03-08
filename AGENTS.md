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
- Stencil ops: read from `field`, write to `field_new`, caller swaps after
- Point-wise ops can be in-place (no neighbor reads, no race conditions)
- Check `mask[ni, nj] == 1` before reading neighbors
- Clamp to physical bounds: `ti.max(0.0, ...)`
- CFL stability: `dt <= dx^2 / (4*D)` for diffusion

## Style

- Write the physics, not the plumbing
- No file until it has two uses. No package until it has two modules
- Comments describe code, not the coder's thought process
- Variable names carry units: `D  # [m^2/day]`
- Write tests first: `test_<what>_<expected>`
