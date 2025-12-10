# CLAUDE.md

GPU-accelerated ecohydrological simulation using Taichi. See `ecohydro_spec.md` for the math, `IMPLEMENTATION_PLAN.md` for current priorities.

## Principles

1. **Simplicity first** - Naive implementation before optimization. Three similar lines beat a clever abstraction.
2. **Correctness before speed** - Tests alongside implementation. Every kernel needs a mass conservation test.
3. **Explicit over implicit** - Physical variable names with units in comments. No hidden state.

## Conventions

```python
# Fields: snake_case, document units
surface_water = ti.field(dtype=ti.f32, shape=(N, N))  # [m]

# Kernels: verb phrases, docstring with governing equation
@ti.kernel
def compute_flow_directions():
    """Compute MFD fractions: f_k = S_k^p / sum(S_m^p)"""
    ...

# Parameters: group in dataclasses with units
@dataclass
class InfiltrationParams:
    alpha: float = 0.1  # infiltration rate [day⁻¹]
```

**Naming:** `snake_case` for fields/kernels, `UPPER_SNAKE_CASE` for constants, `test_<what>_<expected>` for tests.

## Key Gotchas

- **Flat cells:** Flag with `flow_frac[i,j,0] = -1.0`, handle separately
- **Boundaries:** Mask 0=inactive, 1=active. Neumann for diffusion, outflow for surface water
- **Stability:** CFL for routing (`dt <= dx/v`), diffusion limit (`dt <= dx²/4D`), always clamp to physical bounds
- **Flow accumulation:** 20-50 iterations typically sufficient; perfect convergence unnecessary

## Workflow

1. Read relevant section of `ecohydro_spec.md`
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Document with equation and units
