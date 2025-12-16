# AGENTS.md

GPU-accelerated ecohydrological simulation using Taichi (B200/H100).

## Current Phase: 6 (GPU Optimization)

We have a working end-to-end simulation (140 tests passing). Now optimizing for throughput on B200 (sm_100) with H100 (sm_90) fallback. Target: 10k×10k grid at ≥1 simulated year/minute.

## Principles

1. **Simplicity first** — Naive implementation before optimization
2. **Correctness before speed** — Every kernel needs a mass conservation test
3. **Explicit over implicit** — Physical variable names, units in comments
4. **Memory bandwidth is king** — Simulation is memory-bound; minimize global memory traffic
5. **Numerical equivalence** — Optimized kernels must match naive results within tolerance

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
| GPU optimization | `docs/gpu_optimization.md`, `ecohydro_spec.md:609-714` |
| Current priorities | `IMPLEMENTATION_PLAN.md` |

## Conventions

- **Fields:** `snake_case`, document units in comments
- **Kernels:** `snake_case` verb phrases, docstring with equation
- **Constants:** `UPPER_SNAKE_CASE`
- **Tests:** `test_<what>_<expected>`
- **Optimized kernels:** suffix `_fused` or `_optimized`, keep naive version for regression testing

## Kernel Patterns

### Naive Pattern (Phase 1-5)
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

### Optimized Pattern (Phase 6)
```python
@ti.kernel
def soil_update_fused(M: ti.template(), M_new: ti.template(), P: ti.template(),
                      mask: ti.template(), dt: DTYPE) -> ti.types.vector(2, DTYPE):
    """Fused soil update: ET + leakage + diffusion in single pass."""
    ti.block_local(M)  # Shared memory caching for stencil

    totals = ti.Vector([0.0, 0.0])
    for i, j in ti.ndrange((1, n-1), (1, n-1)):  # j innermost for coalescing
        if mask[i, j] == 0:
            continue
        # Load all needed values once
        M_c, P_c = M[i, j], P[i, j]
        # Compute all updates
        et = ...
        leak = ...
        laplacian = M[i-1,j] + M[i+1,j] + M[i,j-1] + M[i,j+1] - 4*M_c
        diff = D_M * laplacian / (dx*dx) * dt
        # Single write
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M_c - et - leak + diff))
        ti.atomic_add(totals[0], et)
        ti.atomic_add(totals[1], leak)
    return totals
```

Key optimizations:
- **Kernel fusion:** Combine multiple passes into one
- **Shared memory:** `ti.block_local()` for stencil caching
- **Coalesced access:** `j` (column) varies fastest
- **Reduce atomics:** Accumulate locally, atomic once per thread block when possible

## GPU Optimization Gotchas

- **Memory layout:** Taichi uses row-major by default; iterate `(i, j)` with j innermost
- **Block size:** Use `ti.block_dim(256)` or `ti.block_dim(512)` for compute kernels
- **Shared memory limits:** 48KB per block on H100/B200; plan stencil caching accordingly
- **Register pressure:** Fused kernels use more registers; watch occupancy
- **Atomic contention:** Global atomics are slow; use thread-block reduction patterns
- **Kernel launch overhead:** ~10μs per launch; fuse small kernels

## Gotchas (Physics)

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

### For Physics Tasks (Phase 1-5 style)
1. Read relevant doc from table above
2. Write test first (conservation + edge cases)
3. Implement simply, verify mass balance
4. Document with equation and units

### For Optimization Tasks (Phase 6)
1. Write regression test comparing optimized vs naive kernel
2. Implement optimized version
3. Verify numerical equivalence within tolerance (1e-5)
4. Benchmark on GPU (1k, 5k, 10k grids)
5. Document speedup in `docs/gpu_optimization.md`

## Hardware Targets

| GPU | SM | HBM | Bandwidth | Notes |
|-----|-----|-----|-----------|-------|
| B200 | sm_100 | 192GB HBM3e | 8 TB/s | Primary target |
| H100 | sm_90 | 80GB HBM3 | 3.35 TB/s | Fallback target |

Target: >50% of theoretical memory bandwidth
