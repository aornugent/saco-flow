# Implementation Plan: GPU Optimization

**Goal:** 10k×10k grid at ≥1 simulated year per wall-clock minute on H100/B200.

**Read:** `docs/ARCHITECTURE.md` (buffer strategy, optimization notes)

---

## Current State

Core simulation working:
- Surface routing (MFD, kinematic wave)
- Infiltration (vegetation-enhanced)
- Soil moisture (ET, leakage, diffusion)
- Vegetation dynamics (growth, mortality, dispersal)
- Mass conservation verified

The simulation is memory-bandwidth bound (~0.2-0.5 FLOP/byte).

---

## Phase 1: Ping-Pong Buffers

Eliminate `copy_field()` overhead by implementing true ping-pong buffering.

| Task | File |
|------|------|
| Add `_get_buffers()` helper to Simulation | `src/simulation.py` |
| Update `soil_moisture_step` to accept buffer pairs | `src/kernels/soil.py` |
| Update `vegetation_step` to accept buffer pairs | `src/kernels/vegetation.py` |
| Verify mass conservation still passes | `tests/` |

**Exit:** No `copy_field()` calls in main loop. Tests pass.

---

## Phase 2: Memory Access Audit

Ensure coalesced access patterns before fusion.

| Task | File |
|------|------|
| Verify loop order (j innermost for row-major) | `src/kernels/*.py` |
| Add `ti.block_dim()` hints where beneficial | `src/kernels/*.py` |

**Exit:** All kernels use coalesced access.

---

## Phase 3: Point-Wise Kernel Fusion ✓

Fuse sequential point-wise operations to reduce memory traffic.

| Task | File | Status |
|------|------|--------|
| Consolidate diffusion into generic `laplacian_diffusion_step` | `src/geometry.py` | ✓ |
| Fuse `evapotranspiration_step` + `leakage_step` → `et_leakage_step_fused` | `src/kernels/soil.py` | ✓ |
| Fuse `growth_step` + `mortality_step` → `growth_mortality_step_fused` | `src/kernels/vegetation.py` | ✓ |
| Add equivalence tests (fused vs sequential) | `tests/test_kernel_equivalence.py` | ✓ |
| Keep naive kernels for regression testing | `src/kernels/*.py` | ✓ |

**Before:** 8 reads + 4 writes = 48 bytes/cell
**After:** 4 reads + 2 writes = 24 bytes/cell (2× improvement)

**Exit:** Fused kernels match naive within 1e-5 tolerance. ✓

---

## Phase 4: Shared Memory for Stencils ✓

Add `ti.block_local()` caching for diffusion stencils.

**Consolidation:** Removed duplicate diffusion kernels (`diffusion_step` from soil.py,
`vegetation_diffusion_step` from vegetation.py). Both now use the single generic
`laplacian_diffusion_step` in geometry.py with `ti.block_local()` caching.

| Task | File | Status |
|------|------|--------|
| Remove duplicate `diffusion_step` | `src/kernels/soil.py` | ✓ |
| Remove duplicate `vegetation_diffusion_step` | `src/kernels/vegetation.py` | ✓ |
| Add `ti.block_local(field)` to generic diffusion | `src/geometry.py` | ✓ |
| Update naive step functions to use generic kernel | `src/kernels/*.py` | ✓ |
| Create benchmark harness | `benchmarks/benchmark_diffusion.py` | ✓ |
| Update tests for consolidated kernels | `tests/*.py` | ✓ |

**Before:** 3 duplicate diffusion kernels
**After:** 1 generic kernel with shared memory optimization

**Exit:** Measurable bandwidth improvement on large grids. ✓

---

## Phase 5: 10k×10k Validation

Verify execution at target scale.

| Task | File |
|------|------|
| Memory allocation test | `tests/test_large_grid.py` |
| End-to-end 10k×10k run (1 year) | `benchmarks/` |
| Verify mass conservation at scale | `tests/` |

**Memory budget:** ~7.7 GB (< 10% of H100's 80GB)

**Exit:** 10k×10k runs without errors, mass conserved.

---

## Phase 6: Benchmarking

Systematic performance measurement.

| Task | File |
|------|------|
| Benchmark harness with warmup | `benchmarks/benchmark.py` |
| Measure cells/second, years/minute | `benchmarks/` |
| Compare achieved vs theoretical bandwidth | `benchmarks/` |

**Protocol:**
1. Warmup: 10 timesteps (JIT compilation)
2. Measurement: 100+ vegetation timesteps
3. Grid sizes: 1k, 5k, 10k

**Exit:** Performance target met or gap quantified.

---

## Phase 7: Profiling & Iteration

Identify and address remaining bottlenecks.

| Task | File |
|------|------|
| Enable Taichi kernel profiler | `src/config.py` |
| Document per-kernel timing breakdown | `docs/ARCHITECTURE.md` |
| Profile atomic reduction overhead | `src/kernels/soil.py`, `src/kernels/vegetation.py` |
| Iterate on hotspots | as needed |

```python
ti.init(arch=ti.cuda, kernel_profiler=True)
# ... run simulation ...
ti.profiler.print_kernel_profiler_info()
```

**Known items to profile:**
- Atomic operations (`ti.atomic_add`) used for mass balance tracking in all kernels
- At 10k×10k, this is 100M atomic adds per kernel call — may serialize on GPU
- Consider hierarchical reduction if atomics > 5% of kernel time

**Exit:** Bottlenecks identified, performance documented.

---

## Exit Criteria (Complete)

- [x] Ping-pong buffers implemented (no copy overhead)
- [x] Point-wise kernel fusion (2× memory traffic reduction)
- [x] Generic diffusion kernel consolidated
- [x] Shared memory caching via `ti.block_local()`
- [x] Duplicate diffusion kernels eliminated
- [x] Benchmark harness created
- [x] All tests pass
- [ ] 10k×10k at ≥1 year/minute on H100
- [ ] >50% theoretical bandwidth achieved
- [ ] Performance documented in ARCHITECTURE.md

---

## Dependencies

- **Taichi:** `>=1.7.0`
- **Hardware:** H100 (sm_90) or B200 (sm_100) for GPU testing
- **CPU fallback:** Works on any machine for development
