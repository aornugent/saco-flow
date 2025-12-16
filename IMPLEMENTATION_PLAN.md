# Phase 6: GPU Optimization

**Goal:** Maximize throughput on B200 (sm_100) with H100 (sm_90) fallback.

**Target:** 10k×10k grid at ≥1 simulated year per wall-clock minute.

**Read:** `docs/gpu_optimization.md`, `ecohydro_spec.md:609-714`

---

## Overview

The simulation is memory-bandwidth bound. Every optimization targets reducing global memory traffic while maintaining numerical equivalence (within 1e-5 tolerance).

| Sub-phase | Focus |
|-----------|-------|
| 6.0 | Architecture refactoring |
| 6.1 | Memory access patterns |
| 6.2 | Kernel fusion — soil |
| 6.3 | Kernel fusion — vegetation |
| 6.4 | Kernel fusion — routing |
| 6.5 | Temporal blocking |
| 6.6 | 10k×10k validation |
| 6.7 | Benchmarking |
| 6.8 | Profiling |

---

## 6.0: Architecture Refactoring

Clean, modular foundation for optimization.

| Task | File |
|------|------|
| Typed field container | `src/fields.py` |
| Consolidated config | `src/config.py` |
| Kernel registry (naive/optimized) | `src/kernels/__init__.py` |

**Exit:** Tests pass, kernels swappable via config.

---

## 6.1: Memory Access Optimization

Ensure coalesced access before fusion.

| Task | File |
|------|------|
| Audit loop order (`j` innermost) | `src/kernels/*.py` |
| Add `ti.block_dim()` hints | `src/kernels/*.py` |
| Verify SoA layout | `src/fields.py` |

**Exit:** All kernels use coalesced access, tests pass.

---

## 6.2: Kernel Fusion — Soil Moisture

Fuse ET + leakage + diffusion into single pass (4 memory round-trips → 1).

| Task | File |
|------|------|
| Fused kernel with `ti.block_local(M)` | `src/kernels/soil_fused.py` |
| Regression tests | `tests/test_soil_fused.py` |

**Exit:** Results match naive within 1e-5, mass conserved.

---

## 6.3: Kernel Fusion — Vegetation

Fuse growth + mortality + diffusion.

| Task | File |
|------|------|
| Fused kernel | `src/kernels/vegetation_fused.py` |
| Regression tests | `tests/test_vegetation_fused.py` |

**Exit:** Results match naive within tolerance.

---

## 6.4: Kernel Fusion — Routing

Optimize surface water routing (most time-critical during rainfall).

| Task | File |
|------|------|
| Analyze two-pass necessity | `src/kernels/flow.py` |
| Inline CFL computation | `src/kernels/flow_fused.py` |
| Early drainage termination | `src/simulation.py` |

**Constraint:** Two-pass may be required for mass conservation.

**Exit:** Routing conserves mass, reduced kernel overhead.

---

## 6.5: Temporal Blocking

Batch multiple diffusion steps in shared memory.

| Task | File |
|------|------|
| Temporal blocking kernel | `src/kernels/diffusion_temporal.py` |
| Document stability limits | `docs/gpu_optimization.md` |

**Exit:** Document when beneficial vs overhead-dominated.

---

## 6.6: 10k×10k Validation

Verify execution at target scale.

| Task | File |
|------|------|
| Memory allocation test | `tests/test_large_grid.py` |
| End-to-end 10k×10k run | `benchmarks/` |

**Memory budget:** ~7.7 GB (<10% of H100's 80GB)

**Exit:** 10k×10k runs without errors.

---

## 6.7: Benchmarking

Systematic performance measurement.

| Task | File |
|------|------|
| Benchmark harness | `benchmarks/benchmark.py` |
| Throughput metrics | `docs/gpu_optimization.md` |

**Metrics:**
- Cells/second
- Simulated years/minute
- Achieved bandwidth (GB/s)

**Exit:** Performance target met or gap quantified.

---

## 6.8: Profiling

Identify remaining bottlenecks.

| Task | File |
|------|------|
| Taichi profiler integration | `src/config.py` |
| Kernel timing breakdown | `benchmarks/profile.py` |

**Exit:** Per-kernel timing documented, bottlenecks identified.

---

## Exit Criteria (Phase 6 Complete)

- [ ] All tests pass
- [ ] Optimized kernels match naive within tolerance
- [ ] 10k×10k at ≥1 year/minute on H100
- [ ] >50% theoretical bandwidth achieved
- [ ] Performance documented

---

## Dependencies

- **Taichi:** `>=1.7.0`
- **Hardware:** H100 (sm_90) or B200 (sm_100) for GPU testing
