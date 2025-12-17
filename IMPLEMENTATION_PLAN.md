# Phase 6: GPU Optimization

**Goal:** Maximize throughput on B200 (sm_100) with H100 (sm_90) fallback.

**Target:** 10k×10k grid at ≥1 simulated year per wall-clock minute.

**Read:** `docs/gpu_optimization.md`, `docs/ARCHITECTURE.md`, `ecohydro_spec.md:609-714`

---

## Overview

The simulation is memory-bandwidth bound. Every optimization targets reducing global memory traffic while maintaining numerical equivalence (within 1e-5 tolerance).

**Critical**: Complete architecture groundwork (6.0a-6.0e) before kernel fusion (6.2+).

| Sub-phase | Focus | Status |
|-----------|-------|--------|
| 6.0a | Core infrastructure (geometry, dtypes) | ✅ Complete |
| 6.0b | Field management (containers, double-buffering) | ✅ Complete |
| 6.0c | Parameter system (validation, injection) | ✅ Complete |
| 6.0d | Kernel protocols (interfaces, registry) | ✅ Complete |
| 6.0e | Runner refactoring (orchestration) | ⬜ Pending |
| 6.1 | Memory access patterns | ⬜ Pending |
| 6.2 | Kernel fusion — soil | ⬜ Pending |
| 6.3 | Kernel fusion — vegetation | ⬜ Pending |
| 6.4 | Kernel fusion — routing | ⬜ Pending |
| 6.5 | Temporal blocking | ⬜ Pending |
| 6.6 | 10k×10k validation | ⬜ Pending |
| 6.7 | Benchmarking | ⬜ Pending |
| 6.8 | Profiling | ⬜ Pending |

---

## 6.0a: Core Infrastructure ✅

Centralized geometry and type definitions.

| Task | File | Status |
|------|------|--------|
| Move DTYPE to dedicated module | `src/core/dtypes.py` | ✅ |
| GridGeometry dataclass (nx, ny, dx) | `src/core/geometry.py` | ✅ |
| Neighbor vectors (DI, DJ, DIST) | `src/core/geometry.py` | ✅ |
| Neighbor helper functions (@ti.func) | `src/core/geometry.py` | ✅ |
| Unit tests for geometry | `tests/test_geometry.py` | ✅ |

**Exit:** ✅ Geometry module complete with tests, existing code unchanged.

**Completed:** 2024-12-16

**Summary:**
- `src/core/dtypes.py`: DTYPE (ti.f32) definition
- `src/core/geometry.py`: GridGeometry frozen dataclass with validation, 8-connectivity neighbor vectors (NEIGHBOR_DI, NEIGHBOR_DJ, NEIGHBOR_DIST), helper functions (is_interior, get_neighbor, get_neighbor_distance)
- `tests/test_geometry.py`: 29 unit tests covering dataclass, vectors, and Taichi functions
- All 140 existing tests pass (no regressions)

---

## 6.0b: Field Management ✅

Typed field containers with declarative specs.

| Task | File | Status |
|------|------|--------|
| FieldSpec dataclass (name, dtype, shape, role) | `src/fields/base.py` | ✅ |
| FieldContainer class (register, allocate, swap) | `src/fields/base.py` | ✅ |
| FieldRole enum (STATE, STATIC, DERIVED, SCRATCH) | `src/fields/base.py` | ✅ |
| State field factory (h, m, p) | `src/fields/state.py` | ✅ |
| Static field factory (z, mask, flow_frac) | `src/fields/static.py` | ✅ |
| Scratch/derived field factory | `src/fields/scratch.py` | ✅ |
| Convenience wrappers (StateFields, StaticFields, ScratchFields) | `src/fields/*.py` | ✅ |
| Double-buffer swap tests | `tests/test_fields.py` | ✅ |

**Exit:** ✅ Field containers working, can coexist with SimpleNamespace.

**Completed:** 2025-12-16

**Summary:**
- `src/fields/base.py`: FieldSpec frozen dataclass with validation (name, dtype, role, double_buffer, extra_dims), FieldRole enum, FieldContainer class with register/allocate/swap/get operations, memory tracking
- `src/fields/state.py`: StateFields wrapper for h/m/p with swap operations, create_state_specs factory
- `src/fields/static.py`: StaticFields wrapper with tilted plane and DEM initialization, create_static_specs factory
- `src/fields/scratch.py`: ScratchFields wrapper for temporary/derived fields, create_scratch_specs and create_derived_specs factories
- `tests/test_fields.py`: 58 unit tests covering specs, containers, double-buffering, swap operations, convenience wrappers, factory functions, and memory tracking
- All 227 tests pass (no regressions from 169 existing tests + 58 new tests)

---

## 6.0c: Parameter System ✅

Validated, immutable parameter containers.

| Task | File | Status |
|------|------|--------|
| Nested param dataclasses with validation | `src/params/schema.py` | ✅ |
| YAML loader with from_yaml() | `src/params/loader.py` | ✅ |
| TaichiParams injection class | `src/params/taichi_params.py` | ✅ |
| Validation tests | `tests/test_params.py` | ✅ |

**Exit:** ✅ Parameters loadable from YAML, validation catches errors at init.

**Completed:** 2025-12-16

**Summary:**
- `src/params/schema.py`: 8 frozen dataclasses with `__post_init__` validation (GridParams, RainfallParams, InfiltrationParams, SoilParams, VegetationParams, RoutingParams, DrainageParams, TimestepParams), SimulationConfig aggregator with `to_dict`/`from_dict`/`with_updates` methods, convenience property accessors for backward compatibility
- `src/params/loader.py`: YAML loading (`load_config`), saving (`save_config`), and merge utilities (`load_config_with_overrides`, `merge_configs`)
- `src/params/taichi_params.py`: TaichiParams class bridging Python dataclasses to Taichi scalar fields for kernel access without recompilation
- `tests/test_params.py`: 51 unit tests covering validation, YAML roundtrip, Taichi injection, and kernel readability
- All 278 tests pass (no regressions from 227 existing tests + 51 new tests)

---

## 6.0d: Kernel Protocols ✅

Abstract interfaces for swappable implementations.

| Task | File | Status |
|------|------|--------|
| Protocol definitions (SoilKernel, VegetationKernel, FlowKernel, InfiltrationKernel) | `src/kernels/protocol.py` | ✅ |
| Move existing kernels to naive/ | `src/kernels/naive/*.py` | ✅ |
| Wrap naive kernels in protocol-compliant classes | `src/kernels/naive/*.py` | ✅ |
| KernelRegistry with variant selection | `src/kernels/__init__.py` | ✅ |
| Registry tests | `tests/test_kernel_registry.py` | ✅ |

**Exit:** ✅ Existing kernels accessible via registry, all tests pass.

**Completed:** 2025-12-17

**Summary:**
- `src/kernels/protocol.py`: Protocol definitions for SoilKernel, VegetationKernel, InfiltrationKernel, FlowKernel, FlowDirectionKernel; KernelVariant enum (NAIVE, FUSED, TEMPORAL); result dataclasses (SoilFluxes, VegetationFluxes, InfiltrationFluxes, RoutingFluxes)
- `src/kernels/naive/soil.py`: Soil moisture kernels (ET, leakage, diffusion) with NaiveSoilKernel wrapper
- `src/kernels/naive/vegetation.py`: Vegetation kernels (growth, mortality, dispersal) with NaiveVegetationKernel wrapper
- `src/kernels/naive/infiltration.py`: Infiltration kernel with NaiveInfiltrationKernel wrapper
- `src/kernels/naive/flow.py`: Flow direction and routing kernels with NaiveFlowKernel, NaiveFlowDirectionKernel wrappers
- `src/kernels/__init__.py`: KernelRegistry class with get/register methods for each kernel type, variant selection, global registry accessor
- Original kernel files (soil.py, vegetation.py, etc.) now re-export from naive/ for backwards compatibility
- `tests/test_kernel_registry.py`: 37 unit tests covering registry basics, error handling, registration, protocol compliance, and integration
- All 315 tests pass (no regressions from 278 existing tests + 37 new tests)

---

## 6.0e: Runner Refactoring

Clean orchestration using new components.

| Task | File |
|------|------|
| SimulationRunner using FieldContainer + Registry | `src/simulation/runner.py` |
| Callback hooks (on_day, on_year) | `src/simulation/runner.py` |
| MassTracker diagnostics class | `src/diagnostics/conservation.py` |
| Integration tests with new runner | `tests/test_runner.py` |

**Exit:** All existing tests pass with refactored runner.

---

## 6.1: Memory Access Optimization

Ensure coalesced access before fusion.

| Task | File |
|------|------|
| Audit loop order (`j` innermost) | `src/kernels/naive/*.py` |
| Add `ti.block_dim()` hints | `src/kernels/naive/*.py` |
| Verify SoA layout in FieldContainer | `src/fields/base.py` |

**Exit:** All kernels use coalesced access, tests pass.

---

## 6.2: Kernel Fusion — Soil Moisture

Fuse ET + leakage + diffusion into single pass (4 memory round-trips → 1).

| Task | File |
|------|------|
| Fused kernel with `ti.block_local(M)` | `src/kernels/optimized/soil_fused.py` |
| Register fused variant in registry | `src/kernels/__init__.py` |
| Equivalence tests (naive vs fused) | `tests/test_kernel_equivalence.py` |

**Exit:** Results match naive within 1e-5, mass conserved.

---

## 6.3: Kernel Fusion — Vegetation

Fuse growth + mortality + diffusion.

| Task | File |
|------|------|
| Fused kernel | `src/kernels/optimized/vegetation_fused.py` |
| Register fused variant | `src/kernels/__init__.py` |
| Equivalence tests | `tests/test_kernel_equivalence.py` |

**Exit:** Results match naive within tolerance.

---

## 6.4: Kernel Fusion — Routing

Optimize surface water routing (most time-critical during rainfall).

| Task | File |
|------|------|
| Analyze two-pass necessity | `src/kernels/naive/flow.py` |
| Inline CFL computation | `src/kernels/optimized/flow_fused.py` |
| Early drainage termination | `src/simulation/runner.py` |

**Constraint:** Two-pass may be required for mass conservation.

**Exit:** Routing conserves mass, reduced kernel overhead.

---

## 6.5: Temporal Blocking

Batch multiple diffusion steps in shared memory.

| Task | File |
|------|------|
| Temporal blocking kernel | `src/kernels/optimized/diffusion_temporal.py` |
| Register TEMPORAL variant | `src/kernels/__init__.py` |
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
| Taichi profiler integration | `src/core/dtypes.py` |
| Profiling hooks in runner | `src/diagnostics/profiling.py` |
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
