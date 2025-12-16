# Implementation Plan

Phased development from surface water routing to end-to-end simulation.

## Philosophy

1. **Start simple** — Synthetic terrain, small grids, basic physics
2. **Test continuously** — Every kernel gets a conservation test
3. **Defer optimization** — Correctness first, performance in Phase 6

**Mass conservation** is verified throughout all phases (see `docs/mass_conservation.md`), not just at the end.

## Phase Overview

| Phase | Focus | Key Docs |
|-------|-------|----------|
| 0 | Project Setup | — |
| 1 | Surface Water Routing | `docs/kernels/flow_*.md`, `docs/kernels/surface_routing.md` |
| 2 | Infiltration & Soil | `docs/kernels/infiltration.md`, `docs/kernels/soil_moisture.md` |
| 3 | Vegetation | `docs/kernels/vegetation.md` |
| 4 | Integration | `docs/timesteps.md`, `docs/boundaries.md` |
| 5 | Validation | `docs/overview.md`, `ecohydro_spec.md:786-803` |
| 6 | Optimization | `ecohydro_spec.md:609-714` |

---

## Phase 0: Project Setup

**Goal:** Development infrastructure.

### Tasks

- [x] `requirements.txt` — taichi, numpy, rasterio, matplotlib, pytest, ruff
- [x] Package structure — `src/`, `src/kernels/`, `tests/`
- [x] Taichi init — GPU detection, CPU fallback, f32 default (`src/config.py`)
- [x] Test fixtures — small grid factory, synthetic terrain, conservation assertions (`tests/conftest.py`)
- [x] Utility kernels — field operations, neighbor indexing (`src/kernels/utils.py`)
- [x] CUDA verification — sm90/sm100 compatibility documented, CPU fallback tested

### Exit Criteria

- [x] `pytest tests/` passes (22 tests)
- [x] Taichi kernels execute on CPU (GPU optional)

---

## Phase 1: Surface Water Routing

**Goal:** MFD flow directions and kinematic wave routing on synthetic terrain.

**Read:** `docs/kernels/flow_directions.md`, `docs/kernels/flow_accumulation.md`, `docs/kernels/surface_routing.md`

### Tasks

| Task | File | Status |
|------|------|--------|
| Synthetic terrain (tilted plane, valley, hill) | `tests/conftest.py` | ✓ |
| MFD flow direction computation | `src/kernels/flow.py` | ✓ |
| Iterative flow accumulation | `src/kernels/flow.py` | ✓ |
| Kinematic wave routing with CFL | `src/kernels/flow.py` | ✓ |
| Outlet boundary conditions | `src/kernels/flow.py` | ✓ |

### Tests (13 passing)

- [x] `test_flow_directions_tilted_plane` — all flow downslope
- [x] `test_flow_directions_symmetric` — symmetric terrain → symmetric flow
- [x] `test_flat_terrain_flagged` — flat cells marked
- [x] `test_flow_fractions_sum_to_one` — MFD fractions normalized
- [x] `test_flow_accumulation_conservation` — total in = total out
- [x] `test_accumulation_increases_downslope` — accumulation grows
- [x] `test_routing_mass_conservation` — mass balance verified
- [x] `test_water_flows_downslope` — center of mass moves south
- [x] `test_no_flow_on_flat_terrain` — flat terrain stable
- [x] `test_cfl_timestep_finite` — CFL returns valid dt
- [x] `test_cfl_timestep_infinite_no_water` — no water → infinite dt
- [x] `test_stability_with_cfl_timestep` — no NaN or negative values

### Exit Criteria

- [x] Flow directions correct on all synthetic terrains
- [x] Routing conserves mass to floating point tolerance

---

## Phase 2: Infiltration & Soil Moisture

**Goal:** Water transfer from surface to soil, ET, leakage, diffusion.

**Read:** `docs/kernels/infiltration.md`, `docs/kernels/soil_moisture.md`

### Tasks

| Task | File | Status |
|------|------|--------|
| Vegetation-enhanced infiltration | `src/kernels/infiltration.py` | ✓ |
| Evapotranspiration | `src/kernels/soil.py` | ✓ |
| Deep leakage | `src/kernels/soil.py` | ✓ |
| Soil moisture diffusion (5-point Laplacian) | `src/kernels/soil.py` | ✓ |

### Tests (22 passing)

- [x] `test_conservation_h_to_M` — Δh = -ΔM
- [x] `test_no_infiltration_when_saturated` — no infiltration when M = M_sat
- [x] `test_no_infiltration_when_dry_surface` — no infiltration when h = 0
- [x] `test_vegetation_enhances_infiltration` — more veg → more infiltration
- [x] `test_infiltration_limited_by_available_water` — can't exceed h
- [x] `test_infiltration_limited_by_capacity` — can't exceed M_sat
- [x] `test_et_reduces_moisture` — ET decreases M
- [x] `test_vegetation_enhances_et` — more veg → more ET
- [x] `test_leakage_reduces_moisture` — leakage decreases M
- [x] `test_leakage_quadratic` — wet soil leaks more
- [x] `test_diffusion_conserves_mass` — total M unchanged
- [x] `test_diffusion_smooths_gradient` — variance decreases
- [x] `test_diffusion_timestep_stability` — stable at computed dt

### Exit Criteria

- [x] Infiltration conserves mass with vegetation feedback
- [x] Diffusion stable and conserving

---

## Phase 3: Vegetation Dynamics

**Goal:** Growth, mortality, seed dispersal.

**Read:** `docs/kernels/vegetation.md`

### Tasks

| Task | File |
|------|------|
| Monod growth kinetics | `src/kernels/vegetation.py` |
| Constant mortality | `src/kernels/vegetation.py` |
| Seed dispersal (diffusion) | `src/kernels/vegetation.py` |

### Tests

- `test_growth_increases_biomass` — positive moisture → growth
- `test_mortality_decreases_biomass` — decay without moisture
- `test_vegetation_positivity` — P ≥ 0 always
- `test_equilibrium_biomass` — growth = mortality at steady state

### Exit Criteria

- Vegetation responds correctly to moisture
- System reaches equilibrium with constant forcing

---

## Phase 4: Integration

**Goal:** End-to-end simulation loop.

**Read:** `docs/timesteps.md`, `docs/boundaries.md`

### Tasks

| Task | File |
|------|------|
| Rainfall event handling | `src/simulation.py` |
| Hierarchical time stepping | `src/simulation.py` |
| Main simulation loop | `src/simulation.py` |
| GeoTIFF output + thumbnails | `src/output.py` |
| Initialization (synthetic terrain + random P) | `src/simulation.py` |

### Tests

- `test_simulation_runs_without_crash` — smoke test
- `test_simulation_mass_conservation` — track all fluxes
- `test_output_files_created` — GeoTIFFs written

### Exit Criteria

- Multi-year simulation completes
- Mass balance verified across timesteps
- Output files readable

---

## Phase 5: Validation

**Goal:** Verify pattern formation and physical plausibility.

**Read:** `docs/overview.md` (feedback mechanism), `ecohydro_spec.md:786-803` (spinup protocol)

### Tasks

| Task | File | Status |
|------|------|--------|
| Long-run numerical stability | `tests/test_validation.py` | ✓ |
| Pattern emergence verification | `tests/test_validation.py` | ✓ |
| FFT analysis (characteristic wavelength) | `tests/test_validation.py` | ✓ |
| Parameter sensitivity tests | `tests/test_validation.py` | ✓ |
| Turing instability mechanism validation | `tests/test_validation.py` | ✓ |
| Slope effects on water redistribution | `tests/test_validation.py` | ✓ |
| Equilibrium state tests | `tests/test_validation.py` | ✓ |

### Tests (20 passing)

**Long-Run Stability:**
- [x] `test_no_nan_or_inf_after_one_year` — no numerical failures
- [x] `test_fields_within_physical_bounds_after_one_year` — h, M, P bounded
- [x] `test_mass_conservation_over_multiple_years` — mass balance verified
- [x] `test_no_numerical_drift_five_years` — no unbounded growth/decay

**Pattern Emergence:**
- [x] `test_vegetation_heterogeneity_increases` — variance grows from uniform start
- [x] `test_pattern_not_uniform_after_spinup` — CV > 10% after spinup
- [x] `test_spatial_structure_emerges` — positive autocorrelation (coherent patterns)

**Pattern Wavelength:**
- [x] `test_fft_detects_dominant_scale` — FFT shows peak at non-zero frequency
- [x] `test_wavelength_physically_reasonable` — wavelength in 1-200m range

**Turing Mechanism:**
- [x] `test_vegetation_captures_water_locally` — positive local feedback verified
- [x] `test_vegetation_reduces_runoff_to_neighbors` — negative nonlocal feedback
- [x] `test_instability_amplifies_perturbations` — perturbations grow >5x

**Parameter Sensitivity:**
- [x] `test_higher_rainfall_increases_vegetation` — more rain → more veg
- [x] `test_higher_mortality_decreases_vegetation` — more death → less veg
- [x] `test_bare_soil_factor_affects_pattern_contrast` — W_0 affects patterns

**Slope Effects:**
- [x] `test_water_accumulates_downslope` — downslope wetter
- [x] `test_steeper_slope_faster_drainage` — steep slope → smaller CFL dt

**Equilibrium States:**
- [x] `test_system_approaches_steady_state` — changes decrease over time
- [x] `test_vegetation_persists_with_rainfall` — no extinction
- [x] `test_growth_mortality_balance_reached` — net change < 50%/year

### Exit Criteria

- [x] Patterns emerge from uniform initial conditions
- [x] Spatial wavelength in physically reasonable range
- [x] System stable over long simulations

---

## Phase 6: GPU Optimization

**Goal:** Maximize throughput on B200 (sm_100), with H100 (sm_90) as fallback. Target: 10k×10k grid at ≥1 simulated year per wall-clock minute.

**Read:** `ecohydro_spec.md:609-714` (Performance Optimization), `docs/gpu_optimization.md` (new)

**Design Philosophy:** The simulation is memory-bandwidth bound. Every optimization targets reducing global memory traffic while maintaining numerical equivalence with Phase 5 results (within tolerance). We use pure Taichi with advanced features; custom CUDA kernels are a future extension.

---

### Phase 6.0: Architecture Refactoring

**Goal:** Clean, modular foundation for optimization work.

Before optimizing, refactor for clarity and maintainability. This enables isolated benchmarking and simplifies kernel fusion.

| Task | File | Description |
|------|------|-------------|
| Unified field container | `src/fields.py` | Replace `SimpleNamespace` with typed `FieldSet` class |
| Parameter dataclass consolidation | `src/config.py` | Single `SimulationConfig` with nested groups |
| Kernel registry pattern | `src/kernels/__init__.py` | Enable swapping naive/optimized kernel implementations |
| Pin Taichi version | `requirements.txt` | Lock to `taichi>=1.7.0,<1.8.0` for `ti.block_local` support |

**Exit Criteria:**
- [ ] All 140 tests pass with refactored code
- [ ] Kernels can be selected via config (naive vs optimized)
- [ ] Clean separation between field allocation and kernel logic

---

### Phase 6.1: Memory Access Optimization

**Goal:** Ensure optimal memory access patterns before fusion.

Modern GPUs (H100/B200) achieve peak bandwidth only with coalesced, aligned access. Verify and fix access patterns.

| Task | File | Description |
|------|------|-------------|
| Audit loop order in all kernels | `src/kernels/*.py` | Ensure `j` (column) is innermost for row-major layout |
| Add `ti.block_dim()` hints | `src/kernels/*.py` | Explicit thread block sizing (256 or 512 threads) |
| Verify SoA layout | `src/fields.py` | Confirm no AoS patterns crept in |
| Eliminate redundant mask checks | `src/kernels/*.py` | Restructure loops to skip boundary cells efficiently |

**Memory Access Pattern:**
```python
# CORRECT: j varies fastest (coalesced)
for i, j in ti.ndrange((1, n-1), (1, n-1)):

# INCORRECT: i varies fastest (strided)
for j, i in ti.ndrange((1, n-1), (1, n-1)):
```

**Exit Criteria:**
- [ ] All kernels use column-major innermost iteration
- [ ] Block dimensions specified for compute-heavy kernels
- [ ] Tests still pass (numerical equivalence)

---

### Phase 6.2: Kernel Fusion — Soil Moisture

**Goal:** Fuse ET + leakage + diffusion into single kernel pass.

Currently `soil_moisture_step` calls three separate kernels plus a `copy_field`. This requires 4 global memory round-trips. Fusion reduces to 1.

| Task | File | Description |
|------|------|-------------|
| Fused soil kernel | `src/kernels/soil_fused.py` | Single kernel: read M,P → compute ET,leakage,diffusion → write M_new |
| Shared memory caching | `src/kernels/soil_fused.py` | Use `ti.block_local(M)` for diffusion stencil |
| Mass balance tracking | `src/kernels/soil_fused.py` | Atomic accumulation of ET/leakage totals |
| Regression tests | `tests/test_soil_fused.py` | Verify equivalence with naive implementation |

**Fused Kernel Structure:**
```python
@ti.kernel
def soil_update_fused(M: ti.template(), M_new: ti.template(), P: ti.template(),
                      mask: ti.template(), params: ti.template(), dt: DTYPE) -> ti.types.vector(2, DTYPE):
    """Fused: ET + leakage + diffusion. Returns (total_et, total_leakage)."""
    ti.block_local(M)  # Cache M in shared memory for Laplacian

    totals = ti.Vector([0.0, 0.0])
    for i, j in ti.ndrange((1, n-1), (1, n-1)):
        if mask[i, j] == 0:
            continue
        # Load once
        M_c, P_c = M[i, j], P[i, j]
        # ET
        et = compute_et(M_c, P_c, params, dt)
        # Leakage
        leak = compute_leakage(M_c, params, dt)
        # Diffusion Laplacian (reads from block_local cache)
        laplacian = M[i-1,j] + M[i+1,j] + M[i,j-1] + M[i,j+1] - 4*M_c
        diff = params.D_M * laplacian / (params.dx * params.dx) * dt
        # Single write
        M_new[i, j] = clamp(M_c - et - leak + diff, 0, params.M_sat)
        ti.atomic_add(totals[0], et)
        ti.atomic_add(totals[1], leak)
    return totals
```

**Expected Speedup:** 2-3× for soil update (reduced memory traffic)

**Exit Criteria:**
- [ ] Fused kernel produces results within 1e-5 of naive implementation
- [ ] Mass conservation maintained
- [ ] Benchmark shows measurable speedup on GPU

---

### Phase 6.3: Kernel Fusion — Vegetation

**Goal:** Fuse growth + mortality + diffusion into single kernel pass.

Same pattern as soil moisture, but simpler (no cross-field dependencies within timestep).

| Task | File | Description |
|------|------|-------------|
| Fused vegetation kernel | `src/kernels/vegetation_fused.py` | Single kernel for all vegetation dynamics |
| Shared memory for P | `src/kernels/vegetation_fused.py` | Cache P for diffusion stencil |
| Regression tests | `tests/test_vegetation_fused.py` | Verify equivalence |

**Exit Criteria:**
- [ ] Fused kernel matches naive to tolerance
- [ ] Benchmark shows measurable speedup

---

### Phase 6.4: Kernel Fusion — Surface Routing

**Goal:** Optimize the CFL-limited surface water routing loop.

Surface routing is the most time-critical component during rainfall events. Currently uses two-pass scheme (compute_outflow → apply_fluxes). This is correct but requires careful optimization.

| Task | File | Description |
|------|------|-------------|
| Fused outflow+apply kernel | `src/kernels/flow_fused.py` | Single pass where safe (no race conditions) |
| CFL computation optimization | `src/kernels/flow_fused.py` | Compute max velocity inline during routing |
| Adaptive event detection | `src/simulation.py` | Skip routing entirely when h < threshold everywhere |
| Early termination | `src/simulation.py` | Exit subcycle loop faster when drained |

**Constraint:** Two-pass may still be necessary for mass conservation. Analyze carefully before fusing.

**Exit Criteria:**
- [ ] Routing still conserves mass
- [ ] Reduced kernel launch overhead
- [ ] Faster drainage detection

---

### Phase 6.5: Temporal Blocking for Diffusion

**Goal:** Batch multiple diffusion sub-steps in shared memory.

For pure diffusion (stable, no cross-field dependencies), we can apply N steps in shared memory before writing to global. Reduces bandwidth by factor of ~N.

| Task | File | Description |
|------|------|-------------|
| Temporal blocking kernel | `src/kernels/diffusion_temporal.py` | Multi-step diffusion in shared memory |
| Stability analysis | `docs/gpu_optimization.md` | Document step count limits for stability |
| Integration with fused kernels | `src/kernels/soil_fused.py` | Apply temporal blocking within fused kernel |

**Temporal Blocking Structure:**
```python
@ti.kernel
def diffusion_temporal_block(M: ti.template(), D: DTYPE, dx: DTYPE, dt: DTYPE, steps: int):
    """Apply `steps` diffusion iterations in shared memory."""
    ti.block_local(M)

    for i, j in ti.ndrange((BLOCK, N-BLOCK), (BLOCK, N-BLOCK)):
        local = M[i, j]
        for _ in range(steps):
            laplacian = # from shared memory neighbors
            local += D * laplacian / (dx*dx) * dt
            ti.simt.block.sync()  # Synchronize block
        M[i, j] = local
```

**Limitation:** Only applicable when diffusion is the sole operation. May not combine well with fused ET/leakage.

**Exit Criteria:**
- [ ] Temporal blocking works for isolated diffusion
- [ ] Document when it's beneficial vs overhead-dominated

---

### Phase 6.6: Large Grid Support

**Goal:** Efficient execution at 10k×10k and documentation for larger scales.

| Task | File | Description |
|------|------|-------------|
| Memory layout for 10k×10k | `src/fields.py` | Verify allocation succeeds on 80GB HBM |
| Tiled execution | `src/kernels/tiled.py` | Optional tiling for grids exceeding GPU memory |
| Scaling documentation | `docs/gpu_optimization.md` | Performance expectations vs grid size |

**Memory Budget at 10k×10k (10⁸ cells):**
| Fields | Bytes/cell | Total |
|--------|------------|-------|
| Primary (h, M, P) | 12 | 1.2 GB |
| Buffers (h_new, M_new, P_new) | 12 | 1.2 GB |
| Flow fractions | 32 | 3.2 GB |
| Static (Z, mask) | 5 | 0.5 GB |
| Routing (q_out, flow_acc, etc.) | 16 | 1.6 GB |
| **Total** | | **~7.7 GB** |

This is <10% of H100's 80GB, <5% of B200's 192GB. Plenty of headroom.

**Scaling to Larger Grids:**
- 20k×20k (4×10⁸ cells): ~31 GB — fits H100
- 50k×50k (2.5×10⁹ cells): ~193 GB — requires B200 or tiling
- For domains exceeding GPU memory: implement overlapping tile execution with halo exchange

**Exit Criteria:**
- [ ] 10k×10k runs without memory errors
- [ ] Document memory usage per grid size
- [ ] (Optional) Prototype tiled execution for future

---

### Phase 6.7: Benchmarking Infrastructure

**Goal:** Measure and track performance systematically.

| Task | File | Description |
|------|------|-------------|
| Benchmark harness | `benchmarks/benchmark.py` | Standardized timing for kernel and full simulation |
| Grid size sweep | `benchmarks/benchmark.py` | Run at 1k, 5k, 10k grids |
| Throughput metrics | `benchmarks/benchmark.py` | Cells/second, years/minute, GB/s achieved |
| Comparison table | `docs/gpu_optimization.md` | Naive vs optimized performance |

**Target Metrics:**
- **Throughput:** 10k×10k at ≥1 simulated year per wall-clock minute
- **Bandwidth efficiency:** >50% of theoretical peak (H100: ~1.7 TB/s, B200: ~4 TB/s)
- **Kernel overhead:** <5% of time in kernel launch/sync

**Exit Criteria:**
- [ ] Benchmark script runs end-to-end
- [ ] Results documented in `docs/gpu_optimization.md`
- [ ] Performance target met or gap quantified

---

### Phase 6.8: Profiling and Analysis

**Goal:** Identify remaining bottlenecks using Taichi profiler.

| Task | File | Description |
|------|------|-------------|
| Enable Taichi profiler | `src/config.py` | Add `ti.init(kernel_profiler=True)` option |
| Profile report generation | `benchmarks/profile.py` | Generate per-kernel timing breakdown |
| Hotspot analysis | `docs/gpu_optimization.md` | Document findings and remaining opportunities |
| Occupancy analysis | `docs/gpu_optimization.md` | Check register pressure, shared memory usage |

**Profiling Protocol:**
1. Run 10k×10k simulation for 1 simulated year
2. Collect kernel timing breakdown
3. Identify top 3 kernels by time
4. Analyze memory vs compute bound nature
5. Document findings

**Exit Criteria:**
- [ ] Profiler integration working
- [ ] Per-kernel timing documented
- [ ] Bottlenecks identified for future work

---

### Phase 6 Summary

| Sub-phase | Focus | Key Deliverable |
|-----------|-------|-----------------|
| 6.0 | Refactoring | Clean field/config architecture |
| 6.1 | Memory access | Coalesced, aligned access patterns |
| 6.2 | Soil fusion | Single-pass soil moisture update |
| 6.3 | Vegetation fusion | Single-pass vegetation update |
| 6.4 | Routing optimization | Faster surface water routing |
| 6.5 | Temporal blocking | Multi-step diffusion in shared memory |
| 6.6 | Large grids | 10k×10k support, scaling docs |
| 6.7 | Benchmarking | Performance measurement infrastructure |
| 6.8 | Profiling | Taichi profiler integration, analysis |

### Exit Criteria (Phase 6 Complete)

- [ ] All 140+ tests pass
- [ ] Optimized kernels match naive results within tolerance
- [ ] 10k×10k grid runs at ≥1 simulated year/minute on H100
- [ ] >50% theoretical memory bandwidth achieved
- [ ] Performance documented with benchmark results
- [ ] Profiler analysis identifies remaining opportunities

### Dependencies

- **Taichi:** `>=1.7.0,<1.8.0` (for `ti.block_local`, stable GPU backend)
- **Hardware:** H100 (sm_90) or B200 (sm_100) for GPU testing
- **CPU fallback:** Maintained for development but not optimized

---

## Future Work (Deferred)

Per `ecohydro_spec.md` Section 14:
- Multiple soil layers, groundwater
- Multiple vegetation types, fire, grazing
- Real DEM support, erosion/deposition
- Adaptive mesh, implicit diffusion
- Ensemble simulations, data assimilation

---

## Current Status

**Completed:** Phase 0 (Project Setup), Phase 1 (Surface Water Routing), Phase 2 (Infiltration & Soil Moisture), Phase 3 (Vegetation Dynamics), Phase 4 (Integration), Phase 5 (Validation)

**Active Phase:** Phase 6 (GPU Optimization)

**Test Summary:** 140 tests passing
- Setup/Fixtures: 22 tests
- Flow/Routing: 13 tests
- Infiltration: 7 tests
- Soil Moisture: 13 tests
- Vegetation: 15 tests
- Kernel Dynamics: 28 tests
- Simulation: 22 tests
- Validation: 20 tests

**Next Milestone:** Phase 6.0 (Architecture Refactoring)
