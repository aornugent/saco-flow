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

- Long simulation (decades) on tilted plane
- Visual inspection for pattern emergence
- FFT analysis of vegetation field (characteristic wavelength)
- Parameter sensitivity: vary infiltration feedback, rainfall frequency

### Tests

- `test_long_run_stability` — no numerical drift over decades
- `test_pattern_emerges` — vegetation not uniform after spinup
- `test_pattern_wavelength_reasonable` — matches expected spatial scale

### Exit Criteria

- Patterns emerge from uniform initial conditions
- Spatial wavelength in physically reasonable range
- System stable over long simulations

---

## Phase 6: Optimization

**Goal:** Target performance on H100/B200.

**Read:** `ecohydro_spec.md:609-714` (Performance Optimization)

### Tasks

- Profile with Taichi profiler
- Kernel fusion for soil update
- Verify coalesced memory access
- Benchmark at 1k, 5k, 10k grids

### Exit Criteria

- >50% theoretical memory bandwidth
- 10k×10k at ~1 simulated year/minute

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

**Completed:** Phase 0 (Project Setup), Phase 1 (Surface Water Routing), Phase 2 (Infiltration & Soil Moisture)

**Active Phase:** 3 (Vegetation Dynamics)

**Next Milestone:** Monod growth, mortality, and seed dispersal
