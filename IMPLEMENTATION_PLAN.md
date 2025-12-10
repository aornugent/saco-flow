# EcoHydro Implementation Plan

This document outlines the phased development approach for the EcoHydro simulation, starting with the minimal core (surface water routing) and building toward a complete end-to-end system.

## Development Philosophy

1. **Start simple:** Synthetic terrain, basic physics, small grids
2. **Test continuously:** Every component gets conservation tests
3. **Build incrementally:** Each phase produces working, tested code
4. **Defer optimization:** Get it correct first, fast second

## Phase Overview

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| 0 | Project Setup | Build system, dependencies, CI |
| 1 | Surface Water Routing | MFD flow directions, kinematic wave routing |
| 2 | Infiltration & Soil Moisture | Water transfer, ET, diffusion |
| 3 | Vegetation Dynamics | Growth, mortality, seed dispersal |
| 4 | Integration | End-to-end simulation loop |
| 5 | Validation | Pattern formation, mass balance verification |
| 6 | Optimization | Performance tuning for target hardware |

---

## Phase 0: Project Setup

**Goal:** Establish development infrastructure.

### Tasks

- [ ] **0.1** Create `requirements.txt` with pinned dependencies
  - taichi>=1.7.0
  - numpy>=1.24.0
  - rasterio>=1.3.0
  - matplotlib>=3.7.0
  - pytest>=7.0.0
  - pytest-cov>=4.0.0

- [ ] **0.2** Create package structure
  ```
  src/
    __init__.py
    fields.py
    kernels/__init__.py
  tests/
    __init__.py
    conftest.py
  ```

- [ ] **0.3** Configure Taichi initialization
  - Detect available GPU architecture
  - Fall back to CPU for testing
  - Set default precision (f32)

- [ ] **0.4** Create test fixtures in `conftest.py`
  - Small grid factory (100×100 for fast tests)
  - Synthetic terrain generators
  - Common assertions (mass conservation)

- [ ] **0.5** Verify CUDA compatibility
  - Test on available hardware
  - Document sm90/sm100 requirements

### Tests

- `test_taichi_init.py`: Verify Taichi initializes correctly
- `test_field_creation.py`: Verify fields allocate on GPU

### Exit Criteria

- [x] `pytest tests/` passes
- [x] Taichi kernels execute on GPU
- [x] Project structure matches README

---

## Phase 1: Surface Water Routing

**Goal:** Implement MFD flow direction and kinematic wave routing on synthetic terrain.

This is the computational core—getting flow routing correct and efficient is critical.

### Tasks

#### 1.1 Synthetic Terrain Generation

- [ ] **1.1.1** Tilted plane: `Z[i,j] = slope * i * dx`
- [ ] **1.1.2** V-shaped valley: `Z[i,j] = abs(j - N/2) * cross_slope + i * down_slope`
- [ ] **1.1.3** Gaussian hill: `Z[i,j] = height * exp(-((i-ci)² + (j-cj)²) / (2*sigma²))`
- [ ] **1.1.4** Domain mask handling (rectangular for now)

**File:** `src/terrain.py`

#### 1.2 Flow Direction Computation

- [ ] **1.2.1** Implement neighbor indexing convention (8-connected, clockwise from E)
- [ ] **1.2.2** Compute slope to each neighbor: `S_k = (Z_center - Z_neighbor) / d_k`
- [ ] **1.2.3** Compute MFD fractions: `f_k = max(0, S_k)^p / sum(max(0, S_m)^p)`
- [ ] **1.2.4** Handle flat cells / local minima (flag for special handling)
- [ ] **1.2.5** Store in `flow_frac[N, N, 8]` field

**File:** `src/kernels/flow.py`

**Test cases:**
- Tilted plane: all flow in one direction
- Symmetric valley: flow splits equally at ridge
- Flat terrain: no flow (fractions = 0 or flagged)

#### 1.3 Flow Accumulation

- [ ] **1.3.1** Implement iterative parallel accumulation
  ```
  A_new[i,j] = local[i,j] + sum(f_{k→(i,j)} * A[k] for k in donors)
  ```
- [ ] **1.3.2** Convergence detection: `max|A_new - A| < epsilon`
- [ ] **1.3.3** Double buffering for parallel update
- [ ] **1.3.4** Fixed iteration count option (for predictable performance)

**File:** `src/kernels/flow.py`

**Test cases:**
- Total accumulation at outlet equals total input
- Tilted plane: linear increase downslope
- Valley: accumulation concentrates in channel

#### 1.4 Surface Water Routing

- [ ] **1.4.1** Compute outflow rate: `q = min(h/dt, v * h / dx)` where `v = h^(2/3) * sqrt(S) / n`
- [ ] **1.4.2** Distribute outflow via MFD fractions
- [ ] **1.4.3** Accumulate inflow from donor cells
- [ ] **1.4.4** Update `h` field with positivity constraint
- [ ] **1.4.5** CFL timestep calculation

**File:** `src/kernels/flow.py`

**Test cases:**
- Mass conservation: total h unchanged (closed domain)
- Outlet boundary: water exits, mass decreases correctly
- Steady state: constant input reaches equilibrium profile

#### 1.5 Boundary Conditions

- [ ] **1.5.1** No-flux boundaries (default for closed domain tests)
- [ ] **1.5.2** Outlet boundary: remove water flowing off-domain
- [ ] **1.5.3** Track cumulative outflow for mass balance

**File:** `src/kernels/flow.py`

### Phase 1 Tests

| Test | Purpose |
|------|---------|
| `test_flow_directions_tilted_plane` | All flow goes downslope |
| `test_flow_directions_symmetric` | Symmetric terrain → symmetric flow |
| `test_flow_accumulation_conservation` | Total in = total accumulated at outlets |
| `test_routing_mass_conservation` | Water not created or destroyed |
| `test_routing_reaches_steady_state` | Constant input → stable output |
| `test_cfl_timestep_stability` | No blow-up with computed dt |

### Exit Criteria

- [ ] Flow directions computed correctly on all synthetic terrains
- [ ] Flow accumulation converges and conserves mass
- [ ] Surface routing conserves mass to within floating point tolerance
- [ ] CFL timestep keeps simulation stable

---

## Phase 2: Infiltration & Soil Moisture

**Goal:** Implement water transfer from surface to soil, evapotranspiration, and lateral diffusion.

### Tasks

#### 2.1 Infiltration

- [ ] **2.1.1** Basic infiltration: `I = alpha * h * (1 - M/M_sat)`
- [ ] **2.1.2** Vegetation enhancement: `I *= (P + k_P * W_0) / (P + k_P)`
- [ ] **2.1.3** Limit by available water and capacity
- [ ] **2.1.4** Update h and M atomically

**File:** `src/kernels/infiltration.py`

**Test cases:**
- Saturated soil: no infiltration
- No surface water: no infiltration
- Conservation: h decrease = M increase

#### 2.2 Evapotranspiration

- [ ] **2.2.1** ET rate: `E = E_max * M / (M + k_M) * (1 + beta_E * P)`
- [ ] **2.2.2** Track cumulative ET for mass balance

**File:** `src/kernels/soil.py`

#### 2.3 Deep Leakage

- [ ] **2.3.1** Leakage rate: `L = L_max * (M / M_sat)^2`
- [ ] **2.3.2** Track cumulative leakage for mass balance

**File:** `src/kernels/soil.py`

#### 2.4 Soil Moisture Diffusion

- [ ] **2.4.1** 5-point Laplacian stencil
- [ ] **2.4.2** Neumann boundary conditions (no-flux at edges)
- [ ] **2.4.3** Double buffering for parallel update
- [ ] **2.4.4** Stability check: `dt <= dx² / (4 * D_M)`

**File:** `src/kernels/soil.py`

#### 2.5 Fused Soil Update

- [ ] **2.5.1** Combine infiltration + ET + leakage + diffusion in single kernel
- [ ] **2.5.2** Use `ti.block_local` for shared memory caching

**File:** `src/kernels/soil.py`

### Phase 2 Tests

| Test | Purpose |
|------|---------|
| `test_infiltration_conservation` | h + M unchanged |
| `test_infiltration_saturation_limit` | No infiltration when M = M_sat |
| `test_et_reduces_moisture` | ET decreases M |
| `test_diffusion_smooths_gradient` | Sharp boundary diffuses |
| `test_diffusion_conservation` | Total M unchanged (no ET/leakage) |

### Exit Criteria

- [ ] Infiltration transfers water correctly with vegetation feedback
- [ ] ET and leakage remove water at expected rates
- [ ] Diffusion is stable and conserves mass
- [ ] Full soil update conserves mass (input - ET - leakage)

---

## Phase 3: Vegetation Dynamics

**Goal:** Implement vegetation growth, mortality, and seed dispersal.

### Tasks

#### 3.1 Growth

- [ ] **3.1.1** Growth rate: `G = g_max * M / (M + k_G) * P`
- [ ] **3.1.2** Monod kinetics: saturates at high moisture

**File:** `src/kernels/vegetation.py`

#### 3.2 Mortality

- [ ] **3.2.1** Mortality rate: `mu * P`
- [ ] **3.2.2** Constant mortality (simplest model)

**File:** `src/kernels/vegetation.py`

#### 3.3 Seed Dispersal (Diffusion)

- [ ] **3.3.1** 5-point Laplacian stencil
- [ ] **3.3.2** Neumann boundary conditions
- [ ] **3.3.3** Positivity constraint: `P >= 0`

**File:** `src/kernels/vegetation.py`

#### 3.4 Vegetation Update Kernel

- [ ] **3.4.1** Fused kernel: growth + mortality + dispersal
- [ ] **3.4.2** Weekly timestep (vegetation is slow)

**File:** `src/kernels/vegetation.py`

### Phase 3 Tests

| Test | Purpose |
|------|---------|
| `test_growth_increases_biomass` | Positive moisture → growth |
| `test_mortality_decreases_biomass` | Biomass decays without moisture |
| `test_dispersal_spreads_vegetation` | Sharp boundary diffuses |
| `test_vegetation_positivity` | P never goes negative |
| `test_equilibrium_biomass` | Growth = mortality at steady state |

### Exit Criteria

- [ ] Vegetation grows when moisture available
- [ ] Mortality reduces biomass at expected rate
- [ ] Dispersal smooths vegetation gradients
- [ ] System reaches equilibrium with constant moisture

---

## Phase 4: Integration

**Goal:** Combine all components into a working simulation loop.

### Tasks

#### 4.1 Rainfall Event Handling

- [ ] **4.1.1** Apply uniform rainfall to surface
- [ ] **4.1.2** Event-based: discrete pulses (not continuous drizzle)
- [ ] **4.1.3** Parameterized: depth, duration, frequency

**File:** `src/simulation.py`

#### 4.2 Hierarchical Time Stepping

- [ ] **4.2.1** Vegetation timestep: 7 days
- [ ] **4.2.2** Soil timestep: 1 day
- [ ] **4.2.3** Surface timestep: adaptive (CFL)
- [ ] **4.2.4** Subcycle surface routing during rainfall events

**File:** `src/simulation.py`

#### 4.3 Simulation Loop

- [ ] **4.3.1** Main loop structure matching spec Section 12
- [ ] **4.3.2** Event scheduling (Poisson process for rainfall)
- [ ] **4.3.3** State output at configurable intervals

**File:** `src/simulation.py`

#### 4.4 Output

- [ ] **4.4.1** GeoTIFF output for h, M, P fields
- [ ] **4.4.2** Thumbnail generation (PNG from GeoTIFF)
- [ ] **4.4.3** Mass balance diagnostics

**File:** `src/output.py`

#### 4.5 Initialization

- [ ] **4.5.1** Initialize from synthetic terrain
- [ ] **4.5.2** Random initial vegetation (small perturbation)
- [ ] **4.5.3** Uniform initial soil moisture

**File:** `src/simulation.py`

### Phase 4 Tests

| Test | Purpose |
|------|---------|
| `test_simulation_runs_without_crash` | Basic smoke test |
| `test_simulation_mass_conservation` | Track all fluxes over time |
| `test_rainfall_event_handling` | Water added correctly |
| `test_output_files_created` | GeoTIFFs written |

### Exit Criteria

- [ ] Complete simulation runs for multiple years
- [ ] Mass balance verified across all timesteps
- [ ] Output files generated and readable
- [ ] No crashes or numerical instabilities

---

## Phase 5: Validation

**Goal:** Verify the simulation produces physically reasonable results.

### Tasks

#### 5.1 Mass Conservation Verification

- [ ] **5.1.1** Implement diagnostic kernel from spec Section 8
- [ ] **5.1.2** Track: rainfall in, ET out, leakage out, boundary outflow
- [ ] **5.1.3** Assert: `|expected - actual| / expected < 1e-6`

#### 5.2 Pattern Formation

- [ ] **5.2.1** Run long simulation (decades) on tilted plane
- [ ] **5.2.2** Visual inspection: do patterns emerge?
- [ ] **5.2.3** FFT analysis of vegetation field (wavelength)

#### 5.3 Parameter Sensitivity

- [ ] **5.3.1** Vary infiltration feedback strength
- [ ] **5.3.2** Vary rainfall frequency/intensity
- [ ] **5.3.3** Document parameter space exploration

### Phase 5 Tests

| Test | Purpose |
|------|---------|
| `test_long_run_stability` | No drift over decades |
| `test_pattern_emerges` | Vegetation not uniform after spinup |
| `test_pattern_wavelength_reasonable` | Matches expected scale |

### Exit Criteria

- [ ] Mass conservation holds over long simulations
- [ ] Vegetation patterns emerge from uniform initial conditions
- [ ] Patterns have reasonable spatial scales

---

## Phase 6: Optimization

**Goal:** Achieve target performance on H100/B200 hardware.

### Tasks

#### 6.1 Profiling

- [ ] **6.1.1** Profile with Taichi's built-in profiler
- [ ] **6.1.2** Identify memory-bound vs compute-bound kernels
- [ ] **6.1.3** Measure achieved bandwidth

#### 6.2 Kernel Fusion

- [ ] **6.2.1** Fuse soil update kernels (as in spec Section 9.1)
- [ ] **6.2.2** Evaluate temporal blocking for diffusion

#### 6.3 Memory Optimization

- [ ] **6.3.1** Verify coalesced access patterns
- [ ] **6.3.2** Use `ti.block_local` for stencil operations
- [ ] **6.3.3** Minimize global memory traffic

#### 6.4 Scaling Tests

- [ ] **6.4.1** Benchmark at 1k, 5k, 10k grid sizes
- [ ] **6.4.2** Verify linear scaling
- [ ] **6.4.3** Compare against spec projections (Section 13)

### Exit Criteria

- [ ] Achieve >50% of theoretical memory bandwidth
- [ ] 10k × 10k grid runs at ~1 simulated year per minute
- [ ] No performance regressions from Phase 4

---

## Future Work (Out of Scope)

These items from spec Section 14 are explicitly deferred:

- Multiple soil layers
- Groundwater dynamics
- Multiple vegetation types
- Fire disturbance
- Grazing pressure
- Erosion/deposition
- Real DEM support
- Adaptive mesh refinement
- Implicit diffusion
- Ensemble simulations
- Climate forcing from reanalysis
- Remote sensing integration

---

## Current Status

**Active Phase:** 0 (Project Setup)

**Next Milestone:** Phase 1 complete—surface water routing on synthetic terrain

---

## Appendix: Test Coverage Goals

| Component | Unit Tests | Integration Tests | Conservation Tests |
|-----------|------------|-------------------|-------------------|
| Terrain generation | 4 | - | - |
| Flow directions | 4 | 1 | 1 |
| Flow accumulation | 3 | 1 | 1 |
| Surface routing | 4 | 2 | 2 |
| Infiltration | 3 | 1 | 1 |
| Soil moisture | 4 | 1 | 1 |
| Vegetation | 4 | 1 | 1 |
| Full simulation | - | 3 | 2 |
| **Total** | **26** | **10** | **9** |
