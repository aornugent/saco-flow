# Documentation Consolidation Plan

This plan consolidates ~2,700 lines of scattered documentation into a modular ~1,200 line system.

**Guiding principles:**
- Code is the source of truth for implementation
- Docs describe physics, algorithms, and architecture
- No pseudocode (read the actual code)
- Parameter values must match `src/params.py`

---

## Target Structure

```
saco-flow/
├── README.md              # Entry point, quick start (update)
├── AGENTS.md              # Developer guide (update reference table)
└── docs/
    ├── overview.md        # NEW: System description, feedback mechanism
    ├── physics.md         # NEW: Equations, parameters (from spec §2)
    ├── discretization.md  # NEW: Numerical methods, boundaries, rainfall
    ├── architecture.md    # NEW: Code structure, data, buffers, GPU
    ├── kernels/
    │   ├── flow.md        # NEW: Flow routing kernels
    │   ├── soil.md        # NEW: Soil moisture kernels
    │   └── vegetation.md  # NEW: Vegetation kernels
    ├── testing.md         # NEW: Conservation, verification
    └── BENCHMARKS.md      # KEEP: Performance data (unchanged)
```

---

## Phase 1: Delete Obsolete Files

**Goal:** Remove files that will be replaced by the new structure.

### Files to Delete

```
IMPLEMENTATION_PLAN.md           # All phases complete, historical
docs/overview.md                 # Replaced by new overview.md
docs/boundaries.md               # Merged into discretization.md
docs/data_structures.md          # Merged into architecture.md
docs/mass_conservation.md        # Merged into testing.md
docs/timesteps.md                # Merged into discretization.md
docs/kernels/flow_directions.md  # Replaced by kernels/flow.md
docs/kernels/surface_routing.md  # Replaced by kernels/flow.md
docs/kernels/flow_accumulation.md # Replaced by kernels/flow.md
docs/kernels/infiltration.md     # Merged into kernels/flow.md
docs/kernels/soil_moisture.md    # Replaced by kernels/soil.md
docs/kernels/vegetation.md       # Replaced by kernels/vegetation.md
```

### Verification
- Run `git status` to confirm deletions
- Ensure `docs/BENCHMARKS.md` is NOT deleted

---

## Phase 2: Create Core Documentation

**Goal:** Create `docs/overview.md`, `docs/physics.md`, `docs/discretization.md`, `docs/architecture.md`

### 2.1 docs/overview.md (~80 lines)

**Purpose:** High-level system description for newcomers.

**Content structure:**
```markdown
# System Overview

## What This System Does
- GPU-accelerated ecohydrological simulation
- Vegetation pattern formation in semiarid landscapes
- Turing-type instability from water-vegetation feedback

## State Variables
Table with columns: Field | Symbol | Units | Description
- h: Surface water depth [m]
- M: Soil moisture [m]
- P: Vegetation biomass [kg/m²]
- Z: Elevation [m] (static)
- mask: Domain mask (static, 1=active, 0=boundary)

## Governing Equations
Brief PDE forms (no constitutive relations - those go in physics.md):
- Surface: ∂h/∂t = R - I - ∇·q_s
- Soil: ∂M/∂t = I - E - L + D_M·∇²M
- Vegetation: ∂P/∂t = G·P - μP + D_P·∇²P

## Core Feedback Mechanism
1. Vegetation enhances local infiltration (positive local feedback)
2. Enhanced infiltration reduces runoff to neighbors (negative nonlocal feedback)
3. This instability drives pattern formation: bands, spots, labyrinths

## Scale and Performance
- Target: 10⁸ cells (10k × 10k at 10m resolution)
- Duration: Decades in hours
- Hardware: NVIDIA H100/B200
```

### 2.2 docs/physics.md (~200 lines)

**Purpose:** Complete mathematical specification. All equations and parameters.

**Content structure:**
```markdown
# Physics

All constitutive relations and parameters. See `overview.md` for governing equations.

## Infiltration (Critical Nonlinearity)

I(h, P, M) = α · h · [(P + k_P·W_0)/(P + k_P)] · (1 - M/M_sat)⁺

Encodes:
- Proportional to available surface water h
- Vegetation enhancement: bare soil infiltrates at rate α·W_0, dense vegetation at rate α
- Saturation limitation: no infiltration when M = M_sat

## Surface Water Flux

Kinematic wave with MFD partitioning:
q_s = h^(5/3) · √|∇Z| / n

Where n is Manning's roughness.

## Evapotranspiration

E(M, P) = E_max · M/(M + k_ET) · (1 + β_ET·P)

Monod kinetics with vegetation enhancement.

## Deep Leakage

L(M) = L_max · (M/M_sat)²

Quadratic - negligible when dry, significant near saturation.

## Vegetation Growth

G(M) = g_max · M/(M + k_G)

Monod kinetics - growth rate saturates at high moisture.

## Equilibrium Analysis

At vegetation equilibrium, growth = mortality:
g_max · M/(M + k_G) = μ

Solving: M* = μ·k_G / (g_max - μ)

This is the moisture level where vegetation is stable.
See `src/kernels/vegetation.py:compute_equilibrium_moisture()`

## Parameter Table

IMPORTANT: These values match src/params.py (the source of truth).

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Grid size | n | 64 | cells | n × n grid |
| Cell size | dx | 1.0 | m | Spatial resolution |
| Rain depth | rain_depth | 0.02 | m | Mean event depth (~20mm) |
| Storm duration | storm_duration | 0.25 | days | ~6 hours |
| Interstorm period | interstorm | 18.0 | days | ~20 events/year |
| Infiltration rate | α (alpha) | 0.1 | 1/day | |
| Vegetation half-sat | k_P | 1.0 | kg/m² | Infiltration enhancement |
| Bare soil factor | W_0 | 0.2 | - | Reduced infiltration on bare soil |
| Saturation capacity | M_sat | 0.4 | m | Maximum soil moisture |
| Max ET rate | E_max | 0.005 | m/day | ~5mm/day |
| ET half-saturation | k_ET | 0.1 | m | (Note: spec called this k_M) |
| Vegetation ET factor | β_ET (beta_ET) | 0.5 | - | |
| Max leakage rate | L_max | 0.002 | 1/day | Deep drainage coefficient |
| Soil diffusivity | D_M | 0.1 | m²/day | Lateral moisture movement |
| Max growth rate | g_max | 0.02 | 1/day | |
| Growth half-sat | k_G | 0.1 | m | |
| Mortality rate | μ (mu) | 0.001 | 1/day | |
| Dispersal diffusivity | D_P | 0.01 | m²/day | Seed spread |
| Manning's n | manning_n | 0.03 | s/m^(1/3) | Surface roughness |

## Derived Quantities

From SimulationParams:
- Cell area: dx² [m²]
- Events per year: 365 / interstorm
- Annual rainfall: rain_depth × events_per_year [m/year]
```

### 2.3 docs/discretization.md (~180 lines)

**Purpose:** Numerical methods, timesteps, boundaries, rainfall handling.

**Content structure:**
```markdown
# Discretization

Spatial and temporal numerical methods.

## Spatial Discretization

### Grid
Regular Cartesian with spacing dx = dy.

### 5-Point Laplacian (Diffusion)
∇²u ≈ (u_E + u_W + u_N + u_S - 4u_center) / dx²

Neumann (no-flux) boundary: only include neighbors where mask=1.

### 8-Connected Neighbors (Flow Routing)
Clockwise from East:
```
Index:  5  6  7
        4  X  0
        3  2  1
```
Direction 0=East, 1=SE, 2=South, etc.
Distances: cardinal=1, diagonal=√2 (in cell units).
Reverse direction: (k+4) % 8

See `src/geometry.py` for NEIGHBOR_DI, NEIGHBOR_DJ, NEIGHBOR_DIST.

### Multiple Flow Direction (MFD)
Distributes flow to lower neighbors proportional to slope^p:

f_k = max(0, S_k)^p / Σ max(0, S_m)^p

Where S_k = (Z_center - Z_neighbor) / d_k.

Flow exponent p:
- p = 1.0: Most diffuse
- p = 1.5: Default (FLOW_EXPONENT in src/kernels/flow.py)
- p > 5: Approaches single-direction (D8)

Flat cells (no downslope neighbors): flagged with flow_frac[i,j,0] = -1

## Temporal Discretization

### Hierarchical Operator Splitting

```
VEGETATION timestep (dt_veg = 7 days):
└── SOIL timestep (dt_soil = 1 day):
    └── SURFACE timestep (adaptive, fractions of a day):
```

### Timestep Constraints

| Process | Constraint | Typical (days) |
|---------|------------|----------------|
| Surface routing | dt ≤ CFL · dx / v_max | 0.0001-0.01 |
| Soil diffusion | dt ≤ 0.25 · dx² / D_M | ~2.5 |
| Vegetation diffusion | dt ≤ 0.25 · dx² / D_P | ~25 |

Only surface routing is constraining in practice.

### CFL Timestep Calculation

dt = CFL · dx / v_max

Where:
- CFL = 0.5 (default safety factor)
- v_max = max(h^(2/3) · √S / n) over all cells

See `src/kernels/flow.py:compute_cfl_timestep()`

### Simulation Loop Structure

From `src/simulation.py:run()`:
1. Rainfall events scheduled via Poisson process (exponential inter-arrival)
2. During event: adaptive subcycling (max 10,000 steps per event)
3. Daily: soil moisture update (ET, leakage, diffusion)
4. Weekly: vegetation update (growth, mortality, dispersal)
5. Every 30 days: mass balance verification

## Boundary Conditions

### Domain Mask
- mask[i,j] = 1: Active cell (interior)
- mask[i,j] = 0: Inactive/boundary

Always check mask before updating or reading neighbors.

### Diffusion: Neumann (No-Flux)
Only include neighbors where mask=1 in Laplacian. This naturally enforces zero gradient at boundaries.

### Surface Water: Outflow
Water exits domain at boundary cells with outward flow. Boundary outflow is tracked for mass balance.

### Depression Handling
Current implementation: depressions allowed (water accumulates).
Future option: Priority-Flood fill to create flow-routing surface.

## Rainfall Event Handling

From `src/simulation.py:run_rainfall_event()`:

1. Apply rainfall as intensity over duration
2. While water present (h > h_threshold) and t < duration + drainage_time:
   a. Compute CFL timestep
   b. Apply rainfall increment (if t < duration)
   c. Route surface water (two-pass scheme)
   d. Compute infiltration
   e. Track boundary outflow

### Important Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| h_threshold | 1e-6 m | Minimum water depth to continue routing |
| drainage_time | 1.0 day | Extra time after event for drainage |
| MIN_DEPTH | 1e-8 m | Threshold for flow computation |
| MIN_SLOPE_SUM | 1e-10 | Flat cell detection threshold |
```

### 2.4 docs/architecture.md (~250 lines)

**Purpose:** Code organization, data structures, buffer patterns, GPU optimization.

**Content structure:**
```markdown
# Architecture

Code organization for a ~2000 line GPU-accelerated simulation.

## Design Principles

1. **Simplicity first** - Flat structure, minimal indirection
2. **Correctness before speed** - Every kernel needs conservation test
3. **Explicit over implicit** - Physical variable names, units in comments
4. **Taichi-idiomatic** - Leverage ti.template(), ti.static(), ping-pong buffers

## Module Structure

```
src/
├── config.py          # Taichi initialization, backend detection
├── geometry.py        # Grid constants, neighbor offsets, generic diffusion
├── params.py          # SimulationParams dataclass + validation
├── fields.py          # Field allocation, swap_buffers()
├── initialization.py  # Terrain and vegetation setup
├── simulation.py      # Main loop, operator splitting, SimulationState
├── diagnostics.py     # MassBalance class, conservation checks
├── output.py          # GeoTIFF and PNG export
└── kernels/
    ├── flow.py        # MFD, flow accumulation, surface routing
    ├── infiltration.py # Vegetation-enhanced infiltration
    ├── soil.py        # ET, leakage, diffusion (fused + naive)
    └── vegetation.py  # Growth, mortality, dispersal (fused + naive)
```

## Data Structures

### Field Storage (Structure of Arrays)

All fields stored as separate contiguous arrays for coalesced GPU access:

```python
h = ti.field(dtype=ti.f32, shape=(n, n))      # Surface water [m]
M = ti.field(dtype=ti.f32, shape=(n, n))      # Soil moisture [m]
P = ti.field(dtype=ti.f32, shape=(n, n))      # Vegetation [kg/m²]
Z = ti.field(dtype=ti.f32, shape=(n, n))      # Elevation [m]
mask = ti.field(dtype=ti.i8, shape=(n, n))    # Domain mask
flow_frac = ti.field(dtype=ti.f32, shape=(n, n, 8))  # MFD fractions
```

### Memory Budget (10⁸ cells)

| Fields | Bytes/cell | Total |
|--------|------------|-------|
| Primary (h, M, P) | 12 | 1.2 GB |
| Flow fractions | 32 | 3.2 GB |
| Static (Z, mask) | 5 | 0.5 GB |
| Buffers | 16 | 1.6 GB |
| **Total** | | **~6.5 GB** |

This is <10% of H100's 80GB HBM.

### SimulationState

Bundles all runtime state (from `src/simulation.py`):
- fields: SimpleNamespace with all Taichi fields
- mass_balance: MassBalance tracker
- current_day: Simulation time
- dx: Cell size

### MassBalance

Tracks cumulative fluxes (from `src/diagnostics.py`):
- initial_water, cumulative_rain, cumulative_et, cumulative_leakage, cumulative_outflow
- expected_water(): Computes expected total
- check(actual): Verifies conservation, raises on violation

## Buffer Strategy: Ping-Pong

### Why Ping-Pong
Stencil operations read neighbors while writing center. Without double-buffering, race conditions occur.

### Pattern
```python
# Reads from fields.M, writes to fields.M_new
laplacian_diffusion_step(fields.M, fields.M_new, ...)

# O(1) pointer swap (not O(n²) copy)
swap_buffers(fields, "M")
# Now fields.M holds updated values
```

### Correctness Rules

**Rule 1: Point-wise operations can modify in-place**
```python
et_leakage_step_fused(M, P, mask, ...)  # Modifies M[i,j] directly
```
Each cell is independent, no neighbor reads.

**Rule 2: Stencil operations use double buffer**
```python
laplacian_diffusion_step(M, M_new, mask, ...)  # Reads M, writes M_new
```
Must be followed by swap_buffers().

**Rule 3: Stencil must be LAST operation before swap**
After stencil writes to M_new, M is stale. Don't read M after stencil.

### Surface Water: Two-Pass (Not Ping-Pong)

Surface routing uses different pattern:
1. compute_outflow(): reads h, writes q_out
2. apply_fluxes(): reads q_out from neighbors, updates h in-place

No race condition because stencil reads q_out, not h.

## GPU Optimization

### Bottleneck: Memory Bandwidth

Arithmetic intensity ~0.2-0.5 FLOP/byte. Well below compute-bound threshold.

### Fused Kernels

Combine point-wise operations to reduce memory traffic:

Before (naive):
- evapotranspiration_step: Read M,P → Write M
- leakage_step: Read M → Write M
- Total: 8 reads + 4 writes = 48 bytes/cell

After (fused):
- et_leakage_step_fused: Read M,P → Write M
- Total: 4 reads + 2 writes = 24 bytes/cell (2× improvement)

Both naive and fused versions are kept. Fused is default; naive for regression testing.

### Shared Memory for Stencils

`ti.block_local()` caches stencil reads in shared memory:
```python
@ti.kernel
def laplacian_diffusion_step(...):
    ti.block_local(field)  # Cache in shared memory
    for i, j in ti.ndrange(...):
        # Neighbor reads benefit from cache
```

### Coalesced Memory Access

Threads in a warp should access contiguous memory:
```python
# Correct: j innermost (row-major, coalesced)
for i, j in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...
```

## CRITICAL: ti.template() vs Closure Capture

**Always pass fields as ti.template() arguments. Never capture in closures.**

```python
# WRONG: Field captured at JIT compile time
@ti.kernel
def bad_kernel():
    for i, j in M:  # M baked in at compile
        M_new[i, j] = M[i, j]

# CORRECT: Field passed as argument
@ti.kernel
def good_kernel(M: ti.template(), M_new: ti.template()):
    for i, j in M:
        M_new[i, j] = M[i, j]
```

If a field will be swapped via swap_buffers(), it MUST be passed as ti.template().
```

---

## Phase 3: Create Kernel Documentation

**Goal:** Create `docs/kernels/flow.md`, `docs/kernels/soil.md`, `docs/kernels/vegetation.md`

### 3.1 docs/kernels/flow.md (~120 lines)

**Purpose:** Flow direction, accumulation, and surface routing kernels.

**Content structure:**
```markdown
# Flow Kernels

Surface water routing using MFD and kinematic wave.

Source: `src/kernels/flow.py`

## Overview

Three-stage process:
1. **Flow directions** (once): Compute MFD fractions from elevation
2. **Flow accumulation** (as needed): Iterative parallel relaxation
3. **Surface routing** (per timestep): Two-pass kinematic wave

## Flow Direction Computation

`compute_flow_directions(Z, mask, flow_frac, dx, p)`

Distributes outflow to lower neighbors proportional to slope^p.

### Algorithm
For each cell:
1. Compute slope S_k = (Z_center - Z_k) / d_k to each neighbor
2. Compute slope^p for positive (downslope) slopes
3. Normalize to fractions summing to 1

### Edge Cases
- **Flat cells**: slope_sum < MIN_SLOPE_SUM → flag with flow_frac[i,j,0] = -1
- **Boundary cells**: Can flow to inactive cells (water exits domain)

## Flow Accumulation

`compute_flow_accumulation(local_source, flow_acc, flow_acc_new, flow_frac, mask)`

Iterative parallel algorithm:
```
A[i,j] = local[i,j] + Σ f_{k→(i,j)} · A[k]
```

Converges in ~20-50 iterations for 10k×10k grids. Returns iteration count.

## Surface Routing

`route_surface_water(h, Z, flow_frac, mask, q_out, dx, dt, manning_n)`

Two-pass scheme for mass conservation:

### Pass 1: compute_outflow()
Kinematic wave velocity:
```
v = h^(2/3) · √S_max / manning_n
q_out = min(h/dt, v·h/dx)  # CFL-limited
```

### Pass 2: apply_fluxes()
```
h_new = h - q_out·dt + Σ(inflow from donors)
```
Returns boundary outflow for mass balance.

## CFL Timestep

`compute_cfl_timestep(h, Z, flow_frac, mask, dx, manning_n, cfl=0.5)`

```
dt = cfl · dx / v_max
```

Returns infinity if no flow (v_max ≈ 0).

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| FLOW_EXPONENT | 1.5 | MFD concentration |
| MIN_DEPTH | 1e-8 m | Flow threshold |
| MIN_SLOPE_SUM | 1e-10 | Flat cell detection |

## Tests Required

1. Tilted plane → all flow in one direction
2. Symmetric valley → flow splits equally
3. Mass conservation (closed domain)
4. CFL stability maintained
```

### 3.2 docs/kernels/soil.md (~100 lines)

**Purpose:** Soil moisture dynamics kernels.

**Content structure:**
```markdown
# Soil Kernels

Evapotranspiration, leakage, and lateral diffusion.

Source: `src/kernels/soil.py`

## Governing Equation

∂M/∂t = -E(M,P) - L(M) + D_M·∇²M

Infiltration (I) handled separately in `src/kernels/infiltration.py`.

## Evapotranspiration

`evapotranspiration_step(M, P, mask, E_max, k_M, beta_E, dt)`

```
E = E_max · M/(M + k_ET) · (1 + β_ET·P) · dt
```

Monod kinetics with vegetation enhancement. Limited to available moisture.

## Deep Leakage

`leakage_step(M, mask, L_max, M_sat, dt)`

```
L = L_max · (M/M_sat)² · dt
```

Quadratic - negligible when dry, significant near saturation.

## Fused Kernel

`et_leakage_step_fused(M, P, mask, E_max, k_M, beta_E, L_max, M_sat, dt)`

Combines ET + leakage in single pass:
- 2× less memory traffic
- Sequential application: ET first, then leakage on remaining moisture
- Returns [total_et, total_leakage] vector

## Diffusion

Uses generic `laplacian_diffusion_step()` from `src/geometry.py`:
- 5-point Laplacian stencil
- Neumann (no-flux) boundaries
- ti.block_local() for shared memory caching
- Non-negativity constraint

## Combined Step

`soil_moisture_step(fields, E_max, k_ET, beta_ET, L_max, M_sat, D_M, dx, dt)`

1. Fused ET+leakage (point-wise, in-place on M)
2. Diffusion (stencil, M → M_new)
3. swap_buffers(fields, "M")

Returns (total_et, total_leakage) for mass balance.

## Naive Version

`soil_moisture_step_naive(...)` - Separate kernels for regression testing.

## Stability

```
dt ≤ 0.25 · dx² / D_M
```

See `compute_diffusion_timestep()`.

## Tests Required

1. ET reduces moisture (proportional to M and P)
2. Leakage quadratic in saturation ratio
3. Diffusion smooths gradients
4. Diffusion alone conserves mass
5. Fused matches naive within tolerance
```

### 3.3 docs/kernels/vegetation.md (~100 lines)

**Purpose:** Vegetation dynamics kernels.

**Content structure:**
```markdown
# Vegetation Kernels

Growth, mortality, and seed dispersal.

Source: `src/kernels/vegetation.py`

## Governing Equation

∂P/∂t = G(M)·P - μ·P + D_P·∇²P

## Growth

`growth_step(P, M, mask, g_max, k_G, dt)`

Monod kinetics:
```
G(M) = g_max · M / (M + k_G)
dP = G(M) · P · dt
```

Growth rate saturates at high moisture.

## Mortality

`mortality_step(P, mask, mu, dt)`

Constant rate:
```
dP = -μ · P · dt
```

Limited to available biomass.

## Fused Kernel

`growth_mortality_step_fused(P, M, mask, g_max, k_G, mu, dt)`

Combines growth + mortality:
- Growth applied first
- Mortality applied to grown value (matches naive sequential behavior)
- Returns [total_growth, total_mortality] vector

## Diffusion (Seed Dispersal)

Uses generic `laplacian_diffusion_step()` from `src/geometry.py`.

## Combined Step

`vegetation_step(fields, g_max, k_G, mu, D_P, dx, dt)`

1. Fused growth+mortality (point-wise, in-place on P)
2. Diffusion (stencil, P → P_new)
3. swap_buffers(fields, "P")

Returns (total_growth, total_mortality).

## Equilibrium Analysis

`compute_equilibrium_moisture(g_max, k_G, mu)`

At equilibrium, growth rate = mortality:
```
g_max · M / (M + k_G) = μ
```

Solving:
```
M* = μ · k_G / (g_max - μ)
```

Returns infinity if growth never exceeds mortality (g_max ≤ μ).

## Timestep

Weekly updates (dt_veg = 7 days). Vegetation dynamics are slow.

Stability: dt ≤ 0.25 · dx² / D_P (usually satisfied).

## Tests Required

1. Growth increases with moisture
2. Mortality proportional to biomass
3. Dispersal smooths gradients
4. Equilibrium: stable at M = M*
5. Fused matches naive within tolerance
```

---

## Phase 4: Create Testing Documentation and Update Entry Points

**Goal:** Create `docs/testing.md`, update `AGENTS.md` and `README.md`

### 4.1 docs/testing.md (~100 lines)

**Content structure:**
```markdown
# Testing

Mass conservation verification and debugging patterns.

## Conservation Law

Total water must satisfy:
```
d/dt(H_total + M_total) = R_total - ET_total - L_total - Q_out
```

Where:
- H_total = Σ h[i,j] · dx²
- M_total = Σ M[i,j] · dx²
- Q_out = boundary outflow

## MassBalance Class

From `src/diagnostics.py`:

```python
@dataclass
class MassBalance:
    initial_water: float = 0.0
    cumulative_rain: float = 0.0
    cumulative_et: float = 0.0
    cumulative_leakage: float = 0.0
    cumulative_outflow: float = 0.0

    def expected_water(self) -> float:
        return initial + rain - et - leakage - outflow

    def check(self, actual, rtol=1e-4, atol=1e-8):
        # Raises AssertionError if violated
```

## Per-Kernel Conservation

| Kernel | Check |
|--------|-------|
| Infiltration | Δh = -ΔM (water transfers, not created/destroyed) |
| Surface routing | Σh unchanged (closed domain) |
| Soil diffusion | ΣM unchanged |
| Vegetation diffusion | ΣP unchanged |
| ET/Leakage | Track cumulative removal |
| Boundary outflow | Track in apply_fluxes() |

## Common Violations

1. **Boundary leaks**: Water escaping without tracking
2. **Timestep instability**: CFL violation causing negative water
3. **Clamping errors**: max(0, h) without tracking removed water
4. **Float accumulation**: Small errors compound over long runs

## Debugging Mass Errors

1. Test on flat terrain with uniform initial conditions
2. Track each flux term separately
3. Check boundary cells - water escaping uncounted?
4. Verify timestep satisfies stability constraints
5. Run conservation check every timestep during debugging

## Running Tests

```bash
# Fast tests only
pytest -m "not slow"

# All tests including slow validation
pytest

# With coverage
pytest --cov=src --cov-report=term-missing
```

**Test markers:**
- `@pytest.mark.slow`: Multi-year simulations (minutes)
- Unmarked: Unit tests (seconds)

## Kernel Equivalence Tests

Fused kernels must match naive:
```python
def test_soil_fused_matches_naive():
    # Run naive
    soil_moisture_step_naive(fields, ...)

    # Run fused
    soil_moisture_step(fields_copy, ...)

    assert np.allclose(fields.M, fields_copy.M, rtol=1e-5)
```
```

### 4.2 Update AGENTS.md

Replace the "Before Starting Work" table with:

```markdown
## Before Starting Work

Read the docs relevant to your task:

| Task | Read |
|------|------|
| Understanding the system | `docs/overview.md` |
| Physics and parameters | `docs/physics.md` |
| Numerical methods | `docs/discretization.md` |
| Code structure / GPU | `docs/architecture.md` |
| Flow/routing kernels | `docs/kernels/flow.md` |
| Soil moisture kernels | `docs/kernels/soil.md` |
| Vegetation kernels | `docs/kernels/vegetation.md` |
| Testing/debugging | `docs/testing.md` |
| Performance data | `docs/BENCHMARKS.md` |
```

Also remove references to `ecohydro_spec.md` and `IMPLEMENTATION_PLAN.md`.

### 4.3 Update README.md

Update the "Project Structure" section to reflect new docs structure.
Update "Key Documentation" table.
Remove reference to IMPLEMENTATION_PLAN.md.
Keep quick start and installation sections largely unchanged.

---

## Phase 5: Final Cleanup and Verification

**Goal:** Delete `ecohydro_spec.md`, verify all references work.

### Actions

1. Delete `ecohydro_spec.md` (content now distributed across docs/)
2. Verify no broken references in any .md file
3. Run `pytest` to ensure tests still pass
4. Commit all changes with descriptive message

### Verification Checklist

- [ ] All old docs/ files deleted
- [ ] IMPLEMENTATION_PLAN.md deleted
- [ ] ecohydro_spec.md deleted
- [ ] New docs/ structure complete (7 files + BENCHMARKS.md)
- [ ] AGENTS.md updated with new reference table
- [ ] README.md updated
- [ ] No broken cross-references
- [ ] Parameter values match src/params.py
- [ ] Tests pass

---

## Summary

| Phase | Files Created | Files Deleted | Lines (approx) |
|-------|---------------|---------------|----------------|
| 1 | 0 | 12 | -500 |
| 2 | 4 | 0 | +710 |
| 3 | 3 | 0 | +320 |
| 4 | 1 + updates | 0 | +100 |
| 5 | 0 | 1 | -990 |
| **Total** | 8 new | 13 deleted | ~1,130 final |

Final docs: ~1,200 lines (down from ~2,700)
