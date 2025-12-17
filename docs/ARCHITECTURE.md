# EcoHydro Architecture

Simple architecture for a ~2000 line ecohydrological simulation. Add abstraction only when you feel pain.

## Design Principles

1. **Simplicity first** — Flat structure, minimal indirection
2. **Correctness before speed** — Every kernel needs a mass conservation test
3. **Explicit over implicit** — Physical variable names, units in comments
4. **Taichi-idiomatic** — Leverage `ti.template()`, `ti.static()`, ping-pong buffers
5. **Test with physics** — Validate against analytical solutions, not just conservation

## Module Structure

```
src/
├── geometry.py      # Grid constants, neighbor offsets
├── params.py        # Parameter dataclass + validation
├── fields.py        # All Taichi fields, swap_buffers(), initialization
├── kernels/         # Taichi kernels (one file per process)
│   ├── flow.py
│   ├── infiltration.py
│   ├── soil.py
│   └── vegetation.py
├── simulation.py    # Main loop, operator splitting
└── diagnostics.py   # Conservation check, timing utilities
```

## Key Components

### 1. Geometry (`geometry.py`)

Centralized neighbor indexing prevents subtle bugs and enables `ti.static()` unrolling.

```python
# Compile-time constants for 8-neighbor connectivity (D8)
NEIGHBOR_DI = ti.Vector([-1, -1, -1, 0, 0, 1, 1, 1])
NEIGHBOR_DJ = ti.Vector([-1, 0, 1, -1, 1, -1, 0, 1])
NEIGHBOR_DIST = ti.Vector([1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414])

@ti.func
def for_each_neighbor(i: int, j: int):
    """Use with ti.static(range(8)) for compile-time unrolling."""
    for k in ti.static(range(8)):
        ni = i + NEIGHBOR_DI[k]
        nj = j + NEIGHBOR_DJ[k]
        # ... check bounds and mask before use
```

**Why centralize**: Neighbor indexing is used in every kernel. One place to get it right.

### 2. Parameters (`params.py`)

Simple dataclass with validation. Catches typos and invalid values at load time.

```python
@dataclass
class SimulationParams:
    # Grid
    nx: int
    ny: int
    dx: float  # [m] cell size

    # Infiltration
    alpha: float      # [1/m] sorptive number
    k_sat: float      # [m/s] saturated conductivity

    # ... other parameters with units in comments

    def __post_init__(self):
        """Validate parameter ranges."""
        _check_positive(self, 'dx', 'nx', 'ny')
        _check_non_negative(self, 'alpha', 'k_sat')

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationParams":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

def _check_positive(obj, *names):
    for name in names:
        val = getattr(obj, name)
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")

def _check_non_negative(obj, *names):
    for name in names:
        val = getattr(obj, name)
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")
```

**Why dataclass**: Self-documenting, IDE autocomplete, validation at load time.

### 3. Fields (`fields.py`)

All Taichi fields declared in one place with ping-pong buffer management.

```python
import taichi as ti

# State fields (double-buffered for stencil operations)
M = ti.field(ti.f32)       # [m³/m³] soil moisture - buffer A
M_new = ti.field(ti.f32)   # [m³/m³] soil moisture - buffer B
P = ti.field(ti.f32)       # [kg/m²] plant biomass - buffer A
P_new = ti.field(ti.f32)   # [kg/m²] plant biomass - buffer B

# Surface water (uses two-pass routing, not ping-pong)
h = ti.field(ti.f32)       # [m] surface water depth
q_out = ti.field(ti.f32)   # [m/s] outflow rate (intermediate)

# Static fields (read-only during simulation)
Z = ti.field(ti.f32)       # [m] elevation
mask = ti.field(ti.i32)    # 1=active, 0=inactive
flow_frac = ti.field(ti.f32)  # D8 flow fractions

def allocate(nx: int, ny: int):
    """Allocate all fields for given grid size."""
    ti.root.dense(ti.ij, (nx, ny)).place(M, M_new, P, P_new)
    ti.root.dense(ti.ij, (nx, ny)).place(h, q_out, Z, mask)
    ti.root.dense(ti.ijk, (nx, ny, 8)).place(flow_frac)
```

## Buffer Strategy: Ping-Pong (No Copy)

Use **ping-pong buffering**: alternate which buffer is source vs destination each step. No copy needed.

```python
# Main simulation loop
for step in range(num_steps):
    if step % 2 == 0:
        M_cur, M_nxt = M, M_new
        P_cur, P_nxt = P, P_new
    else:
        M_cur, M_nxt = M_new, M
        P_cur, P_nxt = P_new, P

    # All kernels use M_cur/M_nxt consistently this step
    soil_moisture_step(M_cur, M_nxt, P_cur, ...)
    vegetation_step(P_cur, P_nxt, M_nxt, ...)  # Note: reads M_nxt (updated soil)
```

### Correctness Rules for Mixed In-Place + Stencil

**The pattern for each physics step:**
```
[point-wise ops on current]* → [stencil: current → next] → (buffer flips)
```

**Rule 1: Point-wise ops modify current buffer in-place**

Point-wise operations (no neighbor reads) can safely modify the current buffer:
```python
@ti.kernel
def evapotranspiration_step(M_cur: ti.template(), ...):
    for i, j in M_cur:
        if mask[i, j] == 0:
            continue
        # Only reads/writes M_cur[i, j] — no neighbors
        et = E_max * M_cur[i, j] / (M_cur[i, j] + k_M) * dt
        M_cur[i, j] -= ti.min(et, M_cur[i, j])
```

**Why safe**: Each cell is independent. Update order doesn't matter.

**Rule 2: Stencil op must be LAST and writes to next buffer**

Stencil operations (read neighbors) must:
1. Be the final operation in the sequence for that field
2. Read from current buffer, write to next buffer

```python
@ti.kernel
def diffusion_step(M_cur: ti.template(), M_nxt: ti.template(), ...):
    for i, j in M_cur:
        if mask[i, j] == 0:
            continue
        # Reads neighbors from M_cur
        laplacian = (M_cur[i-1, j] + M_cur[i+1, j] +
                     M_cur[i, j-1] + M_cur[i, j+1] - 4*M_cur[i, j])
        # Writes to M_nxt
        M_nxt[i, j] = M_cur[i, j] + D * dt / (dx*dx) * laplacian
```

**Why last**: After stencil writes to `M_nxt`, the current buffer (`M_cur`) is stale. Any subsequent point-wise op would modify the wrong buffer.

**Rule 3: Complete sequence for soil moisture**

```python
def soil_moisture_step(M_cur, M_nxt, P_cur, ...):
    # 1. Point-wise: modify M_cur in-place (order doesn't matter)
    evapotranspiration_step(M_cur, P_cur, ...)  # M_cur[i,j] -= ET
    leakage_step(M_cur, ...)                     # M_cur[i,j] -= L

    # 2. Stencil: MUST be last, reads M_cur neighbors, writes M_nxt
    diffusion_step(M_cur, M_nxt, ...)

    # After this: M_nxt holds the updated state, M_cur is stale
```

### What Breaks Correctness

**❌ Wrong: Stencil followed by point-wise**
```python
def broken_step(M_cur, M_nxt, ...):
    diffusion_step(M_cur, M_nxt, ...)  # writes to M_nxt
    leakage_step(M_cur, ...)           # WRONG: modifies stale M_cur!
```

**❌ Wrong: Two stencils in same step**
```python
def broken_step(M_cur, M_nxt, ...):
    diffusion_step(M_cur, M_nxt, ...)   # M_nxt has diffused values
    some_other_stencil(M_cur, M_nxt, ...)  # WRONG: reads stale M_cur!
```

**❌ Wrong: Reading wrong buffer after swap**
```python
# After soil step, M_nxt has updated soil moisture
vegetation_step(P_cur, P_nxt, M_cur, ...)  # WRONG: should read M_nxt!
```

**✓ Correct: Consistent buffer usage within timestep**
```python
def timestep(step):
    if step % 2 == 0:
        M_cur, M_nxt = M, M_new
    else:
        M_cur, M_nxt = M_new, M

    soil_moisture_step(M_cur, M_nxt, ...)   # M_nxt now current
    vegetation_step(P_cur, P_nxt, M_nxt, ...)  # reads updated M_nxt ✓
```

### Surface Water: Two-Pass (Not Ping-Pong)

Surface routing uses a different pattern — two-pass with intermediate `q_out`:

```python
def route_surface_water(h, q_out, ...):
    # Pass 1: compute outflow rates (reads h, writes q_out)
    compute_outflow(h, q_out, ...)

    # Pass 2: apply fluxes (reads q_out from neighbors, updates h)
    apply_fluxes(h, q_out, ...)
```

**Why this works**: Pass 1 stores velocities in `q_out`. Pass 2 reads `q_out[neighbors]`, not `h[neighbors]`. No race condition because the stencil reads from a separately-computed field.

### Summary Table

| Operation Type | Buffer Pattern | Example |
|---------------|----------------|---------|
| Point-wise | In-place on current | ET, leakage, growth, mortality |
| Stencil (diffusion) | current → next (ping-pong) | Soil/vegetation diffusion |
| Two-pass routing | h + intermediate q_out | Surface water routing |

### 4. Kernels (`kernels/`)

One file per physical process. Follow the pattern:

```python
@ti.kernel
def update_soil_moisture(dt: ti.f32):
    """Update soil moisture. Physics: dM/dt = I - ET - L"""
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue

        # Compute fluxes (with units in comments)
        infiltration = ...   # [m³/m³/s]
        et = ...             # [m³/m³/s]
        leakage = ...        # [m³/m³/s]

        # Update with clamping to physical bounds
        dM = (infiltration - et - leakage) * dt
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M[i, j] + dM))
```

**Pattern**: Check mask → compute fluxes → clamp to bounds → write to buffer.

### 5. Simulation (`simulation.py`)

Main loop with operator splitting and ping-pong buffer management.

```python
class Simulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.t = 0.0
        self.step_count = 0

        # Allocate fields
        fields.allocate(params.nx, params.ny)

        # Load initial conditions
        self._load_dem(params.dem_path)
        self._initialize_state()

    def _get_buffers(self):
        """Return (current, next) buffer pairs based on step parity."""
        if self.step_count % 2 == 0:
            return (fields.M, fields.M_new), (fields.P, fields.P_new)
        else:
            return (fields.M_new, fields.M), (fields.P_new, fields.P)

    def step(self, dt: float):
        """One timestep with operator splitting."""
        (M_cur, M_nxt), (P_cur, P_nxt) = self._get_buffers()

        # 1. Surface routing (two-pass, no ping-pong)
        kernels.route_surface(fields.h, fields.q_out, dt)

        # 2. Infiltration (point-wise: h and M_cur)
        kernels.infiltrate(fields.h, M_cur, P_cur, dt)

        # 3. Soil moisture: point-wise then stencil
        kernels.soil_moisture_step(M_cur, M_nxt, P_cur, dt)
        # After: M_nxt has updated soil state

        # 4. Vegetation: point-wise then stencil (reads M_nxt!)
        kernels.vegetation_step(P_cur, P_nxt, M_nxt, dt)
        # After: P_nxt has updated vegetation state

        # No copy needed — next step will flip buffer roles
        self.t += dt
        self.step_count += 1

    def current_M(self):
        """Return whichever buffer holds current soil moisture."""
        return fields.M if self.step_count % 2 == 0 else fields.M_new
```

### 6. Diagnostics (`diagnostics.py`)

Simple functions for mass conservation and timing.

```python
def check_conservation(tolerance: float = 1e-6) -> bool:
    """Verify total mass is conserved within tolerance."""
    total_water = compute_total_water()
    total_biomass = compute_total_biomass()

    if abs(total_water - initial_water) > tolerance:
        print(f"Water conservation error: {total_water - initial_water}")
        return False
    return True

@ti.kernel
def compute_total_water() -> ti.f32:
    """Sum all water in system (surface + soil)."""
    total = 0.0
    for i, j in h:
        if mask[i, j] == 1:
            total += h[i, j] + M[i, j] * soil_depth
    return total
```

## GPU Optimization

### Target Hardware

| GPU | Architecture | HBM | Bandwidth | FP32 TFLOPS |
|-----|--------------|-----|-----------|-------------|
| B200 | Blackwell (sm_100) | 192GB HBM3e | 8 TB/s | ~2500 |
| H100 | Hopper (sm_90) | 80GB HBM3 | 3.35 TB/s | ~1979 |

**Performance target:** 10k×10k grid at ≥1 simulated year per wall-clock minute.

### Bottleneck Analysis

The simulation is **memory-bandwidth bound**, not compute bound:
- Arithmetic intensity: ~0.2-0.5 FLOP/byte (crossover at ~0.6 FLOP/byte on H100)
- Each cell update: ~10-20 FLOPs but reads/writes ~40-100 bytes
- Stencil operations dominate: diffusion reads 5 neighbors per cell

### Memory Budget

| Grid Size | Cells | Memory | H100 (80GB) | B200 (192GB) |
|-----------|-------|--------|-------------|--------------|
| 1k × 1k | 10⁶ | ~77 MB | ✓ | ✓ |
| 10k × 10k | 10⁸ | ~7.7 GB | ✓ | ✓ |
| 20k × 20k | 4×10⁸ | ~31 GB | ✓ | ✓ |
| 50k × 50k | 2.5×10⁹ | ~193 GB | ✗ | ✓ |

### Current Architecture: What Works

1. **Ping-pong buffering**: No `copy_field()` calls needed. Buffer roles flip each step.
2. **Two-pass routing**: `compute_outflow` stores to `q_out`, then `apply_fluxes` reads `q_out[neighbors]`. No race conditions.
3. **Centralized neighbor offsets**: `utils.py` has `NEIGHBOR_DI/DJ/DIST` with `ti.static(range(8))` unrolling.
4. **Atomic reductions**: `ti.atomic_add` for mass totals. Optimize only if profiling shows bottleneck.

### Correctness Constraints

1. **Buffer consistency**: After `soil_moisture_step(M_cur, M_nxt)`, vegetation must read `M_nxt`.
2. **Point-wise before stencil**: Don't fuse naively. Point-wise ops must complete before stencil reads neighbors.
3. **Iterative flow accumulation**: Multiple iterations until convergence. Consider priority-flood for large grids.

### Kernel Fusion Strategy

**Goal:** Reduce global memory traffic by combining operations.

Before (naive soil step):
```
1. evapotranspiration: Read M,P → Write M
2. leakage: Read M → Write M
3. diffusion: Read M → Write M_new
Total: 8 reads + 4 writes = 48 bytes/cell
```

After (fused — point-wise only):
```
1. fused_et_leakage: Read M,P → Write M (in-place)
2. diffusion: Read M → Write M_new
Total: 4 reads + 2 writes = 24 bytes/cell (2× improvement)
```

**Constraint:** Cannot fuse point-wise with stencil in single kernel due to neighbor dependencies.

### Shared Memory for Stencils

Use `ti.block_local()` to cache stencil reads:

```python
@ti.kernel
def diffusion_cached(M_cur: ti.template(), M_nxt: ti.template(), ...):
    ti.block_local(M_cur)  # Cache in shared memory

    for i, j in ti.ndrange((1, n-1), (1, n-1)):
        # Stencil reads benefit from L1/shared memory
        laplacian = M_cur[i-1,j] + M_cur[i+1,j] + M_cur[i,j-1] + M_cur[i,j+1] - 4*M_cur[i,j]
        M_nxt[i, j] = M_cur[i, j] + D * dt / (dx*dx) * laplacian
```

### Coalesced Memory Access

GPUs achieve peak bandwidth when threads access contiguous memory.

```python
# Correct: j innermost (row-major, coalesced)
for i, j in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...

# Wrong: strided access (not coalesced)
for j, i in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...  # 4*n byte stride between threads
```

### Benchmarking Protocol

1. **Warmup:** 10 timesteps (triggers JIT compilation)
2. **Measurement:** 100+ vegetation timesteps (700+ simulated days)
3. **Metrics:**
   - Wall time per simulated year
   - Cells updated per second
   - Achieved bandwidth vs theoretical peak
4. **Grid sizes:** 1k, 5k, 10k

### Profiling

```python
ti.init(arch=ti.cuda, kernel_profiler=True)
# ... run simulation ...
ti.profiler.print_kernel_profiler_info()
```

### Future Optimizations

| Optimization | When to Consider |
|-------------|------------------|
| Tiled memory layout | Memory bandwidth bound, grid > 1024×1024 |
| Hierarchical reduction | Grid > 2048×2048 and reduction > 5% of runtime |
| Kernel fusion (point-wise) | Multiple sequential point-wise ops on same field |
| `ti.block_dim()` tuning | After basic implementation works |
| Temporal blocking | Pure diffusion with many small timesteps |

## Testing Strategy

Tests should validate physics, not just code paths. Prefer analytical solutions over conservation-only tests.

### 1. Analytical Solution Tests

Compare simulation output against known exact solutions:

```python
def test_diffusion_gaussian_decay():
    """1D diffusion of Gaussian: M(x,t) = M₀/√(4πDt) · exp(-x²/4Dt)"""
    D = 0.1  # diffusivity
    t = 1.0  # time

    # Initialize Gaussian pulse
    M_initial = gaussian_pulse(sigma=1.0)

    # Run diffusion
    for _ in range(100):
        diffusion_step(M_cur, M_nxt, D=D, dt=0.01)
        M_cur, M_nxt = M_nxt, M_cur

    # Compare to analytical solution
    M_analytical = gaussian_pulse(sigma=np.sqrt(1 + 4*D*t))
    assert np.allclose(M_cur.to_numpy(), M_analytical, rtol=0.01)

def test_kinematic_wave_velocity():
    """Surface water on uniform slope: v = h^(2/3)·√S/n"""
    slope = 0.01
    h = 0.1  # depth
    n = 0.03  # Manning's

    v_expected = h**(2/3) * np.sqrt(slope) / n

    # Run routing, measure front propagation
    front_position = track_wetting_front(...)
    v_measured = front_position / elapsed_time

    assert abs(v_measured - v_expected) / v_expected < 0.05

def test_infiltration_green_ampt():
    """Cumulative infiltration: F = Kt + ψΔθ·ln(1 + F/ψΔθ)"""
    # Compare against Green-Ampt analytical solution
    ...
```

### 2. Equilibrium Tests

Systems should reach known steady states:

```python
def test_vegetation_equilibrium():
    """At equilibrium: growth rate = mortality → G(M*) = μ"""
    # M* = μ·k_G / (g_max - μ)
    M_equilibrium = mu * k_G / (g_max - mu)

    # Initialize at equilibrium moisture
    fill_field(M, M_equilibrium)
    fill_field(P, 1.0)

    # Run for many timesteps
    for _ in range(1000):
        vegetation_step(...)

    # Vegetation should remain stable (not grow or die)
    P_final = compute_total(P)
    assert abs(P_final - P_initial) / P_initial < 0.01
```

### 3. Conservation Tests

Mass/energy balance as a sanity check (necessary but not sufficient):

```python
def test_closed_system_conservation():
    """No inflow/outflow → total mass constant."""
    # Use closed boundary (all mask edges = 0)
    initial_mass = compute_total_water()

    for _ in range(100):
        simulation.step(dt)

    final_mass = compute_total_water()
    assert abs(final_mass - initial_mass) / initial_mass < 1e-6

def test_flux_accounting():
    """Input - output = change in storage."""
    rain_total = 0.0
    et_total = 0.0
    outflow_total = 0.0

    for _ in range(100):
        rain_total += apply_rainfall(...)
        et_total += evapotranspiration_step(...)
        outflow_total += route_surface_water(...)

    storage_change = compute_total_water() - initial_water
    expected_change = rain_total - et_total - outflow_total

    assert abs(storage_change - expected_change) < 1e-6
```

### 4. Kernel Equivalence Tests

When optimizing, verify fused kernels match naive:

```python
def test_soil_fused_matches_naive():
    """Fused kernel produces same result as sequential kernels."""
    # Run naive
    evapotranspiration_step(M_naive, ...)
    leakage_step(M_naive, ...)
    diffusion_step(M_naive, M_naive_new, ...)

    # Run fused
    soil_step_fused(M_fused, M_fused_new, ...)

    assert np.allclose(M_naive_new.to_numpy(), M_fused_new.to_numpy(), rtol=1e-5)
```

## When to Add Abstraction

Add abstraction only when you feel repeated pain:

- **Third time copying neighbor offsets** → Create `geometry.py`
- **Third validation bug from config** → Add `__post_init__` validation
- **Third buffer-swap bug** → Add `swap_buffers()` helper
- **Third conservation bug** → Add `check_conservation()` function

Don't add abstraction because "we might need it later."

## References

- `AGENTS.md` — Development conventions
- `ecohydro_spec.md` — Physical equations and units
- `docs/mass_conservation.md` — Conservation testing patterns
- Taichi documentation: https://docs.taichi-lang.org/
