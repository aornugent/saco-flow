# Architecture

Code organization for a ~2000 line GPU-accelerated simulation.

## Design Principles

1. **Simplicity first** - Flat structure, minimal indirection
2. **Correctness before speed** - Every kernel needs a conservation test
3. **Explicit over implicit** - Physical variable names, units in comments
4. **Taichi-idiomatic** - Leverage `ti.template()`, `ti.static()`, ping-pong buffers
5. **Test with physics** - Validate against analytical solutions, not just conservation

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

## Key Classes

### SimulationState

Bundles all runtime state (`src/simulation.py`):

```python
@dataclass
class SimulationState:
    fields: SimpleNamespace    # All Taichi fields
    mass_balance: MassBalance  # Cumulative flux tracker
    current_day: float = 0.0   # Simulation time
    dx: float = 1.0            # Cell size [m]

    def total_surface_water(self) -> float: ...
    def total_soil_moisture(self) -> float: ...
    def total_water(self) -> float: ...
    def max_surface_water(self) -> float: ...
```

### MassBalance

Tracks cumulative fluxes for conservation verification (`src/diagnostics.py`):

```python
@dataclass
class MassBalance:
    initial_water: float = 0.0       # h + M at start [m^3]
    cumulative_rain: float = 0.0     # Total rainfall [m^3]
    cumulative_et: float = 0.0       # Total evapotranspiration [m^3]
    cumulative_leakage: float = 0.0  # Total deep leakage [m^3]
    cumulative_outflow: float = 0.0  # Total boundary outflow [m^3]

    def expected_water(self) -> float:
        return initial + rain - et - leakage - outflow

    def check(self, actual, rtol=1e-4, atol=1e-8) -> float:
        # Raises AssertionError if conservation violated
        # Returns relative error
```

### SimulationParams

All parameters with validation (`src/params.py`):

```python
@dataclass
class SimulationParams:
    n: int = 64          # Grid size
    dx: float = 1.0      # Cell size [m]
    # ... all parameters with defaults

    def __post_init__(self):
        # Validates all parameter ranges
```

See `physics.md` for the complete parameter table.

## Data Structures

### Field Storage (Structure of Arrays)

All fields stored as separate contiguous arrays for coalesced GPU access:

```python
# State fields (double-buffered for stencil operations)
h = ti.field(dtype=ti.f32, shape=(n, n))      # Surface water [m]
M = ti.field(dtype=ti.f32, shape=(n, n))      # Soil moisture [m]
P = ti.field(dtype=ti.f32, shape=(n, n))      # Vegetation [kg/m^2]

# Static fields (read-only during simulation)
Z = ti.field(dtype=ti.f32, shape=(n, n))      # Elevation [m]
mask = ti.field(dtype=ti.i8, shape=(n, n))    # Domain mask
flow_frac = ti.field(dtype=ti.f32, shape=(n, n, 8))  # MFD fractions

# Intermediate fields
q_out = ti.field(dtype=ti.f32, shape=(n, n))  # Outflow rate [m/day]
```

**Why SoA?** Threads in a warp access consecutive memory locations when iterating over the same field.

### Memory Budget

| Grid Size | Cells | Memory | H100 (80GB) | B200 (192GB) |
|-----------|-------|--------|-------------|--------------|
| 1k x 1k | 10^6 | ~77 MB | Yes | Yes |
| 10k x 10k | 10^8 | ~7.7 GB | Yes | Yes |
| 20k x 20k | 4x10^8 | ~31 GB | Yes | Yes |
| 50k x 50k | 2.5x10^9 | ~193 GB | No | Yes |

At 10k x 10k, memory usage is <10% of H100's capacity.

## Buffer Strategy: Ping-Pong

### Why Ping-Pong

Stencil operations read neighbors while writing center. Without double-buffering, race conditions occur where a thread reads a neighbor that another thread has already updated.

### Pattern

```python
# Kernel reads from field, writes to field_new
laplacian_diffusion_step(fields.M, fields.M_new, ...)

# O(1) pointer swap (not O(n^2) copy)
swap_buffers(fields, "M")
# Now fields.M holds updated values
```

### Correctness Rules

**Rule 1: Point-wise operations can modify in-place**

Operations that only read/write the current cell (no neighbors):

```python
@ti.kernel
def et_leakage_step_fused(M: ti.template(), ...):
    for i, j in ti.ndrange(...):
        M[i, j] = M[i, j] - et - leakage  # In-place OK
```

Each cell is independent; update order doesn't matter.

**Rule 2: Stencil operations use double buffer**

Operations that read neighbors must read from one buffer and write to another:

```python
@ti.kernel
def laplacian_diffusion_step(M: ti.template(), M_new: ti.template(), ...):
    for i, j in ti.ndrange(...):
        laplacian = M[i-1,j] + M[i+1,j] + M[i,j-1] + M[i,j+1] - 4*M[i,j]
        M_new[i, j] = M[i, j] + D * dt / (dx*dx) * laplacian
```

**Rule 3: Stencil must be LAST before swap**

After stencil writes to M_new, the M buffer is stale:

```python
def soil_moisture_step(fields, ...):
    # 1. Point-wise: modify M in-place
    et_leakage_step_fused(fields.M, ...)

    # 2. Stencil: MUST be last, reads M, writes M_new
    laplacian_diffusion_step(fields.M, fields.M_new, ...)

    # 3. Swap: M_new becomes M
    swap_buffers(fields, "M")
```

### Surface Water: Two-Pass (Not Ping-Pong)

Surface routing uses a different pattern:

```python
def route_surface_water(h, q_out, ...):
    # Pass 1: compute outflow rates (reads h, writes q_out)
    compute_outflow(h, ..., q_out, ...)

    # Pass 2: apply fluxes (reads q_out from neighbors, updates h)
    apply_fluxes(h, q_out, ...)
```

**Why this works:** Pass 2 reads q_out[neighbors], not h[neighbors]. No race condition because the stencil reads from a separately-computed field.

### Buffer Pattern Summary

| Operation Type | Buffer Pattern | Examples |
|----------------|----------------|----------|
| Point-wise | In-place on current | ET, leakage, growth, mortality |
| Stencil (diffusion) | current -> next (ping-pong) | Soil/vegetation diffusion |
| Two-pass routing | h + intermediate q_out | Surface water routing |

## CRITICAL: ti.template() vs Closure Capture

**Always pass fields as ti.template() arguments. Never capture fields in closures.**

```python
# WRONG: Closure-captured field (baked in at compile time)
M = ti.field(ti.f32, shape=(n, n))

@ti.kernel
def bad_kernel():
    for i, j in M:  # M captured at JIT compile time
        ...

# After swap_buffers(), this kernel STILL uses the original M!

# CORRECT: Pass fields as arguments
@ti.kernel
def good_kernel(M: ti.template(), M_new: ti.template()):
    for i, j in M:
        M_new[i, j] = ...
```

**Why:** Taichi JIT-compiles kernels on first call. Fields in the kernel body are "baked in" at compile time. Swapping Python references doesn't affect the compiled kernel.

**Rule:** If a field will be swapped via `swap_buffers()`, it MUST be passed as `ti.template()`.

## Fused vs Naive Kernels

Both fused and naive versions are kept in the codebase:

**Fused kernels** (default): Combine point-wise operations to reduce memory traffic.

```python
# Single pass: 2x less memory traffic
def et_leakage_step_fused(M, P, mask, ...):
    # Reads M, P once; writes M once
```

**Naive kernels** (for testing): Separate kernels for regression testing.

```python
# Two passes: easier to verify individually
def evapotranspiration_step(M, P, mask, ...): ...
def leakage_step(M, mask, ...): ...
```

The naive versions ensure the fused versions produce identical results.

## GPU Optimization

### Bottleneck: Memory Bandwidth

The simulation is memory-bandwidth bound, not compute bound:
- Arithmetic intensity: ~0.2-0.5 FLOP/byte
- Each cell: ~10-20 FLOPs but reads/writes ~40-100 bytes

### Kernel Fusion Impact

Before (naive):
```
ET:        Read M,P -> Write M
Leakage:   Read M   -> Write M
Diffusion: Read M   -> Write M_new
Total: 8 reads + 4 writes = 48 bytes/cell
```

After (fused ET+leakage):
```
Fused:     Read M,P -> Write M
Diffusion: Read M   -> Write M_new
Total: 4 reads + 2 writes = 24 bytes/cell (2x improvement)
```

### Shared Memory for Stencils

`ti.block_local()` caches stencil reads in GPU shared memory:

```python
@ti.kernel
def laplacian_diffusion_step(field: ti.template(), ...):
    ti.block_local(field)  # Cache in shared memory

    for i, j in ti.ndrange(...):
        # Neighbor reads benefit from cache
        laplacian = field[i-1,j] + field[i+1,j] + ...
```

### Coalesced Memory Access

Threads in a warp should access contiguous memory:

```python
# Correct: j innermost (row-major, coalesced)
for i, j in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...

# Wrong: strided access
for j, i in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...  # 4*n byte stride between threads
```

### Target Hardware

| GPU | Architecture | HBM | Bandwidth | FP32 TFLOPS |
|-----|--------------|-----|-----------|-------------|
| H100 | Hopper (sm_90) | 80GB HBM3 | 3.35 TB/s | ~1979 |
| B200 | Blackwell (sm_100) | 192GB HBM3e | 8 TB/s | ~2500 |

**Target:** 10k x 10k grid at >=1 simulated year per wall-clock minute.

See `BENCHMARKS.md` for measured performance.

### Profiling

```python
ti.init(arch=ti.cuda, kernel_profiler=True)
# ... run simulation ...
ti.profiler.print_kernel_profiler_info()
```

## When to Add Abstraction

Add abstraction only when you feel repeated pain:

- **Third time copying neighbor offsets** -> Create `geometry.py`
- **Third validation bug from config** -> Add `__post_init__` validation
- **Third buffer-swap bug** -> Add `swap_buffers()` helper
- **Third conservation bug** -> Add `check_conservation()` function

Don't add abstraction because "we might need it later."
