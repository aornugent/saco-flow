# GPU Optimization Guide

This document describes the optimization strategy for achieving high throughput on NVIDIA B200 and H100 GPUs.

## Target Hardware

| GPU | Architecture | HBM | Bandwidth | FP32 TFLOPS |
|-----|--------------|-----|-----------|-------------|
| B200 | Blackwell (sm_100) | 192GB HBM3e | 8 TB/s | ~2500 |
| H100 | Hopper (sm_90) | 80GB HBM3 | 3.35 TB/s | ~1979 |

## Performance Target

- **Grid size:** 10,000 × 10,000 cells (10⁸ cells)
- **Throughput:** ≥1 simulated year per wall-clock minute
- **Bandwidth efficiency:** >50% of theoretical peak

## Bottleneck Analysis

The simulation is **memory-bandwidth bound**, not compute bound. Evidence:

1. **Arithmetic intensity is low:** Each cell update involves ~10-20 FLOPs but reads/writes ~40-100 bytes
2. **Stencil operations dominate:** Diffusion requires reading 5 neighbors per cell
3. **Multiple kernel passes:** Naive implementation reads/writes fields multiple times per timestep

**Roofline analysis (H100):**
- Memory bandwidth: 3.35 TB/s
- Compute: 1979 TFLOPS
- Crossover at ~0.6 FLOP/byte

Our kernels have ~0.2-0.5 FLOP/byte → memory bound.

## Optimization Strategy

### 1. Kernel Fusion

**Goal:** Reduce global memory traffic by combining operations.

**Before (naive):**
```
soil_moisture_step:
  1. evapotranspiration_step: Read M,P → Write M
  2. leakage_step: Read M → Write M
  3. diffusion_step: Read M → Write M_new
  4. copy_field: Read M_new → Write M
Total: 8 reads + 4 writes per cell = 48 bytes/cell
```

**After (fused):**
```
soil_update_fused:
  1. Read M, P once → Compute all → Write M_new once
Total: 2 reads + 1 write per cell = 12 bytes/cell
```

**Expected speedup:** 4× for soil update (48/12 = 4)

### 2. Shared Memory (ti.block_local)

For stencil operations, use Taichi's `ti.block_local()` to cache data in shared memory:

```python
@ti.kernel
def diffusion_with_cache(M: ti.template(), M_new: ti.template(), ...):
    ti.block_local(M)  # Declares M should be cached in shared memory

    for i, j in ti.ndrange((1, n-1), (1, n-1)):
        # Stencil reads benefit from shared memory cache
        laplacian = M[i-1,j] + M[i+1,j] + M[i,j-1] + M[i,j+1] - 4*M[i,j]
        ...
```

**Shared memory budget (per block):**
- H100/B200: 48KB-228KB configurable
- Default: 48KB
- For 32×32 block of f32: 32×32×4 = 4KB + halo = ~5KB ✓

### 3. Coalesced Memory Access

GPUs achieve peak bandwidth only when threads in a warp access contiguous memory.

**Taichi fields are row-major:** `field[i, j]` stores row `i` contiguously.

**Correct loop order:**
```python
# Threads in warp access consecutive j values (coalesced)
for i, j in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...
```

**Incorrect loop order:**
```python
# Threads in warp access strided i values (not coalesced)
for j, i in ti.ndrange((1, n-1), (1, n-1)):
    field[i, j] = ...  # BAD: 4*n byte stride
```

### 4. Temporal Blocking (Advanced)

For pure diffusion, multiple timesteps can be batched in shared memory:

```python
@ti.kernel
def diffusion_temporal(M: ti.template(), steps: int, dt: DTYPE, ...):
    ti.block_local(M)

    for i, j in block_interior:
        local = M[i, j]
        for _ in range(steps):
            # Read neighbors from shared memory
            laplacian = ...
            local += D * laplacian * dt
            ti.simt.block.sync()  # Synchronize block
        M[i, j] = local
```

**Bandwidth reduction:** Factor of `steps` (e.g., 4-8 steps → 4-8× reduction)

**Limitation:** Only works when same operation repeats; not compatible with fused ET/leakage.

### 5. Reducing Atomic Contention

Mass balance tracking uses atomic operations. Strategies:

1. **Thread-local accumulation:** Each thread accumulates locally, atomic once at end
2. **Warp reduction:** Use `ti.simt.warp.reduce_add()` before atomic
3. **Block reduction:** Use shared memory for block-level reduction

```python
# Naive (many atomics):
for i, j in domain:
    ti.atomic_add(total, value)

# Better (one atomic per thread):
local_sum = 0.0
for i, j in domain:
    local_sum += value
ti.atomic_add(total, local_sum)
```

## Memory Budget

| Grid Size | Cells | Memory | H100 | B200 |
|-----------|-------|--------|------|------|
| 1k × 1k | 10⁶ | ~77 MB | ✓ | ✓ |
| 5k × 5k | 2.5×10⁷ | ~1.9 GB | ✓ | ✓ |
| 10k × 10k | 10⁸ | ~7.7 GB | ✓ | ✓ |
| 20k × 20k | 4×10⁸ | ~31 GB | ✓ | ✓ |
| 50k × 50k | 2.5×10⁹ | ~193 GB | ✗ | ✓ |

## Benchmarking Protocol

1. **Warmup:** Run 10 timesteps to trigger JIT compilation
2. **Measurement:** Time 100+ vegetation timesteps (700+ simulated days)
3. **Metrics:**
   - Wall time per simulated year
   - Cells updated per second
   - Achieved bandwidth (bytes read+written / time)
4. **Grid sizes:** 1k, 5k, 10k (and 20k if hardware permits)

## Profiling with Taichi

Enable kernel profiler:
```python
ti.init(arch=ti.cuda, kernel_profiler=True)

# ... run simulation ...

ti.profiler.print_kernel_profiler_info()
```

Key metrics:
- **Time per kernel:** Identify hot kernels
- **Occupancy:** Check register/shared memory pressure
- **Memory throughput:** Compare to theoretical peak

## GPU Gotchas

- **Memory layout:** Taichi uses row-major; iterate `(i, j)` with j innermost
- **Block size:** Use `ti.block_dim(256)` or `ti.block_dim(512)` for compute kernels
- **Shared memory limits:** 48KB per block default on H100/B200
- **Register pressure:** Fused kernels use more registers; watch occupancy
- **Atomic contention:** Global atomics are slow; use reduction patterns
- **Kernel launch overhead:** ~10μs per launch; fuse small kernels

## Optimization Checklist

- [ ] Loop order verified (j innermost for row-major)
- [ ] Block dimensions specified (`ti.block_dim()`)
- [ ] Shared memory used for stencil operations
- [ ] Kernel fusion implemented (soil, vegetation)
- [ ] Atomic operations minimized
- [ ] Benchmark results documented

## Results

*To be filled in as optimization proceeds.*

| Kernel | Naive (ms) | Optimized (ms) | Speedup |
|--------|------------|----------------|---------|
| soil_moisture_step | TBD | TBD | TBD |
| vegetation_step | TBD | TBD | TBD |
| route_surface_water | TBD | TBD | TBD |
| **Full year (10k×10k)** | TBD | TBD | TBD |

## References

- `ecohydro_spec.md:609-714` — Performance optimization section
- `IMPLEMENTATION_PLAN.md` — Phase 6 detailed tasks
- Taichi documentation: https://docs.taichi-lang.org/
