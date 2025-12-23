# Benchmarks

## RTX 3090 (24GB)

**Date:** 2025-12-23
**Driver:** CUDA (Taichi v1.7.4)
**Precision:** f32

### Summary
Scale verification successful. The simulation runs stably at 10k x 10k resolution (100 million cells) on a single RTX 3090.

| Grid Size | Cells | Wall Time (1 year) | Speed (Model Years / Wall Minute) | Throughput (MegaCells / s) |
|-----------|-------|--------------------|-----------------------------------|----------------------------|
| 1,000² | 1.0M | 1.93 s | 31.02 | 188 |
| 2,000² | 4.0M | 2.90 s | 20.68 | 503 |
| 5,000² | 25.0M | 24.29 s | 2.47 | 376 |
| 10,000² | 100.0M | 95.62 s | 0.63 | 382 |

### Throughput Analysis
Throughput (MegaCells/second) peaks at **503 MC/s** for the 2k grid (up from 409 MC/s), demonstrating the effectiveness of the `block_dim=1024` optimization. Performance stabilizes around 382 MC/s for the 10k grid, which is likely memory bandwidth bound by the surface routing kernels.

## Phase 8: Comprehensive Profiling Analysis

Detailed profiling was conducted across three dimensions: Scaling (Full App), Kernel Fusion, and Diffusion Stencils.

### 1. Application Bottlenecks (Scaling Benchmark)
**Scenario:** 10k x 10k simulation (Full Physics)
**Top Kernels by Frame Time:**

| Kernel | Function | % Time | Notes |
|--------|----------|--------|-------|
| `apply_fluxes` | Surface Routing (Pass 2) | **~50%** | **CRITICAL BOTTLENECK** |
| `compute_max_velocity` | CFL Timestep | ~13% | Reduction operation |
| `compute_outflow` | Surface Routing (Pass 1) | ~12% | Kinematic wave |
| `laplacian_diffusion_step` | Diffusion (Soil+Veg) | ~6% | Optimized stencil |
| `infiltration_step` | Infiltration | ~3% | |
| `et_leakage_step_fused` | Soil Physics | ~2% | Fused |

**Findings:**
- **Routing Dominated:** Over 75% of the runtime is spent in surface routing (`apply_fluxes`, `compute_outflow`, `compute_max_velocity`).
- **Memory Bound Routing:** `apply_fluxes` is likely memory bandwidth bound due to uncoalesced indirect access (MFD neighbors) and the two-pass logic (gathering inflow + checking boundary outflow).
- **Physics is Cheap:** Complex point-wise physics (ET, Growth, Mortality) account for < 5% of runtime thanks to kernel fusion.

### 2. Kernel Fusion Efficiency
Comparing "Fused" (point-wise operations combined) vs "Naive" (separate kernels).

| Component | Operation | Speedup | Notes |
|-----------|-----------|---------|-------|
| Soil Moisture | ET + Leakage + Diffusion | **1.32x** | Saves 2x memory roundtrips |
| Vegetation | Growth + Mortality + Diffusion | **1.06x** | Smaller gain (simpler kernels) |

**Conclusion:** Fusion is highly effective for bandwidth-bound point-wise operations.

### 3. Diffusion Stencil Optimization
Comparing 5-point Laplacian stencil performance.

| Grid | Bandwidth (GB/s) | Notes |
|------|------------------|-------|
| 1k | 18.6 GB/s | |
| 2k | 17.8 GB/s | |

**Observation:** The bandwidth numbers are lower than the theoretical limit (936 GB/s for RTX 3090). This indicates that while `ti.block_local()` is used, the memory access pattern or the benchmark harness itself (copying data to CPU for verification) may be limiting the measured throughput.

## Optimization Strategy (Future Work)

1.  **Refactor Routing:** The `apply_fluxes` kernel (50% runtime) is the primary target.
    - **Strategy:** Merge `compute_outflow` and `apply_fluxes` logic where possible, or optimize the neighbor gathering loop to reduce global memory transactions.
    - **Algorithmic:** The current MFD implementation checks 8 neighbors. Switching to D8 (single flow direction) for steeper slopes could reduce memory traffic by 8x for those cells.

2.  **CFL Optimization:** `compute_max_velocity` takes 13%.
    - **Strategy:** This is a global reduction. Lowering the frequency of CFL checking (e.g., adaptive update every N steps) could reclaim this time.
