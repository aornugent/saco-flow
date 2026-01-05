# Saco-Flow Benchmarks

## Overview
Performance benchmarks for `saco-flow` running on NVIDIA RTX 3090.
Simulations were run with `ti.init(arch=ti.gpu)` (CUDA).

## Scaling Performance
Ideally, we want to see high throughput (Megacells/s) that scales well with grid size.

**Date:** 2025-12-23
**Hardware:** NVIDIA RTX 3090
**Version:** Optimized kernels (fused flow routing, scatter-based flux, GPU reductions)

| Grid Size | Total Cells | Wall Time (1 Sim Year) | Simulation Speed | Throughput |
|-----------|-------------|------------------------|------------------|------------|
| 1000x1000 | 1.0 M | 1.57 s | 38.2 Years/Min | 232 MC/s |
| 2000x2000 | 4.0 M | 1.69 s | 35.6 Years/Min | 866 MC/s |
| 5000x5000 | 25.0 M | 2.86 s | 21.0 Years/Min | 3196 MC/s |
| 10000x10000 | 100.0 M | 6.49 s | 9.3 Years/Min | 5627 MC/s |

### Analysis
- **Throughput Scaling:** Throughput improves dramatically as grid size increases, indicating that for smaller grids, the simulation is dominated by kernel launch overhead and Python control flow.
- **Saturation:** At 10k x 10k (100M cells), we achieve **5.6 Gigacells/second**. This suggests the GPU is being utilized very effectively for the massive parallelism available.
- **Overhead:** The 1k grid takes 1.57s while the 10k grid (100x more work) takes only 6.49s (4x more time). This confirms that "fixed costs" (Python overhead, kernel launches) are amortized effectively at scale.

## Methodology
Benchmarks run via `benchmarks/scaling.py`.
- **Warmup:** JIT compilation and caches warmed up with dummy steps.
- **Timing:** Wall-clock time recorded includes all simulation steps (soil, vegetation, surface flow) but excludes initialization.
- **Metric:**

    - `MC/s` = (Total Cells * Simulated Years * 365) / Wall Time / 1e6

## F64 Performance (2025-12-24)
Switching to `ti.f64` (double precision) results in lower throughput on the RTX 3090, which has limited FP64 cores (1:64 or 1:32 ratio usually).

**Date:** 2025-12-24
**Hardware:** NVIDIA RTX 3090
**Precision:** F64 (Double)

| Grid Size | Total Cells | Wall Time (1 Sim Year) | Simulation Speed | Throughput | Change vs F32 |
|-----------|-------------|------------------------|------------------|------------|---------------|
| 1000x1000 | 1.0 M | 7.50 s | 8.00 Years/Min | 48.7 MC/s | ~0.2x |
| 2000x2000 | 4.0 M | 12.26 s | 4.89 Years/Min | 119.1 MC/s | ~0.14x |
| 5000x5000 | 25.0 M | 21.85 s | 2.75 Years/Min | 417.6 MC/s | ~0.13x |
| 10000x10000 | 100.0 M | 42.29 s | 1.42 Years/Min | 863.2 MC/s | ~0.15x |


### Observations
- **Significant Slowdown:** Performance dropped by a factor of 5-7x compared to F32. This is expected on consumer hardware but ensures numerical stability for long-term simulations.
- **Scaling:** Scaling characteristics remain similar (throughput increases with grid size), but the ceiling is much lower.


## Mixed Precision Strategy (2025-12-24)
We adopted a mixed precision strategy: `ti.f32` for simulation state fields (velocity, discharge, etc.) but explicitly casting to `ti.f64` for global mass balance diagnostics. This allows us to maintain the high performance of F32 while ensuring accurate conservation checks.

**Date:** 2025-12-24
**Hardware:** NVIDIA RTX 3090
**Precision:** Mixed (F32 Sim + F64 Diagnostics)

| Grid Size | Total Cells | Wall Time (1 Sim Year) | Simulation Speed | Throughput | Change vs Pure F64 |
|-----------|-------------|------------------------|------------------|------------|--------------------|
| 1000x1000 | 1.0 M | 1.21 s | 49.75 Years/Min | 302.6 MC/s | ~6.2x Faster |
| 2000x2000 | 4.0 M | 1.25 s | 47.89 Years/Min | 1165.4 MC/s | ~9.8x Faster |
| 5000x5000 | 25.0 M | 2.51 s | 23.94 Years/Min | 3641.0 MC/s | ~8.7x Faster |
| 10000x10000 | 100.0 M | 6.63 s | 9.05 Years/Min | 5503.9 MC/s | ~6.4x Faster |

### Outcome
Performance is fully restored to original F32 levels (actually slightly faster due to variability/driver updates). We now achieve **5.5 Gigacells/second** throughput on large grids while maintaining the ability to detect small mass balance errors via F64 reduction accumulators.

