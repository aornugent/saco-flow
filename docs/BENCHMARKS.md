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
