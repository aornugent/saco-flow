# Performance Tuning & Benchmarks

This document details the performance characteristics of the Saco-Flow simulation on NVIDIA RTX 3090 hardware.

## Quick Summary

| Parameter | Recommendation | Notes |
|-----------|----------------|-------|
| **Block Dimension** | `1024` | Significantly faster than default for large grids. |
| **Grid Size** | **Neutral** | No significant advantage to Base-2 ($2^k$). Base-10 is equally performant. |
| **Peak Bandwidth** | ~760 GB/s | Achieved at ~4k x 4k resolution (~80% of theoretical peak). |

## Benchmark Results (RTX 3090)

Benchmarks run on `2025-12-23`.

### 1. Block Dimension Optimization
Testing `tunable_diffusion` (generic 5-point stencil) with various block sizes.

| Grid Size | Block 128 | Block 256 | Block 512 | Block 1024 |
|-----------|-----------|-----------|-----------|------------|
| 1k x 1k | 269 GB/s | 269 GB/s | 267 GB/s | 269 GB/s |
| 4k x 4k | 696 GB/s | 645 GB/s | 765 GB/s | **722 GB/s** |
| 8k x 8k | 495 GB/s | 546 GB/s | 572 GB/s | **681 GB/s** |
| 16k x 16k | 363 GB/s | 360 GB/s | 409 GB/s | **487 GB/s** |

**Conclusion:** Larger block sizes (512 or 1024) consistently outperform smaller ones for large grids. The `1024` block size maximized throughput for the largest grids tested.

### 2. Grid Size Alignment (Base-2 vs Base-10)
Testing whether power-of-two texture alignment offers benefits.

| Size | Type | Megacells/s | Bandwidth |
|------|------|-------------|-----------|
| 4000² | Base-10 | 60,209 | 722 GB/s |
| 4096² | Base-2 | 59,428 | 713 GB/s |
| 8000² | Base-10 | 56,760 | 681 GB/s |
| 8192² | Base-2 | 59,324 | 712 GB/s |
| 16000²| Base-10 | 40,587 | 487 GB/s |
| 16384²| Base-2 | 40,570 | 486 GB/s |

**Conclusion:** No consistent advantage for Base-2 grid sizes. In some cases (e.g., 4000 vs 4096), Base-10 was slightly faster, possibly due to avoidance of cache bank conflicts (partition camping) that can occur with power-of-two strides.

## Theoretical Analysis

### Memory Bandwidth Bound
The simulation is strictly **memory bandwidth bound**.
- Arithmetic Intensity: < 1 FLOP/byte
- Each cell update requires ~12-20 bytes of global memory access (Read M, Read Mask, Write M_new).
- Computing gradients adds minimal overhead compared to the memory cost.

### Optimality of Large Blocks (1024)
- **Latency Hiding:** Memory bound kernels benefit from having many active warps to hide global memory latency. A block size of 1024 ensures the maximum number of threads per block, allowing the scheduler to switch between warps effectively when waiting for memory.
- **Shared Memory:** With `ti.block_local`, the stencil uses shared memory. Larger blocks mean a larger shared memory tile, which increases the ratio of "interior" pixels to "halo" pixels, reducing the relative overhead of loading the halo.

### Resonance & Alignment
The dip in performance at specific sizes (e.g., specific Base-2 sizes) is likely due to **DRAM row conflicts** or **cache set collisions**, where strides of exactly $2^k$ cause all accesses to map to the same set of memory banks. Modern GPUs (like RTX 3090) have swizzling to mitigate this, but it can still appear. This explains why arbitrary Base-10 sizes often perform just as well or better.

## Recommendations for Production

1.  **Set `block_dim=1024`** for all stencil kernels.
2.  **Enable `ti.block_local`** for diffusion kernels to maximize effective bandwidth.
3.  **Do not constrain grid sizes** to powers of two. Use the resolution that makes physical sense for the domain.
