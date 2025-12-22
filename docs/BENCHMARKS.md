# Benchmarks

## RTX 3090 (24GB)

**Date:** 2025-12-22
**Driver:** CUDA (Taichi v1.7.4)
**Precision:** f32

### Summary
Scale verification successful. The simulation runs stably at 10k x 10k resolution (100 million cells) on a single RTX 3090.

| Grid Size | Cells | Wall Time (1 year) | Speed (Model Years / Wall Minute) | Throughput (MegaCells / s) |
|-----------|-------|--------------------|-----------------------------------|----------------------------|
| 1,024² | 1.0M | 1.83 s | 32.83 | 209 |
| 2,048² | 4.2M | 3.70 s | 16.21 | 414 |
| 5,120² | 26.2M | 32.55 s | 1.84 | 294 |
| 10,000² | 100.0M | 97.53 s | 0.62 | 374 |

### Scaling Analysis

The "Throughput" metric here is defined as **Simulated Cell-Years per Second**. Variability across grid sizes is expected due to two competing factors:

1.  **GPU Saturation (Increases Throughput):**
    - **1k (1M cells):** The grid is too small to saturate the RTX 3090 (which has ~10k cores). Kernel launch overheads are significant relative to computation, leading to low efficiency (209 MC/s).
    - **2k+ (4M+ cells):** The GPU is better saturated, amortizing overheads and doubling efficiency (414 MC/s).

2.  **Adaptive Timestepping (Decreases Throughput):**
    - **Physics:** Larger grids collect more water downslope, creating deeper "rivers".
    - **CFL Condition:** Deeper water flows faster ($v \propto h^{2/3}$). To maintain numerical stability, the solver must reduce the timestep ($dt \le dx/v$).
    - **Result:** Larger grids require **more computational steps** to simulate the same amount of physical time. This lowers the "Simulated Years per Minute" metric even if the hardware is running at peak theoretical FLOPs/Bandwidth.

### Notes
- **Memory**: 10k x 10k fits comfortably within 24GB VRAM.
- **Stability**: No OOM or numerical instabilities observed during 1-year run.

### Kernel Fusion Analysis

Comparison of fused (combined point-wise ops) vs naive (separate kernels) implementations reveals significant bandwidth savings.

| Component | Operation | Speedup |
|-----------|-----------|---------|
| Soil Moisture | ET + Leakage + Diffusion | **1.30x** |
| Vegetation | Growth + Mortality + Diffusion | **1.54x** |

Vegetation shows higher speedup because the fused kernel avoids loading/storing `P` and `M` multiple times for growth and mortality steps.
