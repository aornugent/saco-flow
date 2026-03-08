"""Diffusion stencil benchmark: python bench.py"""

import time

import numpy as np
import taichi as ti

from src.diffusion import compute_stable_dt, diffusion_step
from src.fields import allocate, swap_buffers

GRID_SIZES = [256, 512, 1024, 2048, 4096]
N_WARMUP = 10
N_STEPS = 100


def bench_grid(n: int) -> dict:
    """Benchmark diffusion on an n x n grid."""
    fields = allocate(n)
    fields.M.from_numpy(np.random.uniform(0.1, 0.3, (n, n)).astype(np.float32))

    mask_np = np.ones((n, n), dtype=np.int32)
    mask_np[0, :] = mask_np[-1, :] = mask_np[:, 0] = mask_np[:, -1] = 0
    fields.mask.from_numpy(mask_np)

    D, dx = 0.1, 1.0  # m^2/day, m
    dt = compute_stable_dt(D, dx)

    for _ in range(N_WARMUP):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)
    ti.sync()

    start = time.perf_counter()
    for _ in range(N_STEPS):
        diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        swap_buffers(fields.M_new, fields.M)
    ti.sync()
    elapsed = time.perf_counter() - start

    cells = n * n * N_STEPS
    return {
        "grid": f"{n}x{n}",
        "total_s": elapsed,
        "ms_per_step": elapsed / N_STEPS * 1000,
        "cells_per_s": cells / elapsed,
        "bw_gb_s": 44 * cells / elapsed / 1e9,
    }


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    results = []
    for n in GRID_SIZES:
        print(f"  {n}x{n} ({n**2 / 1e6:.1f}M cells)...", end=" ", flush=True)
        try:
            r = bench_grid(n)
            results.append(r)
            print(f"{r['total_s']:.3f}s")
        except Exception as e:
            print(f"failed: {e}")

    fmt = "{:<12} {:<12} {:<14} {:<14} {:<10}"
    print(fmt.format("Grid", "Total (s)", "Per Step (ms)", "Cells/s", "BW (GB/s)"))
    print("-" * 62)
    for r in results:
        print(
            fmt.format(
                r["grid"],
                f"{r['total_s']:.3f}",
                f"{r['ms_per_step']:.3f}",
                f"{r['cells_per_s']:.2e}",
                f"{r['bw_gb_s']:.2f}",
            )
        )
