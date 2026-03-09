"""Baartman daily-step benchmark: python bench.py"""

import time

import numpy as np
import taichi as ti

from src.fields import allocate
from src.flow import compute_flow_fractions
from src.simulate import step_day

GRID_SIZES = [64, 128, 256, 512, 1024]
N_WARMUP = 3
N_STEPS = 20

_DAILY = {
    "dx": 1.0,
    "n_manning": 0.03,
    "cn": 1.0,
    "alpha": 1.0,
    "k2": 5.0,
    "W0": 0.2,
    "g_max": 0.1,
    "k1": 0.1,
    "rw": 0.01,
    "c": 1.0,
    "d": 0.01,
    "Dp": 0.01,
    "c1": 0.01,
    "c2": 1.0,
    "dt": 1.0,
    "n_picard": 10,
}


def setup_grid(n: int):
    """Allocate fields with boundary mask, slope, rainfall, and initial conditions."""
    fields = allocate(n)
    mask = np.ones((n, n), dtype=np.int32)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)

    z = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        z[i, :] = float(n - i)
    fields.z.from_numpy(z)

    fields.R.from_numpy(np.full((n, n), 0.01, dtype=np.float32))
    fields.V.from_numpy(np.full((n, n), 5.0, dtype=np.float32))
    fields.M.from_numpy(np.full((n, n), 0.1, dtype=np.float32))

    compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, 1.0, 1.1)
    return fields


def bench_grid(n: int) -> dict:
    """Benchmark step_day on an n x n grid."""
    fields = setup_grid(n)

    for _ in range(N_WARMUP):
        step_day(fields, **_DAILY)
    ti.sync()

    start = time.perf_counter()
    for _ in range(N_STEPS):
        step_day(fields, **_DAILY)
    ti.sync()
    elapsed = time.perf_counter() - start

    cells = n * n * N_STEPS
    return {
        "grid": f"{n}x{n}",
        "total_s": elapsed,
        "ms_per_step": elapsed / N_STEPS * 1000,
        "cells_per_s": cells / elapsed,
    }


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f32)

    results = []
    for n in GRID_SIZES:
        print(f"  {n}x{n} ({n**2 / 1e6:.2f}M cells)...", end=" ", flush=True)
        try:
            r = bench_grid(n)
            results.append(r)
            print(f"{r['total_s']:.3f}s")
        except Exception as e:
            print(f"failed: {e}")

    fmt = "{:<12} {:<12} {:<14} {:<14}"
    print(fmt.format("Grid", "Total (s)", "Per Step (ms)", "Cells/s"))
    print("-" * 52)
    for r in results:
        print(
            fmt.format(
                r["grid"],
                f"{r['total_s']:.3f}",
                f"{r['ms_per_step']:.3f}",
                f"{r['cells_per_s']:.2e}",
            )
        )
