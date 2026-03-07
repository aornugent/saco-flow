"""Diffusion stencil benchmark — measures GPU throughput for the Laplacian kernel."""

import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from benchmarks.harness import Benchmark
from src.fields import allocate, swap_buffers
from src.kernels.diffusion import compute_stable_dt, diffusion_step


@dataclass
class DiffusionMetrics:
    grid_size: int
    n_steps: int
    total_time_s: float

    @property
    def time_per_step_ms(self) -> float:
        return (self.total_time_s / self.n_steps) * 1000

    @property
    def cells_per_second(self) -> float:
        return (self.grid_size**2 * self.n_steps) / self.total_time_s

    @property
    def bandwidth_gb_s(self) -> float:
        # ~44 bytes/cell/step: read 5 floats + mask, write 1 float
        return 44 * self.grid_size**2 * self.n_steps / self.total_time_s / 1e9


class DiffusionBenchmark(Benchmark):
    """Measures diffusion stencil throughput across grid sizes."""

    def run(self) -> list[DiffusionMetrics]:
        results = []
        grid_sizes = [256, 512, 1024, 2048, 4096]

        self.print_header("DIFFUSION STENCIL BENCHMARK")

        for n in grid_sizes:
            try:
                metrics = self._run_single(n, n_warmup=10, n_steps=100)
                results.append(metrics)
            except Exception as e:
                print(f"  {n}x{n} failed: {e}")

        self._print_report(results)
        self.teardown()
        return results

    def _run_single(self, n: int, n_warmup: int, n_steps: int) -> DiffusionMetrics:
        print(f"\n  {n}x{n} ({n**2 / 1e6:.1f}M cells)...", end=" ", flush=True)

        fields = allocate(n)

        # Random initial moisture
        fields.M.from_numpy(np.random.uniform(0.1, 0.3, (n, n)).astype(np.float32))

        # Interior mask (boundaries inactive)
        mask_np = np.ones((n, n), dtype=np.int32)
        mask_np[0, :] = mask_np[-1, :] = mask_np[:, 0] = mask_np[:, -1] = 0
        fields.mask.from_numpy(mask_np)

        D = 0.1  # m^2/day
        dx = 1.0  # m
        dt = compute_stable_dt(D, dx)

        # Warmup JIT
        for _ in range(n_warmup):
            diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
            swap_buffers(fields.M_new, fields.M)
        ti.sync()

        # Measure
        start = time.perf_counter()
        for _ in range(n_steps):
            diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
            swap_buffers(fields.M_new, fields.M)
        ti.sync()
        elapsed = time.perf_counter() - start

        print(f"{elapsed:.3f}s")
        return DiffusionMetrics(grid_size=n, n_steps=n_steps, total_time_s=elapsed)

    def _print_report(self, results: list[DiffusionMetrics]):
        self.print_header("RESULTS")
        fmt = "{:<12} {:<8} {:<12} {:<14} {:<14} {:<10}"
        print(
            fmt.format(
                "Grid", "Steps", "Total (s)", "Per Step (ms)", "Cells/s", "BW (GB/s)"
            )
        )
        print("-" * 80)
        for r in results:
            print(
                fmt.format(
                    f"{r.grid_size}x{r.grid_size}",
                    r.n_steps,
                    f"{r.total_time_s:.3f}",
                    f"{r.time_per_step_ms:.3f}",
                    f"{r.cells_per_second:.2e}",
                    f"{r.bandwidth_gb_s:.2f}",
                )
            )
        self.print_footer()
