
import time
from dataclasses import dataclass

import numpy as np
import taichi as ti

from benchmarks.harness import Benchmark
from src.fields import allocate
from src.geometry import laplacian_diffusion_step
from src.kernels.soil import compute_diffusion_timestep


@dataclass
class DiffusionMetrics:
    grid_size: int
    n_steps: int
    total_time_s: float
    time_per_step_ms: float
    cells_per_second: float
    bandwidth_gb_s: float

class DiffusionBenchmark(Benchmark):
    """Benchmarks diffusion stencil performance (blocks Shared Memory usage)."""

    def run(self) -> list[DiffusionMetrics]:
        results = []

        # Grid sizes to test
        grid_sizes = [256, 512, 1024, 2048]

        self.print_header("DIFFUSION BENCHMARK")

        for n in grid_sizes:
            print(f"\nBenchmarking {n}x{n} grid...")
            try:
                metrics = self._run_single(n, n_warmup=10, n_steps=100)
                results.append(metrics)
            except Exception as e:
                print(f"  Failed: {e}")

        self._print_report(results)
        self.teardown()
        return results

    def _run_single(self, n: int, n_warmup: int, n_steps: int) -> DiffusionMetrics:
        fields = allocate(n)

        # Init random
        M_np = np.random.uniform(0.1, 0.3, (n, n)).astype(np.float32)
        fields.M.from_numpy(M_np)

        # Mask interior
        mask_np = np.ones((n, n), dtype=np.int32)
        mask_np[0, :] = mask_np[-1, :] = mask_np[:, 0] = mask_np[:, -1] = 0
        fields.mask.from_numpy(mask_np)

        D = 0.1
        dx = 1.0
        dt = compute_diffusion_timestep(D, dx, cfl=0.2)

        # Warmup
        print(f"  Warming up ({n_warmup} steps)...", end=" ", flush=True)
        for _ in range(n_warmup):
            laplacian_diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
            fields.M.from_numpy(fields.M_new.to_numpy())
        print("done")

        ti.sync()

        # Measure
        print(f"  Measuring ({n_steps} steps)...", end=" ", flush=True)
        start = time.perf_counter()

        for _ in range(n_steps):
            laplacian_diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
            fields.M.from_numpy(fields.M_new.to_numpy())

        ti.sync()
        elapsed = time.perf_counter() - start
        print("done")

        n_cells = n * n
        time_per_step_ms = (elapsed / n_steps) * 1000
        cells_per_second = (n_cells * n_steps) / elapsed

        # Bandwidth estimate: ~44 bytes/cell/step
        bytes_per_cell = 44
        total_bytes = bytes_per_cell * n_cells * n_steps
        bandwidth_gb_s = total_bytes / elapsed / 1e9

        return DiffusionMetrics(
            grid_size=n,
            n_steps=n_steps,
            total_time_s=elapsed,
            time_per_step_ms=time_per_step_ms,
            cells_per_second=cells_per_second,
            bandwidth_gb_s=bandwidth_gb_s,
        )

    def _print_report(self, results: list[DiffusionMetrics]):
        self.print_header("DIFFUSION RESULTS")
        print(f"{'Grid':<10} {'Steps':<8} {'Total (s)':<12} {'Per Step (ms)':<14} {'Cells/s':<12} {'BW (GB/s)':<10}")
        print("-" * 80)

        for r in results:
            print(
                f"{r.grid_size:>4}x{r.grid_size:<5} "
                f"{r.n_steps:<8} "
                f"{r.total_time_s:<12.3f} "
                f"{r.time_per_step_ms:<14.3f} "
                f"{r.cells_per_second:<12.2e} "
                f"{r.bandwidth_gb_s:<10.2f}"
            )
        self.print_footer()
