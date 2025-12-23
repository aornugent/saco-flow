
import gc
import time
from dataclasses import dataclass

import taichi as ti

from benchmarks.harness import Benchmark
from src.simulation import Simulation, SimulationParams


@dataclass
class ScalingMetrics:
    grid_size: int
    n_cells: int
    simulated_years: float
    wall_time_s: float

    @property
    def years_per_minute(self) -> float:
        return self.simulated_years / (self.wall_time_s / 60.0)

    @property
    def megacells_per_second(self) -> float:
        return (self.n_cells * self.simulated_years * 365.0) / self.wall_time_s / 1e6


class ScalingBenchmark(Benchmark):
    """Benchmarks full simulation validation across grid sizes."""

    def run(self) -> list[ScalingMetrics]:
        results = []
        sizes = [1024, 2048, 5120, 10000]

        self.print_header("SCALING BENCHMARK (RTX 3090)")

        for n in sizes:
            try:
                metrics = self._run_single(n, years=1.0)
                results.append(metrics)
            except Exception as e:
                print(f"FAILED on {n}x{n}: {e}")
                import traceback
                traceback.print_exc()

        self._print_report(results)
        self.teardown()
        return results

    def _run_single(self, n: int, years: float) -> ScalingMetrics:
        print(f"\nBenchmarking {n}x{n} ({n**2/1e6:.1f} M cells)...")

        gc.collect()
        ti.sync()

        params = SimulationParams(n=n)
        sim = Simulation(params)
        sim.initialize(seed=42)

        # Warmup
        print("  Warming up JIT...", end=" ", flush=True)
        for _ in range(100):
            sim.step_soil(dt=0.1)
        for _ in range(10):
            sim.step_vegetation(dt=1.0)

        if self.profile:
            try:
                ti.profiler.clear_kernel_profiler_info()
            except Exception:
                pass

        ti.sync()
        print("Done.")

        # Measurement
        print(f"  Simulating {years} year(s)...", end=" ", flush=True)
        start_time = time.perf_counter()

        sim.run(years=years, check_mass_balance=False, verbose=False)

        ti.sync()
        end_time = time.perf_counter()
        print("Done.")

        return ScalingMetrics(
            grid_size=n,
            n_cells=n*n,
            simulated_years=years,
            wall_time_s=end_time - start_time
        )

    def _print_report(self, results: list[ScalingMetrics]):
        self.print_header("RESULTS SUMMARY")
        print(f"{'Grid':<10} {'Cells':<12} {'Time (s)':<12} {'Speed (Yr/Min)':<18} {'Throughput (MC/s)':<18}")
        print("-" * 80)

        for r in results:
            print(
                f"{r.grid_size:<10} "
                f"{r.n_cells/1e6:>6.1f}M      "
                f"{r.wall_time_s:>6.2f}      "
                f"{r.years_per_minute:>10.2f}        "
                f"{r.megacells_per_second:>10.2f}"
            )
        self.print_footer()
