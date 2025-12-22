"""
Systematic benchmark harness for saco-flow.

Measures performance across multiple grid sizes to quantify:
1. Throughput (Megacells/second)
2. Simulation speed (Simulated years / Wall minute)
3. Scaling efficiency
"""

import time
import gc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import taichi as ti

from src.simulation import Simulation, SimulationParams
from src.config import init_taichi


@dataclass
class BenchmarkMetrics:
    grid_size: int
    n_cells: int
    simulated_years: float
    wall_time_s: float
    
    @property
    def years_per_minute(self) -> float:
        return self.simulated_years / (self.wall_time_s / 60.0)
    
    @property
    def megacells_per_second(self) -> float:
        # Total cell updates = (steps_soil + steps_veg) * n_cells
        # This is harder to count exactly without instrumentation, 
        # so we use a simplified "state updates" metric or just raw cell-years/second
        # For comparison, we'll use (n_cells * simulated_days) / wall_seconds
        # This is a proxy for total work.
        return (self.n_cells * self.simulated_years * 365.0) / self.wall_time_s / 1e6


class BenchmarkRunner:
    def __init__(self):
        print("Initializing Benchmark Runner...")
        # Initialize Taichi once
        try:
            init_taichi(backend="cuda", debug=False)
        except Exception as e:
            print(f"Warning: CUDA init failed ({e}), falling back to default")
            init_taichi(debug=False)

    def run_single(self, n: int, years: float = 1.0) -> BenchmarkMetrics:
        """Run benchmark for a single grid size."""
        print(f"\nBenchmarking {n}x{n} ({n**2/1e6:.1f} M cells)...")
        
        # Cleanup
        gc.collect()
        ti.sync()
        
        # Setup
        params = SimulationParams(n=n)
        # Fix seed for reproducibility
        sim = Simulation(params)
        sim.initialize(seed=42)
        
        # Warmup
        print("  Warming up JIT...", end=" ", flush=True)
        for _ in range(100):
            sim.step_soil(dt=0.1)
        for _ in range(10):
            sim.step_vegetation(dt=1.0)
        ti.sync()
        print("Done.")
        
        # Measurement
        print(f"  Simulating {years} year(s)...", end=" ", flush=True)
        start_time = time.perf_counter()
        
        sim.run(years=years, check_mass_balance=False, verbose=False)
        
        ti.sync()
        end_time = time.perf_counter()
        print("Done.")
        
        wall_time = end_time - start_time
        
        return BenchmarkMetrics(
            grid_size=n,
            n_cells=n*n,
            simulated_years=years,
            wall_time_s=wall_time
        )

    def print_report(self, results: List[BenchmarkMetrics]):
        print("\n" + "="*80)
        print(f"{'BENCHMARK RESULTS (RTX 3090)':^80}")
        print("="*80)
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
        print("="*80)

def main():
    runner = BenchmarkRunner()
    
    # Grid sizes to test
    # 1k, 2k, 5k, 10k
    sizes = [1024, 2048, 5120, 10000]
    
    results = []
    
    for n in sizes:
        try:
            # We can use valid years=0.1 for quick testing of larger grids if needed,
            # but for 10k we want full 1.0 year to smooth out variance
            years = 1.0
            metrics = runner.run_single(n, years=years)
            results.append(metrics)
        except Exception as e:
            print(f"FAILED on {n}x{n}: {e}")
            import traceback
            traceback.print_exc()
            
    runner.print_report(results)

if __name__ == "__main__":
    main()
