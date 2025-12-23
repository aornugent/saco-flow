
import argparse
import sys
from typing import Type

from benchmarks.harness import Benchmark
from benchmarks.scaling import ScalingBenchmark
from benchmarks.kernels import KernelFusionBenchmark
from benchmarks.diffusion import DiffusionBenchmark

# Registry of available benchmarks
BENCHMARKS = {
    "scaling": ScalingBenchmark,
    "kernels": KernelFusionBenchmark,
    "diffusion": DiffusionBenchmark,
}

def main():
    parser = argparse.ArgumentParser(description="Saco-Flow Benchmark Harness")
    parser.add_argument(
        "benchmark", 
        nargs="?",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Benchmark to run (default: all)"
    )
    parser.add_argument(
        "--profile", 
        action="store_true", 
        help="Enable Taichi kernel profiler"
    )
    
    args = parser.parse_args()
    
    to_run = []
    if args.benchmark == "all":
        to_run = list(BENCHMARKS.values())
    else:
        to_run = [BENCHMARKS[args.benchmark]]
        
    for bench_cls in to_run:
        print(f"\nRunning {bench_cls.__name__}...")
        try:
            # Re-instantiate for each run to ensure clean Taichi state if possible
            # Note: Taichi init is global and sticky, so we just init once in the first one
            # or rely on the harness handling it. 
            # Ideally we'd reset the backend but ti.reset() isn't fully reliable.
            # We trust the harness logic.
            b = bench_cls(profile=args.profile)
            b.run()
        except Exception as e:
            print(f"Error running benchmark: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
