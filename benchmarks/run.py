"""Benchmark runner CLI."""

import argparse
import traceback

from benchmarks.diffusion import DiffusionBenchmark

BENCHMARKS = {
    "diffusion": DiffusionBenchmark,
}


def main():
    parser = argparse.ArgumentParser(description="SACO-Flow Benchmark Runner")
    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=[*BENCHMARKS, "all"],
        default="all",
        help="Benchmark to run (default: all)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable Taichi kernel profiler"
    )
    args = parser.parse_args()

    to_run = (
        list(BENCHMARKS.values())
        if args.benchmark == "all"
        else [BENCHMARKS[args.benchmark]]
    )

    for bench_cls in to_run:
        print(f"\nRunning {bench_cls.__name__}...")
        try:
            bench_cls(profile=args.profile).run()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
