"""Base benchmark harness for SACO-Flow."""

import abc
from typing import Any

import taichi as ti

from src.config import init_taichi


class Benchmark(abc.ABC):
    """Abstract base class for all benchmarks."""

    def __init__(self, profile: bool = False):
        self.profile = profile
        self._init_taichi()

    def _init_taichi(self):
        print(f"Initializing Taichi (profile={self.profile})...")
        try:
            init_taichi(backend="cuda", debug=False, kernel_profiler=self.profile)
        except Exception as e:
            print(f"CUDA unavailable ({e}), falling back to auto-detect")
            init_taichi(debug=False, kernel_profiler=self.profile)

    @abc.abstractmethod
    def run(self) -> Any:
        """Run the benchmark and return results."""

    def teardown(self):
        if self.profile:
            try:
                print("\nProfiler Output:")
                ti.profiler.print_kernel_profiler_info()
                ti.profiler.clear_kernel_profiler_info()
            except Exception as e:
                print(f"Profiler error: {e}")

    def print_header(self, title: str):
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

    def print_footer(self):
        print("=" * 80)
