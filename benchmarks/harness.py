"""
Base benchmark harness for Saco-Flow.
"""
import abc
import time
import taichi as ti
from typing import Any, List

from src.config import init_taichi

class Benchmark(abc.ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(self, profile: bool = False):
        self.profile = profile
        self.init_taichi()
        
    def init_taichi(self):
        """Initialize Taichi backend."""
        print(f"Initializing Taichi (Profile: {self.profile})...")
        try:
            init_taichi(backend="cuda", debug=False, kernel_profiler=self.profile)
        except Exception as e:
            print(f"Warning: CUDA init failed ({e}), falling back to default")
            init_taichi(debug=False, kernel_profiler=self.profile)

    @abc.abstractmethod
    def run(self) -> Any:
        """Run the benchmark logic. Returns results."""
        pass

    def setup(self):
        """Optional setup usually called before run."""
        pass

    def teardown(self):
        """Optional cleanup."""
        if self.profile:
            try:
                print("\nProfiler Output:")
                ti.profiler.print_kernel_profiler_info()
                ti.profiler.clear_kernel_profiler_info()
            except Exception as e:
                print(f"Profiler error: {e}")

    def print_header(self, title: str):
        print("\n" + "="*80)
        print(f"{title:^80}")
        print("="*80)

    def print_footer(self):
        print("="*80)
