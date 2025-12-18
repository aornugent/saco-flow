"""
Benchmark diffusion kernel performance.

Measures the impact of ti.block_local() shared memory caching on stencil operations.

Protocol:
1. Warmup: 10 timesteps (JIT compilation)
2. Measurement: 100+ timesteps
3. Grid sizes: 256, 512, 1024 (scalable based on available memory)

Usage:
    python -m benchmarks.benchmark_diffusion
"""

import time
from dataclasses import dataclass

import numpy as np

from src.config import init_taichi
from src.fields import allocate
from src.geometry import laplacian_diffusion_step
from src.kernels.soil import compute_diffusion_timestep


@dataclass
class BenchmarkResult:
    """Benchmark result for a single grid size."""

    grid_size: int
    n_steps: int
    total_time_s: float
    time_per_step_ms: float
    cells_per_second: float
    bandwidth_gb_s: float  # Approximate achieved bandwidth


def benchmark_diffusion(n: int, n_warmup: int = 10, n_steps: int = 100) -> BenchmarkResult:
    """Benchmark diffusion kernel at given grid size.

    Args:
        n: Grid size (n x n)
        n_warmup: Warmup steps for JIT compilation
        n_steps: Measurement steps

    Returns:
        BenchmarkResult with timing metrics
    """
    # Allocate fields
    fields = allocate(n)

    # Initialize with random data
    M_np = np.random.uniform(0.1, 0.3, (n, n)).astype(np.float32)
    fields.M.from_numpy(M_np)

    # Set up mask (interior only)
    mask_np = np.ones((n, n), dtype=np.int32)
    mask_np[0, :] = mask_np[-1, :] = mask_np[:, 0] = mask_np[:, -1] = 0
    fields.mask.from_numpy(mask_np)

    # Diffusion parameters
    D = 0.1  # m²/day
    dx = 1.0  # m
    dt = compute_diffusion_timestep(D, dx, cfl=0.2)

    # Warmup (JIT compilation)
    print(f"  Warming up ({n_warmup} steps)...", end=" ", flush=True)
    for _ in range(n_warmup):
        laplacian_diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        # Swap buffers manually
        fields.M.from_numpy(fields.M_new.to_numpy())
    print("done")

    # Synchronize before timing
    import taichi as ti

    ti.sync()

    # Measurement
    print(f"  Measuring ({n_steps} steps)...", end=" ", flush=True)
    start = time.perf_counter()

    for _ in range(n_steps):
        laplacian_diffusion_step(fields.M, fields.M_new, fields.mask, D, dx, dt)
        # Swap buffers manually
        fields.M.from_numpy(fields.M_new.to_numpy())

    ti.sync()
    elapsed = time.perf_counter() - start
    print("done")

    # Calculate metrics
    n_cells = n * n
    time_per_step_ms = (elapsed / n_steps) * 1000
    cells_per_second = (n_cells * n_steps) / elapsed

    # Bandwidth estimate:
    # 5-point stencil reads 5 cells (field) + 5 cells (mask) = 10 reads
    # Plus 1 write to field_new
    # Total ~11 floats/ints per cell = ~44 bytes/cell
    bytes_per_cell = 44  # Approximate
    total_bytes = bytes_per_cell * n_cells * n_steps
    bandwidth_gb_s = total_bytes / elapsed / 1e9

    return BenchmarkResult(
        grid_size=n,
        n_steps=n_steps,
        total_time_s=elapsed,
        time_per_step_ms=time_per_step_ms,
        cells_per_second=cells_per_second,
        bandwidth_gb_s=bandwidth_gb_s,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results table."""
    print("\n" + "=" * 70)
    print("DIFFUSION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Grid':<10} {'Steps':<8} {'Total (s)':<12} {'Per Step (ms)':<14} {'Cells/s':<12} {'BW (GB/s)':<10}")
    print("-" * 70)

    for r in results:
        print(
            f"{r.grid_size:>4}x{r.grid_size:<5} "
            f"{r.n_steps:<8} "
            f"{r.total_time_s:<12.3f} "
            f"{r.time_per_step_ms:<14.3f} "
            f"{r.cells_per_second:<12.2e} "
            f"{r.bandwidth_gb_s:<10.2f}"
        )

    print("=" * 70)


def main() -> None:
    """Run diffusion benchmarks."""
    print("Initializing Taichi...")
    backend = init_taichi()
    print(f"Backend: {backend}\n")

    # Grid sizes to test
    # Start small, scale up based on available memory
    grid_sizes = [256, 512, 1024]

    # Optionally add larger sizes for GPU testing
    if backend == "cuda":
        grid_sizes.extend([2048])  # Add more for GPU

    results = []

    for n in grid_sizes:
        print(f"\nBenchmarking {n}x{n} grid...")
        try:
            result = benchmark_diffusion(n, n_warmup=10, n_steps=100)
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")

    print_results(results)

    # Performance summary
    if results:
        print("\nPerformance Notes:")
        print("- ti.block_local() caches stencil reads in GPU shared memory")
        print("- Expected improvement: 1.5-2x on GPU for bandwidth-bound stencils")
        print("- CPU backend may show minimal difference (no shared memory)")


if __name__ == "__main__":
    main()
