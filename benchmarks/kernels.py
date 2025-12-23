
import time
import taichi as ti
from benchmarks.harness import Benchmark
from src.simulation import SimulationParams, allocate
from src.fields import initialize_tilted_plane, initialize_vegetation
from src.kernels.soil import soil_moisture_step, soil_moisture_step_naive
from src.kernels.vegetation import vegetation_step, vegetation_step_naive

class KernelFusionBenchmark(Benchmark):
    """ Compares Fused vs Naive kernel implementations. """

    def run(self):
        self.print_header("KERNEL FUSION COMPARISON")
        
        n = 2048
        print(f"Grid size: {n}x{n} ({n**2/1e6:.1f} M cells)")
        
        # Setup
        p = SimulationParams(n=n)
        fields = allocate(n)
        initialize_tilted_plane(fields)
        initialize_vegetation(fields)
        
        # 1. Soil Moisture Comparison
        speedup_soil = self._run_comparison(
            "Soil Moisture (ET+Leakage+Diff)",
            soil_moisture_step,
            soil_moisture_step_naive,
            fields, p, steps=100
        )
        
        # 2. Vegetation Comparison
        speedup_veg = self._run_comparison(
            "Vegetation (Growth+Mort+Diff)",
            vegetation_step,
            vegetation_step_naive,
            fields, p, steps=100
        )
        
        print("\n" + "="*50)
        print(f"{'SUMMARY':^50}")
        print("="*50)
        print(f"Soil Fusion Speedup: {speedup_soil:.2f}x")
        print(f"Veg Fusion Speedup:  {speedup_veg:.2f}x")
        self.print_footer()
        
        self.teardown()

    def _run_comparison(self, name, fused_func, naive_func, fields, params, steps=1000):
        print(f"\nComparing {name} kernels ({steps} steps)...")
        
        # Common args
        args_soil = (
            params.E_max, params.k_ET, params.beta_ET,
            params.L_max, params.M_sat, params.D_M,
            params.dx, params.dt_soil
        )
        
        args_veg = (
            params.g_max, params.k_G, params.mu,
            params.D_P, params.dx, params.dt_veg
        )
        
        args = args_soil if "Soil" in name else args_veg
        
        # Warmup Fused
        for _ in range(10):
            fused_func(fields, *args)
        ti.sync()
        
        # Measure Fused
        start = time.perf_counter()
        for _ in range(steps):
            fused_func(fields, *args)
        ti.sync()
        mod_fused = time.perf_counter() - start
        
        # Warmup Naive
        for _ in range(10):
            naive_func(fields, *args)
        ti.sync()
        
        # Measure Naive
        start = time.perf_counter()
        for _ in range(steps):
            naive_func(fields, *args)
        ti.sync()
        mod_naive = time.perf_counter() - start
        
        print(f"  Fused: {mod_fused:.4f} s")
        print(f"  Naive: {mod_naive:.4f} s")
        print(f"  Speedup: {mod_naive / mod_fused:.2f}x")
        
        return mod_naive / mod_fused
