
import time
import taichi as ti
import numpy as np
from src.simulation import Simulation, SimulationParams
from src.config import init_taichi

def benchmark_10k():
    """Run 1 year simulation on 10k x 10k grid and measure performance."""
    print("Initializing Taichi...")
    # Force CUDA if available
    try:
        init_taichi(backend="cuda", debug=False)
    except:
        print("Warning: CUDA init failed, using default (CPU?)")
        init_taichi(debug=False)
        
    n = 10000
    print(f"Initializing 10k x 10k simulation (1e8 cells)...")
    
    params = SimulationParams(n=n)
    sim = Simulation(params)
    sim.initialize()
    
    print("Warmup (10 soil steps, 1 veg step)...")
    # Warmup JIT
    for _ in range(10):
        sim.step_soil(dt=0.1)
    sim.step_vegetation(dt=7.0)
    ti.sync()
    
    print("Starting 1-year benchmark...")
    start_time = time.time()
    
    # Run for 1 year
    # run() method handles events and timesteps
    final_state = sim.run(years=1.0, check_mass_balance=True, verbose=True)
    ti.sync()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nBenchmark Complete:")
    print(f"  Grid Size: {n} x {n}")
    print(f"  Duration:  1.0 year")
    print(f"  Wall time: {elapsed:.2f} seconds")
    print(f"  Speed:     {elapsed/60.0:.2f} minutes/year")
    
    target_min = 1.0
    if elapsed < 60.0 * target_min:
        print(f"SUCCESS: Performance target met (< {target_min} min/year)")
    else:
        print(f"FAILURE: Performance target NOT met (> {target_min} min/year)")

if __name__ == "__main__":
    benchmark_10k()
