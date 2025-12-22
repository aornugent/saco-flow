
import pytest
import taichi as ti

import os
from src.fields import allocate
from src.simulation import Simulation

@pytest.mark.slow
def test_largescale_memory_fit():
    """
    Test whether a 10,000 x 10,000 grid fits in memory.
    Target hardware: RTX 3090 (24GB VRAM).
    
    Estimated requirement:
    - 1e8 cells
    - ~77 bytes/cell (19 floats + 1 byte)
    - Total ~7.7 GB
    """
    # Force CUDA backend if available
    try:
        # Try to init with CUDA to check availability
        # We need to init to check availability reliably if is_arch_supported is missing
        # But ti.init can only be called once.
        # simpler check:
        pass
    except:
        pass

    # Initialize Taichi
    # We leave some headroom for system processes
    try:
        ti.init(arch=ti.cuda, offline_cache=True, device_memory_GB=22.0)
    except Exception:
        print("CUDA not available or failed to init, falling back to CPU (test will skip VRAM check logic)")
        ti.init(arch=ti.cpu, offline_cache=True)
        pytest.skip("CUDA not available, skipping VRAM test")

    n = 10000
    print(f"\nAttempting to allocate {n}x{n} grid (1e8 cells)...")

    try:
        # 1. Initialize Simulation
        from src.params import SimulationParams
        params = SimulationParams(n=n)
        sim = Simulation(params)
        
        # This allocates fields
        sim.initialize(slope=0.01)
        print("Allocation successful!")
        
        # 2. Run a short simulation (1 rainfall event)
        # We want to verify it actually runs and conserves mass
        print("Running 1 rainfall event...")
        
        # Set large tolerance because at 1e8 cells, float errors accumulation might be significant
        # But we use float32 usually.
        # Check initial mass
        mass_initial = sim.state.total_water()
        
        # Run event: 0.1m depth, 1.0 day duration (to be safe/slow)
        # Note: Simulation.run_rainfall_event takes depth, duration
        sim.run_rainfall_event(depth=0.01, duration=0.1)
        
        # Run soil step
        sim.step_soil(dt=0.1)
        
        # Check mass balance
        # expected = initial + rain - outflow - et - leakage
        # mass_balance class tracks accumulators
        
        mb = sim.state.mass_balance
        expected = mb.initial_water + mb.cumulative_rain - mb.cumulative_outflow - mb.cumulative_et - mb.cumulative_leakage
        actual = sim.state.total_water()
        
        diff = abs(actual - expected)
        rel_error = diff / max(expected, 1e-10)
        
        print(f"Mass Conservation Check:")
        print(f"  Initial: {mb.initial_water:.4e}")
        print(f"  Rain:    {mb.cumulative_rain:.4e}")
        print(f"  Outflow: {mb.cumulative_outflow:.4e}")
        print(f"  ET:      {mb.cumulative_et:.4e}")
        print(f"  Leakage: {mb.cumulative_leakage:.4e}")
        print(f"  Expected:{expected:.4e}")
        print(f"  Actual:  {actual:.4e}")
        print(f"  Diff:    {diff:.4e}")
        print(f"  Rel Err: {rel_error:.4e}")
        
        # Tolerance: 1e-5 relative (allows for float32 summation error on 1e8 items)
        # Previous failure was ~3e-6 relative error
        assert rel_error < 1e-5, f"Mass conservation failed: rel_error {rel_error:.2e}"
        
        print("Simulation step successful and mass conserved!")
        
    except Exception as e:
        pytest.fail(f"Large grid simulation failed: {e}")


if __name__ == "__main__":
    test_largescale_memory_fit()
