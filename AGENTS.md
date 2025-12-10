# Agent Guidelines for EcoHydro Development

This document helps AI coding agents (Claude Code, etc.) understand the project and contribute effectively.

## Project Context

EcoHydro is a GPU-accelerated simulation of vegetation pattern formation in semiarid landscapes. It implements coupled water-vegetation dynamics using Taichi for GPU compute.

**Read these files first:**
1. `ecohydro_spec.md` - Complete mathematical specification and algorithm design
2. `IMPLEMENTATION_PLAN.md` - Current development phase and priorities
3. `README.md` - Project overview and structure

## Core Principles

### 1. Simplicity Over Complexity

**Prefer the simplest solution that works.** This project models complex physics, but the code should remain approachable.

- Write straightforward code before optimizing
- Avoid premature abstraction—three similar lines are better than a clever helper
- Each function should do one thing well
- If a solution feels complicated, step back and find a simpler approach

### 2. Correctness Before Performance

**Get it right, then make it fast.**

- Implement the naive version first
- Add comprehensive tests before optimizing
- Verify mass conservation at every stage
- Performance optimization is Phase 3—don't jump ahead

### 3. Explicit Over Implicit

- Name variables after their physical meaning (`soil_moisture` not `M_local`)
- Include units in comments where ambiguous
- Make data flow obvious—avoid hidden state
- Prefer verbose clarity over terse cleverness

### 4. Test-Driven Development

- Write tests alongside implementation
- Every kernel should have a mass conservation test
- Use simple synthetic cases with known behavior
- Tests are documentation—they show how code should behave

## Technical Conventions

### Taichi Patterns

```python
# Field declarations - always document physical meaning and units
h = ti.field(dtype=ti.f32, shape=(N, N))  # Surface water depth [m]
M = ti.field(dtype=ti.f32, shape=(N, N))  # Soil moisture [m]
P = ti.field(dtype=ti.f32, shape=(N, N))  # Vegetation biomass [kg/m²]

# Kernel structure - clear sections with comments
@ti.kernel
def update_soil_moisture(dt: ti.f32):
    """
    Update soil moisture field for one timestep.

    Physics: dM/dt = I - ET - L + D_M * laplacian(M)
    """
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue

        # Load local values
        M_local = M[i, j]
        P_local = P[i, j]

        # Compute fluxes
        ET = compute_evapotranspiration(M_local, P_local)
        leakage = compute_leakage(M_local)
        diffusion = compute_laplacian(M, i, j) * D_M

        # Update with bounds checking
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M_local + (diffusion - ET - leakage) * dt))
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Fields | `snake_case`, physical name | `surface_water`, `soil_moisture` |
| Kernels | `snake_case`, verb phrase | `compute_flow_directions`, `update_vegetation` |
| Parameters | Match spec symbols with readable alternatives | `infiltration_rate` or `alpha` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS`, `CONVERGENCE_THRESHOLD` |
| Test functions | `test_<what>_<expected>` | `test_flat_terrain_no_flow` |

### File Organization

```
src/kernels/flow.py      # One kernel type per file
src/kernels/soil.py      # Keep kernels focused
tests/test_flow.py       # Mirror structure in tests
```

### Parameter Handling

```python
# Group related parameters in dataclasses
@dataclass
class InfiltrationParams:
    """Parameters for vegetation-enhanced infiltration."""
    alpha: float = 0.1      # Base infiltration rate [day⁻¹]
    k_P: float = 1.0        # Vegetation half-saturation [kg/m²]
    W_0: float = 0.1        # Bare soil fraction [-]
    M_sat: float = 0.3      # Saturation capacity [m]
```

## Common Tasks

### Adding a New Kernel

1. **Understand the physics** - Read relevant section in `ecohydro_spec.md`
2. **Write the test first** - What behavior should we verify?
3. **Implement simply** - Get it working before optimizing
4. **Verify conservation** - Does mass/energy balance?
5. **Document** - Docstring with physics equation and units

### Debugging Mass Conservation Errors

1. Check boundary conditions—water escaping at edges?
2. Verify timestep stability—CFL condition satisfied?
3. Add intermediate diagnostics—track each flux term
4. Test on trivial case—flat terrain, uniform initial conditions
5. Check for floating point accumulation errors

### Performance Investigation

1. Profile first—don't guess the bottleneck
2. Check memory access patterns—coalesced reads?
3. Consider kernel fusion—multiple passes over same data?
4. Verify occupancy—enough parallelism?

## Architecture Decisions

### Why Taichi?

- Python frontend with GPU backend—rapid iteration
- Automatic differentiation (future: parameter estimation)
- Cross-platform (CUDA, Metal, Vulkan)
- Good balance of productivity and performance

### Why Structure of Arrays (SoA)?

```python
# SoA: Each field is a contiguous array
h = ti.field(...)  # All h values contiguous
M = ti.field(...)  # All M values contiguous

# NOT Array of Structures (AoS):
# cells = ti.Struct.field({"h": ti.f32, "M": ti.f32, ...})

# SoA enables coalesced memory access when threads
# in a warp read the same field at adjacent locations
```

### Why Multiple Flow Direction (MFD)?

Single flow direction (D8) creates unrealistic flow concentration. MFD distributes flow proportionally to slope, producing diffuse hillslope flow appropriate for semiarid systems. The exponent `p` controls concentration:
- `p = 1`: Linear, most diffuse
- `p = 1.5`: Default for hillslopes
- `p > 5`: Approaches single-direction flow

### Why Operator Splitting?

The system spans timescales from minutes (surface flow) to years (vegetation). Solving everything simultaneously would require tiny timesteps. Operator splitting lets us:
- Subcycle fast processes (surface routing) only when needed
- Use larger timesteps for slow processes (vegetation)
- Maintain stability while maximizing efficiency

## Gotchas and Edge Cases

### Flat Terrain / Local Minima

Cells with no downslope neighbors need special handling:
- Pre-fill depressions (simpler, recommended for now)
- Or track dynamic ponding (future enhancement)

The spec flags these with `flow_frac[i, j, 0] = -1.0`

### Boundary Conditions

- **Domain mask:** 0 = inactive, 1 = active
- **Diffusion:** Neumann (no-flux) at boundaries
- **Surface flow:** Water exits at domain edges

### Numerical Stability

- Surface routing: CFL condition `dt <= dx / v_max`
- Diffusion: `dt <= dx² / (4 * D)` for explicit scheme
- Always clamp fields to physical bounds (non-negative, below saturation)

### Convergence in Flow Accumulation

The iterative flow accumulation converges in O(longest_flow_path) iterations. For typical terrain, 20-50 iterations suffice for 10k×10k grids. Don't wait for perfect convergence—physically reasonable flow that conserves mass is sufficient.

## Testing Philosophy

### What to Test

1. **Conservation:** Total water in = water out + storage change
2. **Limiting cases:** Flat terrain, saturated soil, no vegetation
3. **Symmetry:** Symmetric inputs should produce symmetric outputs
4. **Stability:** Long runs shouldn't blow up or drift
5. **Known solutions:** Analytical solutions for simplified cases

### Test Structure

```python
def test_infiltration_conserves_water():
    """Water lost from surface equals water gained in soil."""
    # Arrange
    sim = create_test_simulation(nx=100, ny=100)
    sim.h.fill(0.01)  # 1cm ponded water
    sim.M.fill(0.05)  # Unsaturated soil

    initial_water = sim.total_water()

    # Act
    sim.infiltration_step(dt=0.1)

    # Assert
    final_water = sim.total_water()
    assert abs(final_water - initial_water) < 1e-10
```

### Synthetic Test Cases

| Case | Purpose | Expected Behavior |
|------|---------|-------------------|
| Flat plane | Baseline | No lateral flow, uniform infiltration |
| Tilted plane | Flow routing | Water accumulates at lower edge |
| V-shaped valley | Flow convergence | Flow concentrates in valley bottom |
| Ridge | Flow divergence | Flow splits to both sides |
| Isolated depression | Ponding | Water accumulates until overflow |

## Current Development Focus

Check `IMPLEMENTATION_PLAN.md` for the current phase. As of initial setup:

**Phase 1: Surface Water Routing**
- MFD flow direction computation
- Iterative flow accumulation
- Kinematic wave routing
- Mass conservation verification

**Not yet in scope:**
- Vegetation dynamics
- Real DEM support
- Performance optimization
- Extensions from Section 14 of spec

## Asking for Help

If you're stuck:
1. Re-read the relevant section of `ecohydro_spec.md`
2. Check if there's a test case that clarifies expected behavior
3. Look for similar patterns in existing code
4. When in doubt, implement the simpler version and note limitations

## Contributing Checklist

Before considering work complete:
- [ ] Code follows naming conventions
- [ ] Docstrings explain physics and units
- [ ] Tests cover conservation and edge cases
- [ ] No unnecessary complexity introduced
- [ ] Changes are focused—one logical unit of work
