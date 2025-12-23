# Testing

Mass conservation verification, debugging patterns, and test organization.

## Conservation Law

Total water in the system must satisfy:

```
d/dt(H_total + M_total) = R_total - ET_total - L_total - Q_out
```

Where:
- H_total = sum(h[i,j]) * dx^2 — total surface water [m^3]
- M_total = sum(M[i,j]) * dx^2 — total soil moisture [m^3]
- R_total = cumulative rainfall [m^3]
- ET_total = cumulative evapotranspiration [m^3]
- L_total = cumulative deep leakage [m^3]
- Q_out = cumulative boundary outflow [m^3]

## MassBalance Class

The `MassBalance` dataclass (`src/diagnostics.py`) tracks cumulative fluxes:

```python
@dataclass
class MassBalance:
    initial_water: float = 0.0       # h + M at start [m^3]
    cumulative_rain: float = 0.0     # Total rainfall [m^3]
    cumulative_et: float = 0.0       # Total evapotranspiration [m^3]
    cumulative_leakage: float = 0.0  # Total deep leakage [m^3]
    cumulative_outflow: float = 0.0  # Total boundary outflow [m^3]

    def expected_water(self) -> float:
        """Compute expected total based on fluxes."""
        return (self.initial_water + self.cumulative_rain
                - self.cumulative_et - self.cumulative_leakage
                - self.cumulative_outflow)

    def check(self, actual: float, rtol: float = 1e-4, atol: float = 1e-8) -> float:
        """Check conservation and return relative error.

        Raises AssertionError if |actual - expected| > atol + rtol * |expected|
        """
```

## Per-Kernel Conservation

Each kernel should maintain specific invariants:

| Kernel | Conservation Check |
|--------|-------------------|
| Infiltration | delta_h = -delta_M (water transfers, not created) |
| Surface routing | sum(h) unchanged on closed domain |
| Soil diffusion | sum(M) unchanged |
| Vegetation diffusion | sum(P) unchanged |
| ET | Track cumulative removal |
| Leakage | Track cumulative removal |
| Boundary outflow | Tracked in apply_fluxes() return value |

## Common Conservation Violations

### 1. Boundary Leaks

Water escapes at domain edges without being tracked.

**Symptoms:** Conservation error grows with simulation time.

**Fix:** Ensure `apply_fluxes()` tracks outflow to cells where mask=0.

### 2. Timestep Instability

CFL violation causes negative water depths.

**Symptoms:** Negative h values, oscillations, eventually NaN.

**Fix:** Reduce timestep or fix CFL calculation.

### 3. Clamping Errors

`max(0, h)` removes water without tracking.

**Symptoms:** Small consistent water loss.

**Fix:** Track any water removed by clamping, or fix the source of negative values.

### 4. Float Accumulation

Small rounding errors compound over long simulations.

**Symptoms:** Error grows slowly but steadily.

**Fix:** Use higher tolerance for long runs, or use Kahan summation for accumulation.

## Debugging Mass Errors

Step-by-step approach:

1. **Test on simple geometry:** Flat terrain, uniform initial conditions
2. **Track fluxes separately:** Print each flux component per timestep
3. **Check boundaries:** Is water escaping without being counted?
4. **Verify timesteps:** Are stability constraints satisfied?
5. **Run conservation check frequently:** Every timestep during debugging

## Test Organization

### Running Tests

```bash
# Fast tests only (skip multi-year simulations)
pytest -m "not slow"

# All tests including slow validation
pytest

# Specific test file
pytest tests/test_flow.py

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Test Markers

- `@pytest.mark.slow`: Multi-year simulations that take minutes to run. Used for pattern emergence, parameter sensitivity, and long-term stability tests.
- Unmarked tests: Unit tests and short integration tests (seconds to run).

## Test Categories

### 1. Analytical Solution Tests

Compare simulation output against known exact solutions:

```python
def test_diffusion_gaussian_decay():
    """1D diffusion of Gaussian should spread predictably."""
    # Initial: narrow Gaussian
    # Final: wider Gaussian with known sigma
    # Compare simulation to analytical solution
```

### 2. Equilibrium Tests

Systems should reach known steady states:

```python
def test_vegetation_equilibrium():
    """At M = M*, vegetation should be stable."""
    M_equilibrium = compute_equilibrium_moisture(g_max, k_G, mu)
    # Initialize at equilibrium
    # Run simulation
    # Verify P unchanged
```

### 3. Conservation Tests

Verify mass/energy balance:

```python
def test_closed_system_conservation():
    """With no inflow/outflow, total water is constant."""
    initial = compute_total_water()
    run_simulation(steps=100)
    final = compute_total_water()
    assert abs(final - initial) < tolerance
```

### 4. Kernel Equivalence Tests

Fused kernels must match naive versions:

```python
def test_soil_fused_matches_naive():
    """Fused kernel produces same result as sequential."""
    # Run naive version
    soil_moisture_step_naive(fields_naive, ...)

    # Run fused version
    soil_moisture_step(fields_fused, ...)

    # Compare
    assert np.allclose(
        fields_naive.M.to_numpy(),
        fields_fused.M.to_numpy(),
        rtol=1e-5
    )
```

## Helper Functions

From `src/diagnostics.py`:

```python
@ti.kernel
def compute_total(field: ti.template(), mask: ti.template()) -> float:
    """Sum field values where mask == 1."""
    total = 0.0
    for I in ti.grouped(field):
        if mask[I] == 1:
            total += field[I]
    return total

def check_conservation(initial, final, fluxes=None, rtol=1e-5, atol=1e-10):
    """Check that final == initial - sum(fluxes).

    Raises AssertionError with detailed message if violated.
    """
```

## Simulation-Level Checks

The `Simulation.run()` method includes periodic mass balance checks:

```python
# Every 30 simulated days:
if check_mass_balance and int(current_day) % 30 == 0:
    error = state.mass_balance.check(state.total_water())
```

For debugging, enable verbose mode:

```python
sim.run(years=1, check_mass_balance=True, verbose=True)
```

This prints mass balance error and event count every 30 days.
