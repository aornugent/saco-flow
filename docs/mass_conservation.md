# Mass Conservation

Every kernel must conserve mass. This is the primary correctness check.

## Conservation Law

```
d/dt(H_total + M_total) = R_total - ET_total - L_total - Q_out
```

Where:
- `H_total = Σ h[i,j] · dx²` (total surface water)
- `M_total = Σ M[i,j] · dx²` (total soil moisture)
- `Q_out` = water exiting at boundaries

## Tracking Fluxes

Maintain cumulative totals:

```python
cumulative_rain = 0.0
cumulative_ET = 0.0
cumulative_leakage = 0.0
cumulative_outflow = 0.0
```

Update these in each kernel that adds/removes water.

## Verification

```python
expected = initial_water + cumulative_rain - cumulative_ET - cumulative_leakage - cumulative_outflow
actual = compute_total_water()
error = abs(expected - actual) / max(expected, 1e-10)
assert error < 1e-6, f"Mass conservation violated: {error:.2e}"
```

Run this check:
- After every timestep during development
- Periodically (e.g., monthly) in production runs

## Common Violations

1. **Boundary leaks**: Water escaping at edges without being tracked
2. **Timestep instability**: CFL violation causing negative water
3. **Clamping errors**: `max(0, h)` without tracking removed water
4. **Float accumulation**: Small errors compound over long runs

## Per-Kernel Tests

| Kernel | Conservation Check |
|--------|-------------------|
| Infiltration | `Δh = -ΔM` |
| Surface routing | `Σh` unchanged (closed domain) |
| Soil diffusion | `ΣM` unchanged |
| ET/Leakage | Track cumulative removal |

## Implementation Reference

See `ecohydro_spec.md:569-607` for diagnostic kernel specification.
