# Soil Kernels

Evapotranspiration, deep leakage, and lateral diffusion.

Source: `src/kernels/soil.py`

## Governing Equation

```
dM/dt = -E(M,P) - L(M) + D_M * laplacian(M)
```

Infiltration (I) is handled separately in `src/kernels/infiltration.py`.

## Evapotranspiration

```python
evapotranspiration_step(M, P, mask, E_max, k_M, beta_E, dt) -> float
```

Removes water from soil via plant transpiration and soil evaporation.

### Physics

```
E = E_max * M / (M + k_ET) * (1 + beta_ET * P) * dt
```

- **Monod kinetics:** ET increases with moisture but saturates
- **Vegetation enhancement:** More biomass = more transpiration
- **Limited by available moisture:** `ET = min(ET, M)`

Returns total ET for mass balance tracking.

## Deep Leakage

```python
leakage_step(M, mask, L_max, M_sat, dt) -> float
```

Removes water via drainage below the root zone.

### Physics

```
L = L_max * (M / M_sat)^2 * dt
```

Quadratic relationship:
- Negligible when dry (M << M_sat)
- Significant near saturation
- Limited by available moisture: `L = min(L, M)`

Returns total leakage for mass balance tracking.

## Fused Kernel

```python
et_leakage_step_fused(M, P, mask, E_max, k_M, beta_E, L_max, M_sat, dt) -> Vector[2]
```

Combines ET and leakage in a single pass for 2x less memory traffic.

### Implementation Details

1. Reads M and P once
2. Computes ET first
3. Applies leakage to remaining moisture (after ET)
4. Writes M once

Returns `[total_et, total_leakage]` vector for mass balance.

### Why Fused?

Naive (separate kernels):
```
ET:      Read M,P -> Write M
Leakage: Read M   -> Write M
Total: 6 reads + 2 writes
```

Fused:
```
Combined: Read M,P -> Write M
Total: 3 reads + 1 write (2x improvement)
```

## Diffusion

Uses the generic `laplacian_diffusion_step()` from `src/geometry.py`:

```python
laplacian_diffusion_step(field, field_new, mask, D, dx, dt)
```

### Features

- 5-point Laplacian stencil
- Neumann (no-flux) boundaries: only includes neighbors where mask=1
- `ti.block_local()` for GPU shared memory caching
- Non-negativity constraint: `field_new = max(0, ...)`

## Combined Step

```python
soil_moisture_step(fields, E_max, k_ET, beta_ET, L_max, M_sat, D_M, dx, dt) -> tuple[float, float]
```

The main entry point for soil moisture updates.

### Sequence

1. **Fused ET+leakage:** Point-wise, modifies M in-place
2. **Diffusion:** Stencil operation, reads M, writes M_new
3. **Buffer swap:** `swap_buffers(fields, "M")`

After this call, `fields.M` holds the updated soil moisture.

Returns `(total_et, total_leakage)` for mass balance.

## Naive Version

```python
soil_moisture_step_naive(fields, ...) -> tuple[float, float]
```

Same physics but uses separate kernels for ET and leakage. Kept for regression testing to verify fused kernel correctness.

## Stability

Diffusion stability constraint:

```
dt <= 0.25 * dx^2 / D_M
```

For default parameters (D_M=0.1, dx=1.0): dt <= 2.5 days.

The daily timestep (dt_soil=1.0) is well within this limit.

```python
compute_diffusion_timestep(D_M, dx, cfl=0.25) -> float
```

## Parameters

From `src/params.py`:

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| E_max | E_max | 0.005 | m/day |
| k_ET | k_ET | 0.1 | m |
| beta_ET | beta_ET | 0.5 | - |
| L_max | L_max | 0.002 | 1/day |
| M_sat | M_sat | 0.4 | m |
| D_M | D_M | 0.1 | m^2/day |

## Tests Required

1. **ET reduces moisture:** Higher M and P -> more ET
2. **Leakage quadratic:** Verify (M/M_sat)^2 relationship
3. **Diffusion smooths:** Gradients reduce over time
4. **Diffusion conserves:** Total M unchanged by diffusion alone
5. **Stability:** No oscillations at computed timestep limit
6. **Fused matches naive:** Results identical within tolerance
