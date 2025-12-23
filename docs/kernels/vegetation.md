# Vegetation Kernels

Growth, mortality, and seed dispersal.

Source: `src/kernels/vegetation.py`

## Governing Equation

```
dP/dt = G(M) * P - mu * P + D_P * laplacian(P)
```

Where:
- G(M) = growth rate depending on soil moisture
- mu = constant mortality rate
- D_P = seed dispersal diffusivity

## Growth

```python
growth_step(P, M, mask, g_max, k_G, dt) -> float
```

Vegetation growth using Monod kinetics.

### Physics

```
G(M) = g_max * M / (M + k_G)
dP = G(M) * P * dt
```

- Growth rate saturates at high moisture (Monod kinetics)
- Growth is multiplicative: more biomass enables faster growth
- No growth when M <= 0 or P <= 0

Returns total growth for tracking.

## Mortality

```python
mortality_step(P, mask, mu, dt) -> float
```

Constant mortality rate.

### Physics

```
dP = -mu * P * dt
```

- Linear in biomass
- Limited to available biomass: `mortality = min(mu * P * dt, P)`

Returns total mortality for tracking.

## Fused Kernel

```python
growth_mortality_step_fused(P, M, mask, g_max, k_G, mu, dt) -> Vector[2]
```

Combines growth and mortality in a single pass for 2x less memory traffic.

### Implementation Details

1. Reads P and M once
2. Applies growth first: `P_grown = P + G(M) * P * dt`
3. Applies mortality to grown value: `P_final = P_grown - mu * P_grown * dt`
4. Writes P once

This matches the naive sequential behavior (growth before mortality).

Returns `[total_growth, total_mortality]` vector.

### Why Fused?

Naive:
```
Growth:    Read P,M -> Write P
Mortality: Read P   -> Write P
Total: 5 reads + 2 writes
```

Fused:
```
Combined: Read P,M -> Write P
Total: 3 reads + 1 write (2x improvement)
```

## Diffusion (Seed Dispersal)

Uses the generic `laplacian_diffusion_step()` from `src/geometry.py`.

Represents local seed spread from vegetated areas to neighbors.

## Combined Step

```python
vegetation_step(fields, g_max, k_G, mu, D_P, dx, dt) -> tuple[float, float]
```

The main entry point for vegetation updates.

### Sequence

1. **Fused growth+mortality:** Point-wise, modifies P in-place
2. **Diffusion:** Stencil operation, reads P, writes P_new
3. **Buffer swap:** `swap_buffers(fields, "P")`

After this call, `fields.P` holds the updated vegetation biomass.

Returns `(total_growth, total_mortality)`.

## Naive Version

```python
vegetation_step_naive(fields, ...) -> tuple[float, float]
```

Same physics but uses separate kernels. Kept for regression testing.

## Equilibrium Analysis

```python
compute_equilibrium_moisture(g_max, k_G, mu) -> float
```

At vegetation equilibrium, growth rate equals mortality rate:

```
G(M*) = mu
g_max * M* / (M* + k_G) = mu
```

Solving for equilibrium moisture:

```
M* = mu * k_G / (g_max - mu)
```

This is the moisture level where vegetation is stable:
- Below M*: vegetation declines (mortality > growth)
- Above M*: vegetation increases (growth > mortality)

Returns infinity if `g_max <= mu` (vegetation cannot persist).

### Example

With defaults (g_max=0.02, k_G=0.1, mu=0.001):
```
M* = 0.001 * 0.1 / (0.02 - 0.001) = 0.00526 m
```

Vegetation is stable when soil moisture exceeds ~5.3 mm.

## Timestep

Default: dt_veg = 7 days (weekly updates).

Vegetation dynamics are slow compared to water, so weekly updates are sufficient.

### Stability

Diffusion stability constraint:

```
dt <= 0.25 * dx^2 / D_P
```

For default parameters (D_P=0.01, dx=1.0): dt <= 25 days.

The weekly timestep (7 days) is well within this limit.

```python
compute_vegetation_timestep(D_P, dx, cfl=0.25) -> float
```

## Parameters

From `src/params.py`:

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| g_max | g_max | 0.02 | 1/day |
| k_G | k_G | 0.1 | m |
| mu | mu | 0.001 | 1/day |
| D_P | D_P | 0.01 | m^2/day |

## Tests Required

1. **Growth increases with moisture:** Higher M -> faster growth
2. **Growth multiplicative:** More P -> more absolute growth
3. **Mortality proportional:** Linear in P
4. **Dispersal smooths:** Gradients reduce over time
5. **Dispersal conserves:** Total P unchanged by diffusion alone
6. **Equilibrium stable:** At M=M*, vegetation doesn't change
7. **Fused matches naive:** Results identical within tolerance
