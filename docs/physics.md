# Physics

All constitutive relations and parameters. See `overview.md` for governing equations.

## Infiltration (Critical Nonlinearity)

```
I(h, P, M) = alpha * h * [(P + k_P * W_0) / (P + k_P)] * max(0, 1 - M/M_sat)
```

This encodes three effects:

1. **Proportional to surface water:** More ponded water = more infiltration
2. **Vegetation enhancement:**
   - Bare soil (P=0): infiltrates at rate `alpha * W_0`
   - Dense vegetation (P >> k_P): infiltrates at rate `alpha`
   - k_P controls the transition
3. **Saturation limitation:** No infiltration when soil is saturated (M = M_sat)

This is the **critical nonlinearity** driving pattern formation—vegetation creates a positive feedback on local water availability.

Source: `src/kernels/infiltration.py`

## Surface Water Flux

Kinematic wave approximation with Manning's equation:

```
q_s = h^(5/3) * sqrt(|grad(Z)|) / n
```

Where n is Manning's roughness coefficient.

Flow is partitioned to multiple downslope neighbors using MFD (Multiple Flow Direction). See `discretization.md` for the MFD algorithm.

Source: `src/kernels/flow.py`

## Evapotranspiration

```
E(M, P) = E_max * M / (M + k_ET) * (1 + beta_ET * P)
```

- **Monod kinetics in moisture:** ET increases with M but saturates
- **Vegetation enhancement:** More biomass = more transpiration
- k_ET is the half-saturation moisture

Source: `src/kernels/soil.py`

## Deep Leakage

```
L(M) = L_max * (M / M_sat)^2
```

Quadratic relationship:
- Negligible when dry (M << M_sat)
- Significant near saturation
- Represents drainage below the root zone

Source: `src/kernels/soil.py`

## Vegetation Growth

```
G(M) = g_max * M / (M + k_G)
```

Monod kinetics—growth rate saturates at high moisture. The actual biomass change is:

```
dP/dt = G(M) * P
```

Growth is multiplicative: more existing biomass enables faster growth.

Source: `src/kernels/vegetation.py`

## Vegetation Mortality

```
dP/dt = -mu * P
```

Constant mortality rate. Combined with growth, vegetation persists where G(M) > mu.

Source: `src/kernels/vegetation.py`

## Equilibrium Analysis

At vegetation equilibrium, growth rate equals mortality rate:

```
G(M*) = mu
g_max * M* / (M* + k_G) = mu
```

Solving for equilibrium moisture:

```
M* = mu * k_G / (g_max - mu)
```

This is the moisture level where vegetation is stable. Below M*, vegetation declines; above M*, it grows.

- If g_max <= mu, vegetation cannot persist at any moisture level
- Higher mortality requires higher equilibrium moisture

See `src/kernels/vegetation.py:compute_equilibrium_moisture()`

## Parameter Table

These values are from `src/params.py` (the source of truth).

### Grid Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Grid size | n | 64 | cells | n x n grid |
| Cell size | dx | 1.0 | m | Spatial resolution |

### Rainfall Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Rain depth | rain_depth | 0.02 | m | Mean event depth (~20mm) |
| Storm duration | storm_duration | 0.25 | days | Event duration (~6 hours) |
| Interstorm period | interstorm | 18.0 | days | Mean time between events |

### Infiltration Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Infiltration rate | alpha | 0.1 | 1/day | Base infiltration coefficient |
| Vegetation half-sat | k_P | 1.0 | kg/m^2 | Infiltration enhancement scale |
| Bare soil factor | W_0 | 0.2 | - | Reduced infiltration on bare soil |
| Saturation capacity | M_sat | 0.4 | m | Maximum soil moisture |

### Soil Moisture Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Max ET rate | E_max | 0.005 | m/day | Maximum evapotranspiration (~5mm/day) |
| ET half-saturation | k_ET | 0.1 | m | Moisture for half-max ET |
| Vegetation ET factor | beta_ET | 0.5 | - | ET enhancement by vegetation |
| Max leakage rate | L_max | 0.002 | 1/day | Deep drainage coefficient |
| Soil diffusivity | D_M | 0.1 | m^2/day | Lateral moisture movement |

### Vegetation Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Max growth rate | g_max | 0.02 | 1/day | Maximum relative growth |
| Growth half-sat | k_G | 0.1 | m | Moisture for half-max growth |
| Mortality rate | mu | 0.001 | 1/day | Constant mortality |
| Dispersal diffusivity | D_P | 0.01 | m^2/day | Seed spread rate |

### Routing Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Manning's n | manning_n | 0.03 | s/m^(1/3) | Surface roughness |
| Minimum slope | min_slope | 1e-6 | - | Floor for slope calculations |

### Timestep Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Vegetation timestep | dt_veg | 7.0 | days | Weekly vegetation updates |
| Soil timestep | dt_soil | 1.0 | days | Daily soil updates |

### Drainage Parameters

| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Water threshold | h_threshold | 1e-6 | m | Minimum depth for routing |
| Drainage time | drainage_time | 1.0 | days | Extra time after rainfall |

## Derived Quantities

Computed properties from `SimulationParams`:

- **Cell area:** `dx^2` [m^2]
- **Events per year:** `365 / interstorm`
- **Annual rainfall:** `rain_depth * events_per_year` [m/year]

With defaults: ~20 events/year, ~0.4 m/year annual rainfall.
