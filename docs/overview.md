# System Overview

GPU-accelerated ecohydrological simulation of vegetation pattern formation in semiarid landscapes.

## What This System Does

Plants in semiarid environments create striking spatial patterns—bands, spots, and labyrinths—through a self-organizing process. This simulation implements the core physics in GPU-accelerated form using Taichi, targeting:

- **Scale:** 10^8 cells (10,000 x 10,000 at 10m resolution = 100km x 100km)
- **Duration:** Decades of simulated time in hours of wall time
- **Hardware:** NVIDIA H100 (sm90) and B200 (sm100)

## State Variables

| Field | Symbol | Units | Description |
|-------|--------|-------|-------------|
| `h` | Surface water | m | Ponded water available for infiltration and routing |
| `M` | Soil moisture | m | Plant-available water in root zone (depth equivalent) |
| `P` | Vegetation | kg/m^2 | Above-ground plant biomass density |
| `Z` | Elevation | m | DEM surface elevation (static) |
| `mask` | Domain mask | binary | 1 = active cell, 0 = boundary/inactive (static) |

## Governing Equations

**Surface water balance** (event timescale, minutes to hours):
```
dh/dt = R - I(h, P, M) - div(q_s)
```

**Soil moisture dynamics** (daily timescale):
```
dM/dt = I(h, P, M) - E(M, P) - L(M) + D_M * laplacian(M)
```

**Vegetation dynamics** (weekly timescale):
```
dP/dt = G(M) * P - mu * P + D_P * laplacian(P)
```

Where:
- R = rainfall intensity
- I = infiltration
- q_s = surface water flux
- E = evapotranspiration
- L = deep leakage
- G = growth rate
- mu = mortality rate
- D_M, D_P = diffusion coefficients

See `physics.md` for complete constitutive relations and parameters.

## Core Feedback Mechanism

The simulation captures the **Turing-type instability** that drives vegetation pattern formation:

1. **Positive local feedback:** Vegetation enhances local infiltration—more plants means more water enters the soil locally

2. **Negative nonlocal feedback:** Enhanced infiltration reduces runoff available to downslope neighbors—plants compete for water via runoff redistribution

3. **Pattern emergence:** This combination of local facilitation and nonlocal competition generates emergent patterns: bands on slopes, spots in flat areas, labyrinths at intermediate conditions

## Timescale Hierarchy

The system spans multiple timescales handled via operator splitting:

| Process | Timescale | Typical dt |
|---------|-----------|------------|
| Surface routing | Minutes | Adaptive (CFL-limited) |
| Infiltration | Minutes | Coupled with routing |
| Soil moisture (ET, leakage) | Daily | 1 day |
| Vegetation (growth, mortality) | Weekly | 7 days |

See `discretization.md` for numerical details.

## Code Entry Points

- `src/simulation.py`: Main simulation loop and `Simulation` class
- `src/params.py`: All parameters with defaults and validation
- `src/kernels/`: Physics implementations

## References

1. Rietkerk et al. (2002) - Self-organized patchiness and catastrophic shifts in ecosystems
2. Saco et al. (2007) - Eco-geomorphology of banded vegetation patterns
3. Saco et al. (2013) - Ecogeomorphic coevolution of semiarid hillslopes
