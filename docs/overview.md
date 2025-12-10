# System Overview

The simulation couples three fields through water-vegetation feedbacks that produce Turing-type spatial instabilities.

## State Variables

| Field | Symbol | Units | Description |
|-------|--------|-------|-------------|
| `h` | Surface water | m | Ponded water for infiltration/routing |
| `M` | Soil moisture | m | Plant-available water in root zone |
| `P` | Vegetation | kg/m² | Above-ground biomass density |

Plus static fields: elevation `Z` (m) and domain mask `Ω` (binary).

## Governing Equations

**Surface water** (event timescale, minutes):
```
∂h/∂t = R - I(h,P,M) - ∇·q_s
```

**Soil moisture** (daily timescale):
```
∂M/∂t = I(h,P,M) - E(M,P) - L(M) + D_M·∇²M
```

**Vegetation** (seasonal timescale):
```
∂P/∂t = G(M)·P - μP + D_P·∇²P
```

See `ecohydro_spec.md:39-47` for full equations.

## Core Feedback Mechanism

1. Vegetation enhances local infiltration (positive local feedback)
2. Enhanced infiltration reduces runoff to downslope neighbors (negative nonlocal feedback)
3. This instability drives pattern formation: bands, spots, labyrinths

## Parameter Reference

See `ecohydro_spec.md:72-90` for the complete parameter table with typical values and units.
