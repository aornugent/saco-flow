# Soil Moisture Dynamics

Updates soil moisture via evapotranspiration, deep leakage, and lateral diffusion.

## Governing Equation

```
∂M/∂t = I - E(M,P) - L(M) + D_M·∇²M
```

(Infiltration `I` handled separately in infiltration kernel)

## Evapotranspiration

```
E(M,P) = E_max · M/(M + k_M) · (1 + β_E·P)
```

- Monod kinetics in moisture
- Vegetation enhances ET

## Deep Leakage

```
L(M) = L_max · (M/M_sat)²
```

Quadratic—negligible when dry, significant near saturation.

## Lateral Diffusion

5-point Laplacian stencil:
```
∇²M ≈ (M_E + M_W + M_N + M_S - 4·M_center) / dx²
```

Neumann (no-flux) boundary conditions.

## Parameters

| Parameter | Symbol | Typical | Units |
|-----------|--------|---------|-------|
| Max ET rate | E_max | 0.005 | m/day |
| ET half-sat | k_M | 0.05 | m |
| Vegetation ET factor | β_E | 0.5 | m²/kg |
| Max leakage | L_max | 0.001 | m/day |
| Diffusivity | D_M | 0.1 | m²/day |

## Stability

```
dt <= dx² / (4·D_M)
```

## Implementation Reference

See `ecohydro_spec.md:379-421` for kernel specification.

## Tests Required

1. ET reduces moisture
2. Diffusion smooths gradients
3. Diffusion alone conserves mass
4. Stability at computed timestep
