# Vegetation Dynamics

Updates vegetation biomass via growth, mortality, and seed dispersal.

## Governing Equation

```
∂P/∂t = G(M)·P - μ·P + D_P·∇²P
```

## Growth

Monod kinetics—saturates at high moisture:
```
G(M) = g_max · M/(M + k_G)
```

Growth rate multiplies existing biomass (logistic-like).

## Mortality

Constant rate:
```
mortality = μ·P
```

## Seed Dispersal

Diffusion with 5-point Laplacian:
```
∇²P ≈ (P_E + P_W + P_N + P_S - 4·P_center) / dx²
```

Neumann boundaries. Represents local seed spread.

## Parameters

| Parameter | Symbol | Typical | Units |
|-----------|--------|---------|-------|
| Max growth rate | g_max | 0.02 | day⁻¹ |
| Growth half-sat | k_G | 0.1 | m |
| Mortality rate | μ | 0.001 | day⁻¹ |
| Dispersal diffusivity | D_P | 0.01 | m²/day |

## Timestep

Weekly updates (7 days). Vegetation dynamics are slow relative to water.

Stability: `dt <= dx² / (4·D_P)` (usually satisfied by weekly timestep).

## Implementation Notes

- Clamp `P >= 0` (positivity constraint)
- Use double buffering for parallel update

## Implementation Reference

See `ecohydro_spec.md:427-462` for kernel specification.

## Tests Required

1. Growth increases biomass when moisture available
2. Mortality decreases biomass without moisture
3. Dispersal smooths vegetation gradients
4. Equilibrium: growth = mortality at steady state
