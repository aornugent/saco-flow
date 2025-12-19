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

| Parameter | Symbol | Woody | Units | Notes |
|-----------|--------|-------|-------|-------|
| Max growth rate | g_max | 0.002 | day⁻¹ | Slow woody biomass accumulation |
| Growth half-sat | k_G | 0.1 | m | |
| Mortality rate | μ | 0.0001 | day⁻¹ | ~27-year lifespan for shrubs |
| Dispersal diffusivity | D_P | 0.001 | m²/day | Slow clonal/seed spread |

**Lifespan check**: Expected lifespan = 1/μ = 10,000 days ≈ 27 years (appropriate for woody shrubs).

**Doubling time check**: At high moisture, doubling time ≈ ln(2)/(g_max - μ) ≈ 365 days (appropriate for wood).

## Timestep

Weekly updates (7 days). Vegetation dynamics are slow relative to water.

Stability: `dt <= dx² / (4·D_P)`. With D_P = 0.001 m²/day and dx = 1m:
`dt <= 1 / 0.004 = 250 days` (weekly timestep is stable).

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
