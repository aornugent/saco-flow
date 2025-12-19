# Infiltration

Transfers water from surface ponding to soil moisture with vegetation enhancement.

## Physics

```
I(h,P,M) = α · h · [(P + k_P·W_0)/(P + k_P)] · (1 - M/M_sat)⁺
```

This encodes:
- Infiltration ∝ available surface water `h`
- **Vegetation enhancement**: bare soil infiltrates at rate `α·W_0`, dense vegetation at rate `α`
- **Saturation limit**: no infiltration when `M = M_sat`

## Parameters

| Parameter | Symbol | Woody | Units | Notes |
|-----------|--------|-------|-------|-------|
| Infiltration rate | α | 200.0 | day⁻¹ | ~K_sat for sandy-loam soil |
| Vegetation half-sat | k_P | 5.0 | kg/m² | Requires significant woody biomass |
| Bare soil fraction | W_0 | 0.2 | - | |
| Saturation capacity | M_sat | 0.4 | m | |

**Infiltration timescale check**: With α = 200 day⁻¹ and h = 10mm ponded water,
infiltration rate I ≈ 200 × 0.01 = 2 m/day = 83 mm/hr.
Surface water clears within ~0.1 day (2.4 hours), matching field observations.

## Implementation Notes

- Limit actual infiltration by available water AND remaining capacity:
  ```
  I_actual = min(I_potential, h, M_sat - M)
  ```
- Update both fields atomically: `h -= I`, `M += I`
- This is the **critical nonlinearity** driving pattern formation

## Implementation Reference

See `ecohydro_spec.md:299-327` for kernel specification.

## Tests Required

1. Conservation: `Δh = -ΔM`
2. No infiltration when `M = M_sat`
3. No infiltration when `h = 0`
4. Higher infiltration with more vegetation
