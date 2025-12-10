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

| Parameter | Symbol | Typical | Units |
|-----------|--------|---------|-------|
| Infiltration rate | α | 0.1 | day⁻¹ |
| Vegetation half-sat | k_P | 1.0 | kg/m² |
| Bare soil fraction | W_0 | 0.1 | - |
| Saturation capacity | M_sat | 0.3 | m |

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
