# Surface Water Routing

Routes ponded water downslope using kinematic wave approximation with MFD partitioning.

## Physics

Flux per unit width:
```
q_s = h^(5/3) · √|∇Z| / n
```

Where `n` is Manning's roughness (~0.03 s/m^(1/3)).

## Algorithm

**Pass 1 - Compute outflows:**
```
v = h^(2/3) · √S / n          # velocity
q_out = min(h/dt, v·h/dx)      # CFL-limited outflow
```

**Pass 2 - Apply fluxes:**
```
h_new = h - q_out·dt + Σ(inflow from donors)
```

Inflow from neighbor k: `flow_frac[k, donor_dir] · q_out[k] · dt`

## CFL Condition

```
dt <= dx / v_max
```

Where `v_max ≈ h^(2/3) · √S / n`. Adaptive subcycling during rainfall events.

## Implementation Notes

- Two-pass structure required for mass conservation
- Clamp `h` to non-negative after update
- Threshold `h_min` to skip negligible flow

## Implementation Reference

See `ecohydro_spec.md:329-377` for kernel specification.

## Tests Required

1. Mass conservation (closed domain)
2. Outlet boundary removes water correctly
3. Steady state reached with constant input
4. CFL timestep maintains stability
