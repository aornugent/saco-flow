# Flow Direction Computation

Computes Multiple Flow Direction (MFD) fractions determining how water distributes to downslope neighbors.

## Algorithm

For each cell, distribute outflow to lower neighbors proportional to slope raised to power `p`:

```
f_k = max(0, S_k)^p / Σ max(0, S_m)^p
```

Where `S_k = (Z_center - Z_neighbor) / d_k` is slope to neighbor k.

## Neighbor Convention

8-connected, clockwise from East:
```
5  6  7
4  X  0
3  2  1
```

Distances: cardinal = 1, diagonal = √2 (in cell units).

See `ecohydro_spec.md:198-208` for indexing arrays.

## Flow Exponent

- `p = 1`: Most diffuse (linear)
- `p = 1.5`: Default for hillslope flow
- `p > 5`: Approaches single-direction (D8)

## Edge Cases

- **Flat cells / local minima**: No downslope neighbors. Flag with `flow_frac[i,j,0] = -1.0` for special handling.
- **Boundary cells**: Check mask before accessing neighbors.

## Implementation Reference

See `ecohydro_spec.md:216-250` for the kernel specification.

## Tests Required

1. Tilted plane → all flow in one direction
2. Symmetric valley → flow splits equally at ridge
3. Flat terrain → fractions zero or flagged
