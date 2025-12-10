# Flow Accumulation

Computes contributing area at each cell using iterative parallel relaxation.

## Algorithm

```
Initialize: A[i,j] = local_runoff[i,j]

Repeat until converged:
    A_new[i,j] = local[i,j] + Σ f_{k→(i,j)} · A[k]  (for k in donors)
    converged = max|A_new - A| < ε
    A = A_new
```

This parallelizes trivially—each cell reads from neighbors.

## Convergence

- Typically 20-50 iterations for 10k×10k grids
- Convergence time ∝ longest flow path
- **Key insight**: Perfect convergence unnecessary. Physically reasonable flow that conserves mass is sufficient.

## Implementation Notes

- Use double buffering (`A` and `A_new`) for parallel update
- Reverse direction lookup: neighbor k donates via direction `(k+4) % 8`
- Can use fixed iteration count for predictable performance

## Implementation Reference

See `ecohydro_spec.md:252-297` for kernel specification.

## Tests Required

1. Total accumulation at outlet = total input (conservation)
2. Tilted plane → linear increase downslope
3. Valley → accumulation concentrates in channel
