# Boundary Conditions

## Domain Mask

```
mask[i,j] = 1  →  active cell (interior)
mask[i,j] = 0  →  inactive/boundary (no flux, no update)
```

Always check mask before updating or reading neighbors.

## Diffusion Boundaries

**Neumann (no-flux)** for soil moisture and vegetation diffusion.

In Laplacian computation, only include neighbors where `mask == 1`. This naturally enforces zero-gradient at boundaries.

## Surface Water Boundaries

Water exits the domain at boundary cells with outward flow:

```python
for each boundary cell (i,j):
    for k in flow directions:
        if neighbor is out-of-bounds or mask[neighbor] == 0:
            if flow_frac[i,j,k] > 0:
                h[i,j] = 0  # water exits
```

Track cumulative outflow for mass balance verification.

## Depression Handling

Real DEMs contain local minima. Two approaches:

**Option A: Pre-fill depressions** (recommended initially)
- Use Priority-Flood algorithm to create flow-routing surface
- Store original DEM for visualization, filled DEM for routing

**Option B: Dynamic ponding** (future enhancement)
- Allow water to accumulate in depressions
- Overflow when water level exceeds pour point
- More realistic but complex

## Implementation Reference

See `ecohydro_spec.md:527-566` for boundary specifications.
