# Data Structures and Memory Layout

## Structure of Arrays (SoA)

All fields stored as separate contiguous arrays for coalesced GPU memory access:

```python
h = ti.field(dtype=ti.f32, shape=(N, N))      # Surface water
M = ti.field(dtype=ti.f32, shape=(N, N))      # Soil moisture
P = ti.field(dtype=ti.f32, shape=(N, N))      # Vegetation
Z = ti.field(dtype=ti.f32, shape=(N, N))      # Elevation
mask = ti.field(dtype=ti.i8, shape=(N, N))    # Domain mask
flow_frac = ti.field(dtype=ti.f32, shape=(N, N, 8))  # MFD fractions
```

**Why SoA?** Threads in a warp access consecutive memory locations when iterating over the same field.

## Double Buffering

Parallel stencil updates require reading old values while writing new:

```python
M = ti.field(...)      # Read from
M_new = ti.field(...)  # Write to

# After kernel completes:
M, M_new = M_new, M    # Swap (or copy)
```

## Neighbor Indexing

8-connected, clockwise from East:

```
Index:  5  6  7
        4  X  0
        3  2  1

neighbor_di = [0, 1, 1, 1, 0, -1, -1, -1]   # row offset
neighbor_dj = [1, 1, 0, -1, -1, -1, 0, 1]   # col offset
neighbor_dist = [1, √2, 1, √2, 1, √2, 1, √2]  # distance in cells
```

Reverse direction: `(k + 4) % 8`

## Memory Budget (10⁸ cells)

| Fields | Bytes/cell | Total |
|--------|------------|-------|
| Primary (h, M, P) | 12 | 1.2 GB |
| Flow fractions | 32 | 3.2 GB |
| Static (Z, mask) | 5 | 0.5 GB |
| Buffers | 16 | 1.6 GB |
| **Total** | | **~6.5 GB** |

This is <5% of H100's 80GB HBM, leaving room for larger domains or additional physics.

## Implementation Reference

See `ecohydro_spec.md:159-208` for complete field specifications.
