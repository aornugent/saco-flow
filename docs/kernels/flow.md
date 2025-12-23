# Flow Kernels

Surface water routing using MFD and kinematic wave approximation.

Source: `src/kernels/flow.py`

## Overview

Three-stage process:
1. **Flow directions** (once per DEM): Compute MFD fractions from elevation
2. **Flow accumulation** (as needed): Iterative parallel relaxation
3. **Surface routing** (per timestep): Two-pass kinematic wave

## Flow Direction Computation

```python
compute_flow_directions(Z, mask, flow_frac, dx, p)
```

Computes Multiple Flow Direction (MFD) fractions for each cell. Flow is distributed to all lower neighbors proportional to slope raised to power p.

### Algorithm

For each active cell:
1. Compute slope `S_k = (Z_center - Z_k) / d_k` to each of 8 neighbors
2. For positive (downslope) slopes: compute `slope^p`
3. Normalize to fractions summing to 1.0

### Flow Exponent

The exponent p controls flow concentration:
- p = 1.0: Most diffuse (linear weighting)
- p = 1.5: Default (FLOW_EXPONENT in code)
- p > 5: Approaches single-direction D8

### Edge Cases

**Flat cells:** When slope_sum < MIN_SLOPE_SUM (1e-10), the cell has no downslope neighbors. Flagged with `flow_frac[i,j,0] = -1` for special handling. Water accumulates in these cells.

**Boundary neighbors:** Flow can be directed to inactive cells (mask=0), allowing water to exit the domain.

## Flow Accumulation

```python
compute_flow_accumulation(local_source, flow_acc, flow_acc_new, flow_frac, mask, max_iters=100, tol=1e-6)
```

Iteratively computes contributing area at each cell.

### Algorithm

```
Initialize: A[i,j] = local_source[i,j]

Repeat until converged:
    A_new[i,j] = local[i,j] + sum(f_{k->here} * A[k]) for all donors k
    converged = max|A_new - A| < tolerance
    A = A_new
```

### Convergence

- Typically 20-50 iterations for 10k x 10k grids
- Returns the number of iterations taken
- Uses double-buffering (flow_acc, flow_acc_new)

### Donor Direction

Neighbor k donates to us via direction `(k+4) % 8` (opposite direction).

## Surface Routing

```python
route_surface_water(h, Z, flow_frac, mask, q_out, dx, dt, manning_n) -> float
```

Two-pass scheme for mass-conservative routing. Returns boundary outflow for mass balance.

### Pass 1: compute_outflow()

Computes outflow rate using kinematic wave velocity:

```
v = h^(2/3) * sqrt(S_max) / manning_n
q_out = min(h/dt, v * h / dx)
```

Where:
- S_max is the maximum slope among flow directions
- The min() enforces CFL stability (can't drain more than available)

Cells with h <= MIN_DEPTH (1e-8 m) have zero outflow.

### Pass 2: apply_fluxes()

Updates water depth from net flux:

```
h_new = h - q_out * dt + sum(inflow from donors)
```

Inflow from neighbor k: `flow_frac[k, donor_dir] * q_out[k] * dt`

Also tracks water leaving at boundaries (mask[neighbor]=0) and returns total boundary outflow.

### Why Two-Pass?

Pass 1 writes to q_out, Pass 2 reads q_out[neighbors]. No race condition because the stencil reads from a separately-computed field, not from h.

## CFL Timestep

```python
compute_cfl_timestep(h, Z, flow_frac, mask, dx, manning_n, cfl=0.5) -> float
```

Computes stable timestep for surface routing:

```
dt = cfl * dx / v_max
```

Where v_max is the maximum velocity over all cells:
```
v_max = max(h^(2/3) * sqrt(S) / manning_n)
```

Returns infinity if v_max ~ 0 (no significant flow).

### CFL Factor

Default cfl=0.5 provides safety margin. Lower values are more stable but require more substeps.

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| FLOW_EXPONENT | 1.5 | MFD concentration factor |
| MIN_DEPTH | 1e-8 m | Threshold for flow computation |
| MIN_SLOPE_SUM | 1e-10 | Flat cell detection threshold |

## Infiltration

Infiltration is in a separate file (`src/kernels/infiltration.py`) but closely coupled with flow:

```python
infiltration_step(h, M, P, mask, alpha, k_P, W_0, M_sat, dt) -> float
```

Transfers water from surface (h) to soil (M) with vegetation enhancement. Returns total infiltrated volume.

See `physics.md` for the infiltration equation.

## Tests Required

1. **Flow direction:** Tilted plane -> all flow in one direction
2. **Symmetry:** Symmetric valley -> flow splits equally at ridge
3. **Flat handling:** Flat terrain -> fractions flagged correctly
4. **Conservation:** Closed domain -> total h unchanged by routing
5. **CFL stability:** Large timesteps don't cause negative water
6. **Boundary outflow:** Water exits correctly at domain edges
