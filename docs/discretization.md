# Discretization

Spatial and temporal numerical methods, boundary conditions, and rainfall handling.

## Spatial Discretization

### Grid

Regular Cartesian grid with uniform spacing:
- dx = dy (square cells)
- Row-major storage for coalesced GPU memory access
- Indexed as field[i, j] where i = row, j = column

### 5-Point Laplacian (Diffusion)

For soil moisture and vegetation diffusion:

```
laplacian(u) = (u_E + u_W + u_N + u_S - 4*u_center) / dx^2
```

Where E/W/N/S are the four cardinal neighbors.

**Neumann boundary condition:** Only include neighbors where mask=1. This naturally enforces zero gradient (no-flux) at domain boundaries.

Source: `src/geometry.py:laplacian_diffusion_step()`

### 8-Connected Neighbors (Flow Routing)

For surface water routing, 8 neighbors are used (D8 connectivity):

```
Index:  5  6  7
        4  X  0
        3  2  1
```

Indexed clockwise from East:
- 0 = East (+j)
- 1 = Southeast (+i, +j)
- 2 = South (+i)
- 3 = Southwest (+i, -j)
- 4 = West (-j)
- 5 = Northwest (-i, -j)
- 6 = North (-i)
- 7 = Northeast (-i, +j)

**Distances:**
- Cardinal (0, 2, 4, 6): 1.0 * dx
- Diagonal (1, 3, 5, 7): sqrt(2) * dx

**Reverse direction:** `(k + 4) % 8` gives the opposite direction.

Source: `src/geometry.py` (NEIGHBOR_DI, NEIGHBOR_DJ, NEIGHBOR_DIST)

### Multiple Flow Direction (MFD)

Flow is distributed to all lower neighbors proportional to slope raised to power p:

```
f_k = max(0, S_k)^p / sum(max(0, S_m)^p)
```

Where:
- S_k = (Z_center - Z_neighbor) / d_k is the slope to neighbor k
- d_k is the distance to neighbor k
- p is the flow exponent

**Flow exponent (p) controls concentration:**
- p = 1.0: Most diffuse (linear weighting)
- p = 1.5: Default for hillslope flow (FLOW_EXPONENT in code)
- p > 5: Approaches single-direction D8

**Flat cells:** When no neighbor is lower (slope_sum < MIN_SLOPE_SUM = 1e-10), the cell is flagged with flow_frac[i,j,0] = -1 for special handling.

Source: `src/kernels/flow.py:compute_flow_directions()`

## Temporal Discretization

### Hierarchical Operator Splitting

The system is split by timescale using Lie-Trotter operator splitting:

```
VEGETATION timestep (dt_veg = 7 days):
    SOIL timestep (dt_soil = 1 day):
        SURFACE timestep (adaptive, CFL-limited):
            1. Route surface water
            2. Compute infiltration
            3. Update h, M
        4. Apply ET and leakage to M
        5. Diffuse soil moisture
    6. Update vegetation (growth, mortality)
    7. Diffuse vegetation (dispersal)
```

### Stability Constraints

| Process | Constraint | Typical Value (days) | Notes |
|---------|------------|----------------------|-------|
| Surface routing | dt <= CFL * dx / v_max | 0.0001 - 0.01 | Only active constraint |
| Soil diffusion | dt <= 0.25 * dx^2 / D_M | ~2.5 | Usually satisfied by dt_soil |
| Vegetation diffusion | dt <= 0.25 * dx^2 / D_P | ~25 | Usually satisfied by dt_veg |

Only surface routing requires adaptive subcycling in practice.

### CFL Timestep Calculation

For surface water routing:

```
dt = CFL * dx / v_max
```

Where:
- CFL = 0.5 (default safety factor in code)
- v_max = max over all cells of: h^(2/3) * sqrt(S) / manning_n

Returns infinity if v_max ≈ 0 (no significant flow).

Source: `src/kernels/flow.py:compute_cfl_timestep()`

### Diffusion Stability

For 2D explicit diffusion, stability requires:

```
dt <= CFL * dx^2 / D
```

Where CFL <= 0.25 for the 5-point stencil.

Source: `src/kernels/soil.py:compute_diffusion_timestep()`

## Simulation Loop Structure

From `src/simulation.py:run()`:

1. **Rainfall scheduling:** Events arrive via Poisson process
   - Inter-event times: `np.random.exponential(interstorm)`
   - Approximately `365 / interstorm` events per year

2. **During rainfall event:** Adaptive subcycling
   - Maximum 10,000 substeps per event (safety limit)
   - Continue while h > h_threshold AND t < duration + drainage_time
   - Each substep: route water, compute infiltration

3. **Daily update:** Soil moisture
   - ET and leakage (fused kernel)
   - Lateral diffusion

4. **Weekly update:** Vegetation (every 7 days)
   - Growth and mortality (fused kernel)
   - Seed dispersal

5. **Periodic check:** Mass balance verification every 30 days

## Boundary Conditions

### Domain Mask

```
mask[i,j] = 1  ->  Active cell (interior)
mask[i,j] = 0  ->  Inactive/boundary (no flux, no update)
```

The mask is initialized in `src/initialization.py`:
- Edges (i=0, i=n-1, j=0, j=n-1) are set to 0
- Interior cells are set to 1

**Rule:** Always check mask before updating or reading neighbor values.

### Diffusion: Neumann (No-Flux)

For soil moisture and vegetation diffusion:
- Only include neighbors where mask=1 in Laplacian computation
- This naturally enforces zero gradient at boundaries
- Water/biomass cannot flow out of the domain via diffusion

### Surface Water: Outflow at Boundaries

Water can exit the domain at boundary cells:
- If a cell has flow_frac[i,j,k] > 0 pointing to a cell where mask=0
- That fraction of outflow leaves the system
- Boundary outflow is tracked and returned for mass balance

Source: `src/kernels/flow.py:apply_fluxes()`

### Depression Handling

Current implementation allows water to accumulate in local minima (depressions):
- Cells with no downslope neighbors are flagged (flow_frac[i,j,0] = -1)
- Water accumulates until it exceeds the pour point

**Future option:** Priority-Flood algorithm to pre-fill depressions and create a flow-routing surface.

## Rainfall Event Handling

From `src/simulation.py:run_rainfall_event()`:

```
intensity = depth / duration  [m/day]
t = 0

while t < duration + drainage_time and max(h) > h_threshold:
    dt = compute_cfl_timestep()
    dt = min(dt, remaining_time, 0.1)  # Cap at 0.1 days

    if t < duration:
        h += intensity * dt  # Apply rainfall

    route_surface_water()   # Two-pass scheme
    infiltration_step()     # Transfer h -> M

    t += dt
```

### Key Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| h_threshold | 1e-6 m | params.py | Minimum water depth to continue routing |
| drainage_time | 1.0 day | params.py | Extra time after event for drainage |
| MIN_DEPTH | 1e-8 m | flow.py | Threshold for flow velocity computation |
| MIN_SLOPE_SUM | 1e-10 | flow.py | Flat cell detection threshold |
| CFL (routing) | 0.5 | simulation.py | Safety factor for surface routing |
| CFL (diffusion) | 0.25 | soil.py | Safety factor for diffusion stability |
| max_subcycles | 10,000 | simulation.py | Safety limit per rainfall event |

## Flow Accumulation

Iterative parallel algorithm for computing contributing area:

```
Initialize: A[i,j] = local_source[i,j]

Repeat until converged:
    A_new[i,j] = local[i,j] + sum(f_{k->here} * A[k]) for all donors k
    converged = max|A_new - A| < tolerance
    A = A_new
```

- Typically converges in 20-50 iterations for 10k x 10k grids
- Convergence time proportional to longest flow path
- Perfect convergence not required—physically reasonable flow is sufficient

Source: `src/kernels/flow.py:compute_flow_accumulation()`
