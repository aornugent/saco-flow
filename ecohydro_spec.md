# EcoHydro: Algorithmic Specification for GPU-Accelerated Semiarid Vegetation Dynamics

## 1. Design Philosophy

**Core Principle**: Express the essential physics of vegetation-water feedback in the simplest form that produces emergent patterning, then optimize for GPU execution.

The simulation captures the **Turing-type instability** that drives vegetation pattern formation: plants enhance local infiltration, reducing runoff that would otherwise subsidize downslope neighbors. This positive local feedback combined with negative nonlocal feedback (competition for water via runoff redistribution) generates bands, spots, and labyrinths.

### Design Constraints
- **Single GB200**: 192GB HBM3e, ~2.5 PFLOPS FP32, 8 TB/s memory bandwidth
- **Target scale**: 10⁸ cells (10,000 × 10,000 at 10m resolution = 100km × 100km)
- **Target duration**: Decades of simulated time in hours of wall time
- **Physical realism**: Mass-conserving, produces correct pattern wavelengths and morphologies
- **Real DEMs**: Irregular boundaries, depressions, ridges, variable slopes

---

## 2. Mathematical Formulation

### 2.1 State Variables

Three coupled fields defined on a regular grid:

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| $h$ | Surface water depth | m | Ponded water available for infiltration and routing |
| $M$ | Soil moisture | m | Plant-available water in root zone (depth equivalent) |
| $P$ | Vegetation biomass | kg/m² | Above-ground plant biomass density |

Plus static/slowly-varying fields:

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| $Z$ | Elevation | m | DEM surface elevation |
| $\Omega$ | Domain mask | binary | 1 = active cell, 0 = boundary/inactive |

### 2.2 Governing Equations

**Surface water balance** (event timescale):
$$\frac{\partial h}{\partial t} = R - I(h, P, M) - \nabla \cdot \mathbf{q}_s$$

**Soil moisture dynamics** (daily timescale):
$$\frac{\partial M}{\partial t} = I(h, P, M) - E(M, P) - L(M) + D_M \nabla^2 M$$

**Vegetation dynamics** (seasonal timescale):
$$\frac{\partial P}{\partial t} = G(M) \cdot P - \mu P + D_P \nabla^2 P$$

### 2.3 Constitutive Relations

**Infiltration with vegetation feedback** (the critical nonlinearity):
$$I(h, P, M) = \alpha \cdot h \cdot \frac{P + k_P \cdot W_0}{P + k_P} \cdot \left(1 - \frac{M}{M_{sat}}\right)^+$$

This encodes:
- Infiltration proportional to available surface water $h$
- Vegetation enhancement: bare soil infiltrates at rate $\alpha W_0$, dense vegetation at rate $\alpha$
- Saturation limitation: no infiltration when soil saturated

**Surface water flux** (kinematic wave with MFD partitioning):
$$\mathbf{q}_s = h^{5/3} \cdot \frac{\sqrt{|\nabla Z|}}{n} \cdot \hat{\mathbf{s}}$$

where $\hat{\mathbf{s}}$ is the unit vector in steepest descent direction and $n$ is Manning's roughness.

**Evapotranspiration**:
$$E(M, P) = E_{max} \cdot \frac{M}{M + k_M} \cdot (1 + \beta_E \cdot P)$$

**Deep leakage**:
$$L(M) = L_{max} \cdot \left(\frac{M}{M_{sat}}\right)^2$$

**Vegetation growth**:
$$G(M) = g_{max} \cdot \frac{M}{M + k_G}$$

### 2.4 Parameter Table

| Parameter | Symbol | Typical Value | Units |
|-----------|--------|---------------|-------|
| Infiltration rate | $\alpha$ | 0.1 | day⁻¹ |
| Vegetation feedback half-sat | $k_P$ | 1.0 | kg/m² |
| Bare soil infiltration fraction | $W_0$ | 0.1 | - |
| Soil saturation capacity | $M_{sat}$ | 0.3 | m |
| Max ET rate | $E_{max}$ | 0.005 | m/day |
| ET moisture half-sat | $k_M$ | 0.05 | m |
| Vegetation ET enhancement | $\beta_E$ | 0.5 | m²/kg |
| Max leakage rate | $L_{max}$ | 0.001 | m/day |
| Max growth rate | $g_{max}$ | 0.02 | day⁻¹ |
| Growth moisture half-sat | $k_G$ | 0.1 | m |
| Mortality rate | $\mu$ | 0.001 | day⁻¹ |
| Soil moisture diffusivity | $D_M$ | 0.1 | m²/day |
| Seed dispersal diffusivity | $D_P$ | 0.01 | m²/day |
| Manning's n | $n$ | 0.03 | s/m^(1/3) |

---

## 3. Discretization Strategy

### 3.1 Spatial Discretization

**Grid**: Regular Cartesian with spacing $\Delta x = \Delta y = dx$

**Stencil operations**: 5-point Laplacian for diffusion
$$\nabla^2 u \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{dx^2}$$

**Flow directions**: 8-connected MFD (Multiple Flow Direction)

For cell $(i,j)$, compute outflow fractions to each lower neighbor $k \in \mathcal{N}^-_{i,j}$:
$$f_k = \frac{(\max(0, S_k))^p}{\sum_{m \in \mathcal{N}^-} (\max(0, S_m))^p}$$

where $S_k = (Z_{i,j} - Z_k) / d_k$ is the slope to neighbor $k$, $d_k$ is the distance (1 or $\sqrt{2}$ times $dx$), and $p \in [1, 10]$ controls flow concentration (higher = more channelized).

For diffuse hillslope flow in semiarid systems, $p = 1$ to $1.5$ is appropriate.

### 3.2 Temporal Discretization: Hierarchical Operator Splitting

The system spans timescales from minutes (surface flow) to years (vegetation). We use **Lie-Trotter splitting** with nested subcycling:

```
For each VEGETATION timestep Δt_V (7-30 days):
    For each SOIL timestep Δt_M (1 day):
        For each SURFACE timestep Δt_h (adaptive, ~minutes):
            1. Route surface water
            2. Compute infiltration
            3. Update h, M
        End surface loop
        4. Apply ET and leakage to M
        5. Diffuse soil moisture
    End soil loop
    6. Update vegetation (growth - mortality)
    7. Diffuse vegetation (seed dispersal)
End vegetation loop
```

**Timestep constraints**:
- Surface: CFL condition $\Delta t_h \leq \frac{dx}{v_{max}}$ where $v_{max} \approx \frac{h^{2/3}\sqrt{S}}{n}$
- Soil diffusion: $\Delta t_M \leq \frac{dx^2}{4 D_M}$ (stability)
- Vegetation diffusion: $\Delta t_V \leq \frac{dx^2}{4 D_P}$ (stability, but vegetation is slow anyway)

### 3.3 Flow Accumulation: Iterative Parallel Algorithm

The key algorithmic challenge is computing flow accumulation in parallel. We use an **iterative relaxation** approach that converges in $O(\log n)$ iterations for typical terrain:

```
Initialize: A[i,j] = local_runoff[i,j]  # Contributing area starts with local input

Repeat until converged:
    A_new[i,j] = local_runoff[i,j] + Σ_{k ∈ donors(i,j)} f_{k→(i,j)} · A[k]
    
    converged = max|A_new - A| < ε
    A = A_new
```

This parallelizes trivially (each cell reads from neighbors) and converges quickly on non-pathological terrain.

**Key insight**: We don't need exact steady-state flow—we need *physically reasonable* flow that conserves mass. Running a fixed number of iterations (proportional to longest flow path / parallelism) is sufficient.

---

## 4. Data Structures and Memory Layout

### 4.1 Field Storage

All fields stored as **Structure of Arrays (SoA)** for coalesced GPU memory access:

```python
# Primary state (read-write every timestep)
h: ti.field(dtype=ti.f32, shape=(N, N))      # Surface water
M: ti.field(dtype=ti.f32, shape=(N, N))      # Soil moisture  
P: ti.field(dtype=ti.f32, shape=(N, N))      # Vegetation

# Derived fields (updated as needed)
flow_acc: ti.field(dtype=ti.f32, shape=(N, N))    # Flow accumulation
q_out: ti.field(dtype=ti.f32, shape=(N, N))       # Total outflow from cell

# Static fields (read-only during simulation)
Z: ti.field(dtype=ti.f32, shape=(N, N))           # Elevation
mask: ti.field(dtype=ti.i8, shape=(N, N))         # Domain mask

# MFD flow partitioning (8 neighbors)
flow_frac: ti.field(dtype=ti.f32, shape=(N, N, 8))  # Outflow fractions
```

### 4.2 Memory Budget (10⁸ cells)

| Field | Size | Count | Total |
|-------|------|-------|-------|
| Primary state (h, M, P) | 4 bytes | 3 | 1.2 GB |
| Flow fractions | 4 bytes | 8 | 3.2 GB |
| Derived fields | 4 bytes | 4 | 1.6 GB |
| Static fields | 4 bytes | 2 | 0.8 GB |
| Working buffers | 4 bytes | 4 | 1.6 GB |
| **Total** | | | **~8.5 GB** |

This is **<5% of GB200's 192GB HBM**, leaving ample room for:
- Double buffering
- Larger domains (up to 10⁹ cells feasible)
- Additional physics (e.g., soil layers)

### 4.3 Neighbor Indexing Convention

8-connected neighbors indexed 0-7, clockwise from East:

```
    5  6  7
    4  X  0
    3  2  1

neighbor_di = [0, 1, 1, 1, 0, -1, -1, -1]  # row offset
neighbor_dj = [1, 1, 0, -1, -1, -1, 0, 1]  # col offset
neighbor_dist = [1, √2, 1, √2, 1, √2, 1, √2]  # distance in cell units
```

---

## 5. Kernel Specifications

### 5.1 Kernel: Compute Flow Directions (run once, or when Z changes)

```python
@ti.kernel
def compute_flow_directions():
    """
    Compute MFD flow fractions for each cell.
    Each cell distributes outflow to lower neighbors proportional to slope^p.
    """
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
            
        z_center = Z[i, j]
        slope_sum = 0.0
        slopes = ti.Vector([0.0] * 8)
        
        # Compute slopes to all 8 neighbors
        for k in ti.static(range(8)):
            ni = i + neighbor_di[k]
            nj = j + neighbor_dj[k]
            
            if mask[ni, nj] == 1:
                dz = z_center - Z[ni, nj]
                if dz > 0:  # Downslope neighbor
                    slope = dz / (neighbor_dist[k] * dx)
                    slopes[k] = ti.pow(slope, flow_exponent)
                    slope_sum += slopes[k]
        
        # Normalize to fractions
        if slope_sum > 1e-10:
            for k in ti.static(range(8)):
                flow_frac[i, j, k] = slopes[k] / slope_sum
        else:
            # Flat cell or local minimum - mark for special handling
            flow_frac[i, j, 0] = -1.0  # Flag
```

**Complexity**: O(1) per cell, embarrassingly parallel
**Memory access**: 9 reads (center + 8 neighbors), 8 writes

### 5.2 Kernel: Flow Accumulation Iteration

```python
@ti.kernel
def flow_accumulation_step() -> ti.f32:
    """
    One iteration of parallel flow accumulation.
    Returns maximum change for convergence check.
    """
    max_change = 0.0
    
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        # Start with local contribution (runoff generated at this cell)
        acc_new = h[i, j]  # Or: rainfall - infiltration
        
        # Add contributions from upslope donors
        for k in ti.static(range(8)):
            # Reverse direction: neighbor k donates via direction (k+4)%8
            ni = i + neighbor_di[k]
            nj = j + neighbor_dj[k]
            
            if mask[ni, nj] == 1:
                donor_dir = (k + 4) % 8  # Direction from neighbor to us
                acc_new += flow_frac[ni, nj, donor_dir] * flow_acc[ni, nj]
        
        change = ti.abs(acc_new - flow_acc[i, j])
        ti.atomic_max(max_change, change)
        flow_acc_new[i, j] = acc_new
    
    return max_change

@ti.kernel
def swap_flow_buffers():
    for i, j in flow_acc:
        flow_acc[i, j] = flow_acc_new[i, j]
```

**Convergence**: Typically 20-50 iterations for 10k×10k grid
**Can be replaced** with FastFlow-style rake-compress for O(log n) guarantee

### 5.3 Kernel: Infiltration and Surface Water Update

```python
@ti.kernel
def infiltration_step(dt: ti.f32):
    """
    Compute infiltration from ponded water to soil.
    Updates both h and M fields.
    """
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        h_local = h[i, j]
        M_local = M[i, j]
        P_local = P[i, j]
        
        if h_local > 0 and M_local < M_sat:
            # Vegetation-enhanced infiltration rate
            veg_factor = (P_local + k_P * W_0) / (P_local + k_P)
            sat_factor = max(0.0, 1.0 - M_local / M_sat)
            
            # Infiltration amount (limited by available water and capacity)
            I_potential = alpha * h_local * veg_factor * sat_factor * dt
            I_actual = min(I_potential, h_local, M_sat - M_local)
            
            # Update fields
            h[i, j] = h_local - I_actual
            M[i, j] = M_local + I_actual
```

### 5.4 Kernel: Surface Water Routing (Explicit)

```python
@ti.kernel
def route_surface_water(dt: ti.f32):
    """
    Route surface water using MFD fractions.
    Explicit donor-based scheme for mass conservation.
    """
    # First pass: compute outflows
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        h_local = h[i, j]
        if h_local > h_min:  # Threshold for flow
            # Kinematic wave velocity
            # v = h^(2/3) * sqrt(S) / n
            # For simplicity, use max slope
            S_max = 0.0
            for k in ti.static(range(8)):
                if flow_frac[i, j, k] > 0:
                    ni, nj = i + neighbor_di[k], j + neighbor_dj[k]
                    S = (Z[i,j] - Z[ni, nj]) / (neighbor_dist[k] * dx)
                    S_max = max(S_max, S)
            
            v = ti.pow(h_local, 2.0/3.0) * ti.sqrt(S_max) / manning_n
            q_out[i, j] = min(h_local / dt, v * h_local / dx)  # Limit by CFL
        else:
            q_out[i, j] = 0.0
    
    # Second pass: apply fluxes
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        # Outflow
        delta_h = -q_out[i, j] * dt
        
        # Inflow from neighbors
        for k in ti.static(range(8)):
            ni = i + neighbor_di[k]
            nj = j + neighbor_dj[k]
            if mask[ni, nj] == 1:
                donor_dir = (k + 4) % 8
                delta_h += flow_frac[ni, nj, donor_dir] * q_out[ni, nj] * dt
        
        h[i, j] = max(0.0, h[i, j] + delta_h)
```

### 5.5 Kernel: Soil Moisture Dynamics

```python
@ti.kernel
def soil_moisture_step(dt: ti.f32):
    """
    Update soil moisture: ET, leakage, and lateral diffusion.
    Uses Laplacian with Neumann BCs at domain boundary.
    """
    ti.block_local(M)  # Cache in shared memory for stencil
    
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        M_local = M[i, j]
        P_local = P[i, j]
        
        # Evapotranspiration
        ET = E_max * M_local / (M_local + k_M) * (1.0 + beta_E * P_local)
        
        # Deep leakage
        leakage = L_max * (M_local / M_sat) ** 2
        
        # Lateral diffusion (5-point Laplacian)
        laplacian = 0.0
        count = 0
        for di, dj in ti.static([(-1,0), (1,0), (0,-1), (0,1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += M[ni, nj] - M_local
                count += 1
        laplacian /= (dx * dx)
        
        # Update
        dM = (-ET - leakage + D_M * laplacian) * dt
        M_new[i, j] = max(0.0, min(M_sat, M_local + dM))

@ti.kernel
def swap_M_buffers():
    for i, j in M:
        M[i, j] = M_new[i, j]
```

### 5.6 Kernel: Vegetation Dynamics

```python
@ti.kernel
def vegetation_step(dt: ti.f32):
    """
    Update vegetation: growth, mortality, and seed dispersal.
    """
    ti.block_local(P)
    
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        P_local = P[i, j]
        M_local = M[i, j]
        
        # Growth (Monod kinetics)
        growth = g_max * M_local / (M_local + k_G) * P_local
        
        # Mortality
        mortality = mu * P_local
        
        # Seed dispersal (diffusion)
        laplacian = 0.0
        for di, dj in ti.static([(-1,0), (1,0), (0,-1), (0,1)]):
            ni, nj = i + di, j + dj
            if mask[ni, nj] == 1:
                laplacian += P[ni, nj] - P_local
        laplacian /= (dx * dx)
        
        # Update with positivity constraint
        dP = (growth - mortality + D_P * laplacian) * dt
        P_new[i, j] = max(0.0, P_local + dP)

@ti.kernel
def swap_P_buffers():
    for i, j in P:
        P[i, j] = P_new[i, j]
```

---

## 6. Rainfall Event Handling

### 6.1 Event-Based Precipitation

Rather than continuous low-intensity rain, semiarid systems experience discrete high-intensity events:

```python
@ti.kernel
def apply_rainfall(intensity: ti.f32, dt: ti.f32):
    """
    Apply uniform rainfall to all active cells.
    intensity: m/s (or use m/event for pulse)
    """
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 1:
            h[i, j] += intensity * dt
```

### 6.2 Simulation Structure

```python
def simulate(years: int, events_per_year: int = 20):
    """
    Main simulation loop with episodic rainfall.
    """
    days_per_year = 365
    days_between_events = days_per_year / events_per_year
    
    current_day = 0.0
    next_event_day = random_exponential(days_between_events)
    
    while current_day < years * days_per_year:
        # Check for rainfall event
        if current_day >= next_event_day:
            event_depth = random_gamma(mean=0.02, shape=2)  # ~20mm mean
            event_duration = random_uniform(0.1, 0.5)  # days
            
            # Run surface water dynamics during event
            run_rainfall_event(event_depth, event_duration)
            
            next_event_day += random_exponential(days_between_events)
        
        # Daily soil and vegetation update
        soil_moisture_step(dt=1.0)
        
        # Weekly vegetation update
        if int(current_day) % 7 == 0:
            vegetation_step(dt=7.0)
        
        current_day += 1.0
        
        # Periodic output
        if int(current_day) % 30 == 0:
            save_state(current_day)
```

---

## 7. Boundary Conditions

### 7.1 Domain Mask Handling

The mask field encodes:
- `1`: Active cell (interior)
- `0`: Inactive/boundary (no flux, no update)

Boundary cells act as **no-flux (Neumann)** for diffusion and **outflow** for surface routing.

### 7.2 Depression Handling

For real DEMs with depressions, two options:

**Option A: Pre-fill depressions** (simple)
- Use Priority-Flood algorithm to create a "flow-routing surface"
- Store original DEM for visualization, filled DEM for routing

**Option B: Dynamic ponding** (realistic)
- Allow water to accumulate in depressions
- Overflow when water level exceeds pour point
- More complex but captures real hydrology

For vegetation patterns, Option A is usually sufficient.

### 7.3 Outlet Boundary

Identify outlet cells (boundary cells with inward slope) and remove water:

```python
@ti.kernel
def apply_outlet_boundary():
    for i, j in ti.ndrange(N, N):
        if is_boundary[i, j] and h[i, j] > 0:
            # Check if flow would exit domain
            for k in ti.static(range(8)):
                ni, nj = i + neighbor_di[k], j + neighbor_dj[k]
                if not in_bounds(ni, nj) or mask[ni, nj] == 0:
                    if flow_frac[i, j, k] > 0:
                        h[i, j] = 0  # Water exits system
                        break
```

---

## 8. Mass Conservation Verification

### 8.1 Conservation Law

Total water in system must satisfy:
$$\frac{d}{dt}(H_{total} + M_{total}) = R_{total} - ET_{total} - L_{total} - Q_{out}$$

where:
- $H_{total} = \sum_{i,j} h_{i,j} \cdot dx^2$
- $M_{total} = \sum_{i,j} M_{i,j} \cdot dx^2$
- $Q_{out}$ = water exiting at boundaries

### 8.2 Diagnostic Kernel

```python
@ti.kernel
def compute_water_balance() -> ti.f32:
    total_h = 0.0
    total_M = 0.0
    
    for i, j in ti.ndrange(N, N):
        if mask[i, j] == 1:
            total_h += h[i, j]
            total_M += M[i, j]
    
    return (total_h + total_M) * dx * dx

# Track cumulative fluxes
cumulative_rain = 0.0
cumulative_ET = 0.0
cumulative_leakage = 0.0
cumulative_outflow = 0.0

# Check conservation each timestep
expected = initial_water + cumulative_rain - cumulative_ET - cumulative_leakage - cumulative_outflow
actual = compute_water_balance()
error = abs(expected - actual) / max(expected, 1e-10)
assert error < 1e-6, f"Mass conservation violated: {error:.2e}"
```

---

## 9. Performance Optimization

### 9.1 Kernel Fusion

Combine operations that access the same data:

```python
@ti.kernel
def fused_soil_update(dt: ti.f32):
    """
    Fused kernel: infiltration + ET + leakage + diffusion
    Single pass over M field with shared memory caching
    """
    ti.block_local(M, h, P)
    
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue
        
        # Load all needed values once
        M_c = M[i, j]
        h_c = h[i, j]
        P_c = P[i, j]
        M_neighbors = [M[i+di, j+dj] for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]]
        
        # Infiltration
        I = compute_infiltration(h_c, M_c, P_c, dt)
        
        # ET and leakage
        ET = compute_ET(M_c, P_c, dt)
        L = compute_leakage(M_c, dt)
        
        # Diffusion
        laplacian = sum(M_neighbors) - 4*M_c
        diff = D_M * laplacian / (dx*dx) * dt
        
        # Single write
        M_new[i, j] = clamp(M_c + I - ET - L + diff, 0, M_sat)
        h_new[i, j] = max(0, h_c - I)
```

### 9.2 Temporal Blocking for Diffusion

For pure diffusion, multiple timesteps can be fused:

```python
@ti.kernel
def diffusion_temporal_block(field: ti.template(), D: ti.f32, dt: ti.f32, steps: int):
    """
    Apply multiple diffusion steps in shared memory before writing to global.
    Reduces memory bandwidth by factor of ~steps.
    """
    ti.block_local(field)
    
    for i, j in ti.ndrange((BLOCK, N-BLOCK), (BLOCK, N-BLOCK)):
        # Load block into shared memory
        local = field[i, j]
        
        for _ in range(steps):
            laplacian = # compute from neighbors in shared memory
            local += D * laplacian / (dx*dx) * dt
            ti.simt.block.sync()  # Synchronize threads in block
        
        field[i, j] = local
```

### 9.3 Adaptive Subcycling

Surface water routing needs small timesteps only when water is present:

```python
def run_rainfall_event(depth: float, duration: float):
    """
    Subcycle surface dynamics only during active flow.
    """
    apply_rainfall(depth / duration, duration)
    
    # Adaptive subcycling
    dt_surface = compute_CFL_timestep()
    t = 0.0
    t_end = duration + 1.0  # Extra time for drainage
    
    while t < t_end and max_h() > h_threshold:
        dt = min(dt_surface, t_end - t)
        
        route_surface_water(dt)
        infiltration_step(dt)
        
        t += dt
        dt_surface = compute_CFL_timestep()  # Update based on current h
```

### 9.4 Memory Access Pattern

Ensure coalesced access by iterating over contiguous memory:

```python
# GOOD: Threads in a warp access consecutive j values
for i, j in ti.ndrange(N, N):  # j is innermost, contiguous in memory

# BAD: Non-coalesced access
for j, i in ti.ndrange(N, N):  # i is innermost but j varies first
```

---

## 10. Output and Visualization

### 10.1 State Snapshots

```python
def save_state(day: float):
    """Save current state to disk in GeoTIFF format."""
    import rasterio
    
    profile = base_profile.copy()
    
    with rasterio.open(f'output/h_day{int(day):06d}.tif', 'w', **profile) as dst:
        dst.write(h.to_numpy(), 1)
    
    with rasterio.open(f'output/M_day{int(day):06d}.tif', 'w', **profile) as dst:
        dst.write(M.to_numpy(), 1)
    
    with rasterio.open(f'output/P_day{int(day):06d}.tif', 'w', **profile) as dst:
        dst.write(P.to_numpy(), 1)
```

### 10.2 Diagnostic Outputs

- Pattern wavelength (via FFT of P field)
- Vegetation cover fraction
- Water balance components
- Flow network visualization

---

## 11. Initialization

### 11.1 From Real DEM

```python
def initialize_from_dem(dem_path: str):
    """
    Load DEM and initialize all fields.
    """
    import rasterio
    
    with rasterio.open(dem_path) as src:
        Z_np = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Set elevation
    Z.from_numpy(Z_np)
    
    # Create mask (valid data)
    mask_np = np.where(np.isnan(Z_np) | (Z_np < -1e10), 0, 1).astype(np.int8)
    mask.from_numpy(mask_np)
    
    # Fill depressions for flow routing
    Z_filled = priority_flood_fill(Z_np, mask_np)
    Z.from_numpy(Z_filled)
    
    # Compute flow directions
    compute_flow_directions()
    
    # Initialize state
    h.fill(0.0)
    M.fill(0.1)  # Start with some soil moisture
    P.from_numpy(np.random.uniform(0.1, 0.5, (N, N)) * mask_np)  # Random initial vegetation
```

### 11.2 Spinup Protocol

```python
def spinup(years: int = 50):
    """
    Run simulation to reach quasi-steady vegetation pattern.
    """
    print("Spinning up vegetation pattern...")
    
    for year in range(years):
        simulate_one_year()
        
        # Check for pattern stability
        if year > 10:
            P_mean_old, P_mean_new = P_history[-2], P_history[-1]
            if abs(P_mean_new - P_mean_old) / P_mean_old < 0.01:
                print(f"Pattern stabilized at year {year}")
                break
        
        print(f"Year {year}: mean P = {P.to_numpy().mean():.3f}")
```

---

## 12. Complete Algorithm Summary

```
ECOHYDRO SIMULATION ALGORITHM
=============================

INITIALIZATION:
    Load DEM → Z
    Create domain mask → Ω
    Fill depressions → Z_filled
    Compute MFD flow directions → flow_frac[i,j,k]
    Initialize: h=0, M=M_init, P=P_init

MAIN LOOP (for each simulated day):
    
    IF rainfall event scheduled:
        
        SURFACE WATER ROUTING (subcycled):
            WHILE water present AND t < t_event_end:
                dt_s = CFL_timestep(h, Z)
                
                PARALLEL: route_surface_water(dt_s)
                    - Compute outflow per cell
                    - Distribute to downslope neighbors via MFD fractions
                
                PARALLEL: infiltration_step(dt_s)  
                    - I = α·h·[(P+k_P·W_0)/(P+k_P)]·(1-M/M_sat)
                    - h -= I, M += I
                
                t += dt_s
    
    SOIL MOISTURE UPDATE (daily):
        PARALLEL: soil_moisture_step(dt=1.0)
            - ET = E_max·M/(M+k_M)·(1+β_E·P)
            - L = L_max·(M/M_sat)²
            - ∇²M via 5-point stencil
            - M += (-ET - L + D_M·∇²M)·dt
    
    VEGETATION UPDATE (weekly):
        IF day % 7 == 0:
            PARALLEL: vegetation_step(dt=7.0)
                - G = g_max·M/(M+k_G)·P
                - mortality = μ·P
                - ∇²P via 5-point stencil
                - P += (G - mortality + D_P·∇²P)·dt

    OUTPUT (monthly):
        IF day % 30 == 0:
            save_state(day)
            verify_mass_conservation()

TERMINATION:
    Final output and diagnostics
```

---

## 13. Expected Performance

### 13.1 Computational Cost per Timestep

| Operation | FLOPS/cell | Memory/cell | Bottleneck |
|-----------|-----------|-------------|------------|
| Flow directions | ~100 | 36 bytes R, 32 W | Compute |
| Flow accumulation (iter) | ~50 | 40 bytes R, 4 W | Memory |
| Surface routing | ~80 | 48 bytes R, 8 W | Memory |
| Infiltration | ~30 | 12 bytes R, 8 W | Memory |
| Soil moisture | ~50 | 24 bytes R, 4 W | Memory |
| Vegetation | ~40 | 20 bytes R, 4 W | Memory |

### 13.2 Projected Wall Time (GB200)

For 10⁸ cells:
- Memory bandwidth limited: ~8 TB/s available
- Typical kernel: ~50 bytes/cell → ~160M cells/ms → **~0.6 ms per kernel**
- Surface substeps per event: ~100 → ~60 ms per event
- Daily operations: ~2 ms
- Events per year: ~20 → ~1.2 s per year

**Projected: ~10 simulated years per minute** → decade in ~1 minute, century in ~10 minutes

### 13.3 Scaling

- Linear in cell count (memory bandwidth limited)
- ~20 minutes for 10⁹ cells per decade
- Multi-GPU: Near-linear with domain decomposition (halo exchange minimal for local stencils)

---

## 14. Extensions (Future Work)

### 14.1 Physics Extensions
- Multiple soil layers
- Groundwater dynamics
- Multiple vegetation types / competition
- Fire disturbance
- Grazing pressure
- Topographic modification (erosion/deposition)

### 14.2 Algorithmic Extensions
- Adaptive mesh refinement near vegetation boundaries
- Implicit diffusion for stability at larger timesteps
- Ensemble simulations for uncertainty quantification
- Adjoint-based parameter estimation

### 14.3 Coupling
- Climate forcing from reanalysis data
- Integration with remote sensing (NDVI assimilation)
- Landscape evolution model coupling

---

## Appendix A: Taichi Implementation Skeleton

```python
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, default_fp=ti.f32)

# Grid parameters
N = 10000
dx = 10.0  # meters

# Physical parameters (as Taichi constants for kernel inlining)
alpha = ti.field(dtype=ti.f32, shape=())
k_P = ti.field(dtype=ti.f32, shape=())
W_0 = ti.field(dtype=ti.f32, shape=())
# ... etc

# Fields
h = ti.field(dtype=ti.f32, shape=(N, N))
M = ti.field(dtype=ti.f32, shape=(N, N))
P = ti.field(dtype=ti.f32, shape=(N, N))
Z = ti.field(dtype=ti.f32, shape=(N, N))
mask = ti.field(dtype=ti.i8, shape=(N, N))
flow_frac = ti.field(dtype=ti.f32, shape=(N, N, 8))

# Double buffers for updates
h_new = ti.field(dtype=ti.f32, shape=(N, N))
M_new = ti.field(dtype=ti.f32, shape=(N, N))
P_new = ti.field(dtype=ti.f32, shape=(N, N))

# Neighbor offsets (compile-time constant)
neighbor_di = ti.Vector([0, 1, 1, 1, 0, -1, -1, -1])
neighbor_dj = ti.Vector([1, 1, 0, -1, -1, -1, 0, 1])
neighbor_dist = ti.Vector([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414])

# Include kernels from Section 5...

def main():
    # Initialize
    load_dem("terrain.tif")
    compute_flow_directions()
    
    # Run simulation
    for year in range(100):
        simulate_year()
        print(f"Year {year}: P_mean = {P.to_numpy().mean():.4f}")
    
    # Save final state
    save_state(100 * 365)

if __name__ == "__main__":
    main()
```

---

## Appendix B: Key References

1. **Rietkerk et al. (2002)** - Self-organized patchiness and catastrophic shifts in ecosystems
2. **Saco et al. (2007)** - Eco-geomorphology of banded vegetation patterns
3. **HilleRisLambers et al. (2001)** - Vegetation pattern formation in semi-arid grazing systems
4. **Tarboton (1997)** - D-infinity flow direction algorithm
5. **Barnes (2016-2019)** - Parallel flow algorithms
6. **Jain et al. (2024)** - FastFlow GPU acceleration

---

*Document Version: 1.0*
*Target Platform: NVIDIA GB200 with Taichi*
*Scale: 10⁸ cells, decadal simulations*
