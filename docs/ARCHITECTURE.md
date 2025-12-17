# EcoHydro Architecture

Simple architecture for a ~2000 line ecohydrological simulation. Add abstraction only when you feel pain.

## Design Principles

1. **Simplicity first** — Flat structure, minimal indirection
2. **Correctness before speed** — Every kernel needs a mass conservation test
3. **Explicit over implicit** — Physical variable names, units in comments
4. **No premature abstraction** — 6 fields don't need a FieldContainer class

## Module Structure

```
src/
├── geometry.py      # Grid constants, neighbor offsets
├── params.py        # Parameter dataclass + validation
├── fields.py        # All Taichi fields, swap_buffers(), initialization
├── kernels/         # Taichi kernels (one file per process)
│   ├── flow.py
│   ├── infiltration.py
│   ├── soil.py
│   └── vegetation.py
├── simulation.py    # Main loop, operator splitting
└── diagnostics.py   # Conservation check, timing utilities
```

## Key Components

### 1. Geometry (`geometry.py`)

Centralized neighbor indexing prevents subtle bugs and enables `ti.static()` unrolling.

```python
# Compile-time constants for 8-neighbor connectivity (D8)
NEIGHBOR_DI = ti.Vector([-1, -1, -1, 0, 0, 1, 1, 1])
NEIGHBOR_DJ = ti.Vector([-1, 0, 1, -1, 1, -1, 0, 1])
NEIGHBOR_DIST = ti.Vector([1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414])

@ti.func
def for_each_neighbor(i: int, j: int):
    """Use with ti.static(range(8)) for compile-time unrolling."""
    for k in ti.static(range(8)):
        ni = i + NEIGHBOR_DI[k]
        nj = j + NEIGHBOR_DJ[k]
        # ... check bounds and mask before use
```

**Why centralize**: Neighbor indexing is used in every kernel. One place to get it right.

### 2. Parameters (`params.py`)

Simple dataclass with validation. Catches typos and invalid values at load time.

```python
@dataclass
class SimulationParams:
    # Grid
    nx: int
    ny: int
    dx: float  # [m] cell size

    # Infiltration
    alpha: float      # [1/m] sorptive number
    k_sat: float      # [m/s] saturated conductivity
    psi_sat: float    # [m] air-entry pressure

    # ... other parameters with units in comments

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.k_sat < 0:
            raise ValueError(f"k_sat must be non-negative, got {self.k_sat}")
        # ... other validations

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationParams":
        """Load parameters from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Why dataclass**: Self-documenting, IDE autocomplete, validation at load time. No `frozen=True` — keep it simple.

### 3. Fields (`fields.py`)

All Taichi fields declared in one place. Include a `swap_buffers()` helper.

```python
import taichi as ti

# State fields (double-buffered for stencil operations)
h = ti.field(ti.f32)       # [m] surface water depth
h_new = ti.field(ti.f32)   # buffer for updates
M = ti.field(ti.f32)       # [m³/m³] soil moisture
M_new = ti.field(ti.f32)
P = ti.field(ti.f32)       # [kg/m²] plant biomass
P_new = ti.field(ti.f32)

# Static fields (read-only during simulation)
Z = ti.field(ti.f32)       # [m] elevation
mask = ti.field(ti.i32)    # 1=active, 0=inactive
flow_frac = ti.field(ti.f32)  # D8 flow fractions

def allocate(nx: int, ny: int):
    """Allocate all fields for given grid size."""
    ti.root.dense(ti.ij, (nx, ny)).place(h, h_new, M, M_new, P, P_new)
    ti.root.dense(ti.ij, (nx, ny)).place(Z, mask)
    ti.root.dense(ti.ijk, (nx, ny, 8)).place(flow_frac)

def swap_buffers():
    """Swap state buffers after stencil operations."""
    # Copy new -> current
    _copy_field(h_new, h)
    _copy_field(M_new, M)
    _copy_field(P_new, P)

@ti.kernel
def _copy_field(src: ti.template(), dst: ti.template()):
    for i, j in src:
        dst[i, j] = src[i, j]
```

**Why a swap helper**: Easy to forget which buffer is "current". A thin wrapper prevents bugs.

### 4. Kernels (`kernels/`)

One file per physical process. Follow the pattern:

```python
@ti.kernel
def update_soil_moisture(dt: ti.f32):
    """Update soil moisture. Physics: dM/dt = I - ET - L"""
    for i, j in ti.ndrange((1, N-1), (1, N-1)):
        if mask[i, j] == 0:
            continue

        # Compute fluxes (with units in comments)
        infiltration = ...   # [m³/m³/s]
        et = ...             # [m³/m³/s]
        leakage = ...        # [m³/m³/s]

        # Update with clamping to physical bounds
        dM = (infiltration - et - leakage) * dt
        M_new[i, j] = ti.max(0.0, ti.min(M_sat, M[i, j] + dM))
```

**Pattern**: Check mask → compute fluxes → clamp to bounds → write to buffer.

### 5. Simulation (`simulation.py`)

Main loop with operator splitting. Keep it flat.

```python
class Simulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.t = 0.0
        self.step_count = 0

        # Allocate fields
        fields.allocate(params.nx, params.ny)

        # Load initial conditions
        self._load_dem(params.dem_path)
        self._initialize_state()

    def step(self, dt: float):
        """One timestep with operator splitting."""
        # 1. Surface routing
        kernels.route_surface(dt)

        # 2. Infiltration
        kernels.infiltrate(dt)

        # 3. Soil moisture redistribution
        kernels.update_soil_moisture(dt)

        # 4. Vegetation dynamics
        kernels.update_vegetation(dt)

        # Swap buffers
        fields.swap_buffers()

        self.t += dt
        self.step_count += 1

    def run(self, duration: float, dt: float):
        """Run simulation for given duration."""
        while self.t < duration:
            self.step(dt)

            # Check conservation periodically
            if self.step_count % 100 == 0:
                diagnostics.check_conservation()
```

### 6. Diagnostics (`diagnostics.py`)

Simple functions for mass conservation and timing.

```python
def check_conservation(tolerance: float = 1e-6) -> bool:
    """Verify total mass is conserved within tolerance."""
    total_water = compute_total_water()
    total_biomass = compute_total_biomass()

    if abs(total_water - initial_water) > tolerance:
        print(f"Water conservation error: {total_water - initial_water}")
        return False
    return True

@ti.kernel
def compute_total_water() -> ti.f32:
    """Sum all water in system (surface + soil)."""
    total = 0.0
    for i, j in h:
        if mask[i, j] == 1:
            total += h[i, j] + M[i, j] * soil_depth
    return total
```

## What We Don't Need

| Component | Why Not |
|-----------|---------|
| `FieldSpec` + `FieldRole` enum | 6 fields don't need a type system |
| `FieldContainer` class | Taichi fields are already module-level singletons |
| `TaichiParams` class | Pass parameters directly to kernels |
| Protocol classes | Duck typing works; document signatures in docstrings |
| `SimulationState` dataclass | Instance variables on the runner are fine |
| Kernel registry | We're not swapping implementations at runtime |
| Callback system | Call diagnostics directly where needed |
| `MassTracker` class | A `check_conservation()` function is sufficient |
| Deep module nesting | 6-8 files total, not 20+ |

## Testing Strategy

1. **Unit tests**: Each kernel in isolation with synthetic inputs
2. **Conservation tests**: Verify mass balance on flat terrain
3. **Integration tests**: Full simulation with known analytical solutions

```python
def test_infiltration_conserves_mass():
    """Water leaving surface equals water entering soil."""
    initial_surface = compute_total_surface_water()
    initial_soil = compute_total_soil_water()

    kernels.infiltrate(dt=1.0)

    final_surface = compute_total_surface_water()
    final_soil = compute_total_soil_water()

    # Total should be unchanged
    assert abs((final_surface + final_soil) -
               (initial_surface + initial_soil)) < 1e-6
```

## When to Add Abstraction

Add abstraction only when you feel repeated pain:

- **Third time copying neighbor offsets** → Create `geometry.py`
- **Third validation bug from config** → Add `__post_init__` validation
- **Third buffer-swap bug** → Add `swap_buffers()` helper
- **Third conservation bug** → Add `check_conservation()` function

Don't add abstraction because "we might need it later."

## References

- `AGENTS.md` — Development conventions
- `docs/mass_conservation.md` — Conservation testing patterns
- `ecohydro_spec.md` — Physical equations and units
