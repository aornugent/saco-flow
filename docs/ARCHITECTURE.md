# EcoHydro Architecture

This document describes the target architecture for the refactored EcoHydro simulation. It provides guidance for Phase 6 optimization work.

## Design Goals

1. **Separation of concerns**: Fields, parameters, kernels, and orchestration are independent modules
2. **Testability**: Every component is unit-testable in isolation
3. **Swappable implementations**: Naive and optimized kernels share an interface
4. **Type safety**: Catch configuration errors at initialization, not at kernel launch
5. **Taichi-idiomatic**: Leverage `ti.template()`, `ti.static()`, data-oriented patterns

## Module Structure

```
src/
├── core/
│   ├── dtypes.py           # Type definitions (DTYPE)
│   ├── geometry.py         # Grid geometry, neighbor indexing
│   └── constants.py        # Physical constants
│
├── fields/
│   ├── base.py             # FieldContainer base class
│   ├── state.py            # StateFields (h, M, P)
│   ├── static.py           # StaticFields (Z, mask, flow_frac)
│   └── scratch.py          # ScratchFields (temporary buffers)
│
├── params/
│   ├── schema.py           # Parameter dataclasses with validation
│   ├── loader.py           # YAML loading
│   └── taichi_params.py    # Parameter injection into Taichi fields
│
├── kernels/
│   ├── __init__.py         # Kernel registry and dispatch
│   ├── protocol.py         # Abstract kernel interfaces
│   ├── naive/              # Reference implementations
│   │   ├── soil.py
│   │   ├── vegetation.py
│   │   └── flow.py
│   └── optimized/          # Fused/blocked implementations
│       ├── soil_fused.py
│       ├── vegetation_fused.py
│       └── flow_fused.py
│
├── simulation/
│   ├── timestep.py         # Timestep computation, CFL
│   ├── operators.py        # Operator-split stages
│   └── runner.py           # Main simulation loop
│
├── diagnostics/
│   ├── conservation.py     # Mass balance checks
│   └── profiling.py        # Kernel timing hooks
│
└── io/
    ├── dem.py              # DEM loading
    └── output.py           # State serialization
```

## Key Patterns

### 1. Grid Geometry

Encapsulate all spatial indexing logic in a single, immutable dataclass.

**Current**: Neighbor offsets duplicated in each kernel.

**Target**: Centralized in `core/geometry.py`:
- `GridGeometry` dataclass with `nx`, `ny`, `dx`, computed properties
- Compile-time neighbor vectors (`NEIGHBOR_DI`, `NEIGHBOR_DJ`, `NEIGHBOR_DIST`)
- Shared `@ti.func` helpers for neighbor iteration

**Rationale**: Neighbor indexing is used in every kernel. Centralizing it:
- Eliminates copy-paste errors
- Allows static unrolling via `ti.static(range(8))`
- Documents the convention in one place

### 2. Field Container

Typed containers that manage Taichi field lifecycle and double-buffering.

**Current**: `SimpleNamespace` in `simulation.py` with manual field creation.

**Target**: `FieldContainer` class with:
- Declarative `FieldSpec` registration (name, dtype, shape, role)
- Explicit `double_buffer` flag in spec
- `swap_buffers(name)` method for stencil operations
- Memory tracking (`memory_mb` property)

**Roles**:
| Role | Description | Double-buffered |
|------|-------------|-----------------|
| STATE | Primary state (h, M, P) | Yes |
| STATIC | Read-only during sim (Z, mask) | No |
| DERIVED | Computed from state (flow_acc) | No |
| SCRATCH | Temporary workspace | No |

**Rationale**:
- Declarative specs separate "what fields exist" from "how they're allocated"
- Double-buffering is explicit, not hidden in kernel logic
- Memory tracking catches budget violations early

### 3. Parameter Schema

Validated, immutable parameter containers with units documentation.

**Current**: `DefaultParams` class with constants, `SimulationParams` dataclass.

**Target**: Nested frozen dataclasses with `__post_init__` validation:
- `InfiltrationParams`, `EvapotranspirationParams`, `VegetationParams`, etc.
- `SimulationParams` aggregates all with `from_yaml()` classmethod
- Units documented in docstrings and field comments

**Rationale**: Catch invalid parameters at load time, not mid-simulation.

### 4. Taichi Parameter Injection

Bridge between Python dataclasses and Taichi kernel-accessible fields.

**Current**: Parameters passed as scalar arguments to kernels.

**Target**: `TaichiParams` class holding scalar fields:
- `alpha = ti.field(ti.f32, shape=())`
- `load(params: SimulationParams)` copies values from dataclass
- Kernels read from fields: `params.alpha[None]`

**Trade-off**: One extra global memory read per parameter. For memory-bound kernels, this is negligible. Benefit: parameters can change at runtime without recompilation (useful for sensitivity analysis).

### 5. Kernel Protocol

Abstract interface that both naive and optimized kernels implement.

**Current**: Functions in `src/kernels/*.py`, called directly.

**Target**: Protocol classes defining the interface:

```python
@runtime_checkable
class SoilKernel(Protocol):
    def step(self, state, static, params, dt) -> None: ...
    @property
    def fields_read(self) -> set[str]: ...
    @property
    def fields_written(self) -> set[str]: ...
```

Each kernel type (Soil, Vegetation, Flow) has a protocol. Implementations provide:
- `step()`: Execute one timestep
- `fields_read`/`fields_written`: For dependency tracking and documentation

**Rationale**: Swap implementations without changing orchestration code.

### 6. Kernel Registry

Factory pattern for selecting kernel implementations at runtime.

**Current**: Direct imports in `simulation.py`.

**Target**: `KernelRegistry` class with:
- `KernelVariant` enum: `NAIVE`, `FUSED`, `TEMPORAL`
- `get_soil(variant)`, `get_vegetation(variant)`, `get_flow(variant)`
- `register_soil(variant, impl)` for adding new implementations

**Usage**:
```python
registry = KernelRegistry()
soil_kernel = registry.get_soil(KernelVariant.FUSED)
```

**Rationale**:
- Single point of configuration for kernel selection
- Easy A/B testing between implementations
- Test harness can verify all variants produce equivalent results

### 7. Simulation Runner

Orchestration layer implementing operator splitting.

**Current**: `Simulation` class in `simulation.py` with mixed concerns.

**Target**: `SimulationRunner` with clear responsibilities:
- Operator-split time stepping
- Kernel dispatch (via registry)
- Event scheduling (rainfall)
- Callback hooks for extensibility

**Not responsible for**:
- Field allocation (done externally via FieldContainer factories)
- I/O (handled by separate module)
- Parameter loading (handled by params module)

### 8. Diagnostics

Built-in mass conservation verification.

**Current**: `MassBalance` dataclass with manual tracking.

**Target**: `MassTracker` class with:
- Kernel-based reduction for totals
- `initialize()` to record initial state
- `check()` to verify conservation

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Dataclass params + Taichi field injection | Type safety at config time, runtime flexibility |
| FieldContainer with specs | Declarative fields, automatic double-buffer management |
| Protocol-based kernel interface | Swap implementations without changing orchestration |
| Kernel registry with variants | A/B testing, gradual optimization migration |
| Centralized geometry/neighbor indexing | Single source of truth, `ti.static()` unrolling |
| Callback-based extension points | Output, profiling, checkpointing without coupling |

## Migration Path

The refactoring should proceed incrementally:

### Phase 6.0a: Core Infrastructure
1. Create `core/geometry.py` with `GridGeometry` and neighbor helpers
2. Create `core/dtypes.py` (move `DTYPE` from config)
3. Add tests for geometry utilities

### Phase 6.0b: Field Management
1. Create `fields/base.py` with `FieldContainer` and `FieldSpec`
2. Create `fields/state.py` with state field factory
3. Create `fields/static.py` with static field factory
4. Migrate `create_simulation_fields()` to use containers
5. Add double-buffer swap tests

### Phase 6.0c: Parameter System
1. Create `params/schema.py` with nested dataclasses
2. Create `params/taichi_params.py` with injection class
3. Migrate `SimulationParams` to new schema
4. Add validation tests

### Phase 6.0d: Kernel Protocols
1. Create `kernels/protocol.py` with Protocol definitions
2. Move existing kernels to `kernels/naive/`
3. Wrap existing kernels in protocol-compliant classes
4. Create `kernels/__init__.py` with registry

### Phase 6.0e: Runner Refactoring
1. Create `simulation/runner.py` using new components
2. Verify existing tests pass with new runner
3. Add callback hooks

Only after this groundwork is complete should kernel fusion begin (6.2+).

## Kernel Equivalence Testing

Every optimized kernel must pass equivalence tests:

1. Initialize identical state for naive and optimized
2. Run both with same parameters and timestep
3. Compare results within tolerance (1e-5)
4. Verify mass conservation for both

This ensures optimizations don't introduce numerical drift.

## References

- `docs/gpu_optimization.md` - Performance targets and memory analysis
- `docs/data_structures.md` - Memory layout conventions
- `IMPLEMENTATION_PLAN.md` - Phase 6 task breakdown
