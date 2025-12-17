# EcoHydro: GPU-Accelerated Semiarid Vegetation Dynamics

A Taichi-based simulation of vegetation pattern formation in semiarid landscapes, implementing the ecohydrological feedback mechanisms that drive Turing-type spatial instabilities.

## Overview

Plants in semiarid environments create striking spatial patterns—bands, spots, and labyrinths—through a self-organizing process. Vegetation enhances local infiltration while reducing runoff available to downslope neighbors. This positive local feedback combined with negative nonlocal feedback (competition for water via runoff redistribution) generates emergent patterning.

This project implements the core physics in GPU-accelerated form, targeting:
- **Scale:** 10⁸ cells (10,000 × 10,000 at 10m resolution = 100km × 100km)
- **Duration:** Decades of simulated time in hours of wall time
- **Hardware:** NVIDIA H100 (sm90) and B200 (sm100)

## Scientific Background

The simulation couples three fields:

| Field | Symbol | Description |
|-------|--------|-------------|
| Surface water | h | Ponded water available for infiltration and routing |
| Soil moisture | M | Plant-available water in root zone |
| Vegetation | P | Above-ground biomass density |

Key feedbacks:
- **Infiltration enhancement:** Vegetated areas infiltrate water faster than bare soil
- **Runoff redistribution:** Water flows downslope via Multiple Flow Direction (MFD) routing
- **Water-limited growth:** Vegetation growth depends on soil moisture availability

See `ecohydro_spec.md` for the complete mathematical formulation.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (for sm90/sm100 support)
- NVIDIA GPU with compute capability 9.0+ (H100, B200)

### Setup

```bash
# Clone the repository
git clone https://github.com/aornugent/saco-flow.git
cd saco-flow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
taichi>=1.7.0
numpy>=1.24.0
rasterio>=1.3.0
matplotlib>=3.7.0
```

## Project Structure

```
saco-flow/
├── README.md                 # This file
├── AGENTS.md                 # Guidelines for AI coding agents
├── IMPLEMENTATION_PLAN.md    # Phased development roadmap
├── ecohydro_spec.md          # Mathematical specification (authoritative)
├── docs/
│   ├── ARCHITECTURE.md       # Architecture, buffer strategy, GPU optimization
│   ├── overview.md           # State variables, governing equations
│   ├── boundaries.md         # Domain mask, outlets, depressions
│   ├── timesteps.md          # Operator splitting, CFL conditions
│   ├── data_structures.md    # Memory layout, neighbor indexing
│   ├── mass_conservation.md  # Verification and debugging
│   └── kernels/              # Per-kernel documentation
├── src/
│   ├── config.py             # DTYPE, DefaultParams
│   ├── simulation.py         # Main simulation loop
│   ├── output.py             # Visualization utilities
│   └── kernels/
│       ├── utils.py          # Neighbor offsets, field operations
│       ├── flow.py           # MFD routing, flow accumulation
│       ├── infiltration.py   # Surface → soil water transfer
│       ├── soil.py           # ET, leakage, diffusion
│       └── vegetation.py     # Growth, mortality, dispersal
└── tests/                    # Test suite
```

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/ARCHITECTURE.md` | Module structure, buffer strategy, GPU optimization |
| `ecohydro_spec.md` | Authoritative mathematical specification |
| `AGENTS.md` | Conventions for development |

## Quick Start

```python
import taichi as ti
from src.simulation import EcoHydroSimulation

ti.init(arch=ti.gpu)

# Create simulation with synthetic terrain
sim = EcoHydroSimulation(
    nx=1000,
    ny=1000,
    dx=10.0,  # 10m resolution
)

# Initialize tilted plane terrain
sim.init_tilted_plane(slope=0.02)

# Run for 10 simulated years
sim.run(years=10, output_interval_days=30)
```

## Development Status

See `IMPLEMENTATION_PLAN.md` for the roadmap.

**Current status:** Core simulation working. Mass conservation verified. Next: GPU optimization.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_mass_conservation.py -v
```

## References

1. Rietkerk et al. (2002) - Self-organized patchiness and catastrophic shifts in ecosystems
2. Saco et al. (2007) - Eco-geomorphology of banded vegetation patterns
3. Saco et al. (2013) - Ecogeomorphic coevolution of semiarid hillslopes (see PDF in repo)

## License

[TBD]
