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

See `docs/physics.md` for the complete mathematical formulation.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.0+ (for sm90/sm100 support)
- NVIDIA GPU with compute capability 9.0+ (H100, B200)

### Setup

We use `uv` for speedy dependency management.

#### Development Container (Recommended)

This project includes a `.devcontainer` configuration. This provides a consistent environment with Python 3.12, CUDA 13.0 support, and all dependencies pre-installed.

The environment should be automatically detected. If not, reopen the workspace in the container.

#### Local Setup

If you prefer to run locally:

```bash
# Clone the repository
git clone https://github.com/aornugent/saco-flow.git
cd saco-flow

# Install dependencies into .venv
uv sync

# Activate venv
source .venv/bin/activate
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
├── docs/
│   ├── overview.md           # System description, feedback mechanism
│   ├── physics.md            # Equations, parameters (authoritative)
│   ├── discretization.md     # Numerical methods, boundaries, timesteps
│   ├── architecture.md       # Code structure, buffers, GPU optimization
│   ├── testing.md            # Conservation verification, debugging
│   ├── BENCHMARKS.md         # Performance measurements
│   └── kernels/
│       ├── flow.md           # Flow direction, routing kernels
│       ├── soil.md           # ET, leakage, diffusion kernels
│       └── vegetation.md     # Growth, mortality, dispersal kernels
├── src/
│   ├── config.py             # Taichi initialization
│   ├── params.py             # SimulationParams (single source of truth)
│   ├── simulation.py         # Main simulation loop
│   ├── initialization.py     # Data initialization (DEM, vegetation)
│   ├── fields.py             # Data structure allocation
│   ├── output.py             # Visualization utilities
│   └── kernels/
│       ├── flow.py           # MFD routing, flow accumulation
│       ├── infiltration.py   # Surface -> soil water transfer
│       ├── soil.py           # ET, leakage, diffusion
│       └── vegetation.py     # Growth, mortality, dispersal
└── tests/                    # Test suite
```

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/overview.md` | System description, state variables |
| `docs/physics.md` | Mathematical specification, parameters |
| `docs/architecture.md` | Code structure, buffer strategy, GPU |
| `AGENTS.md` | Development conventions |

## Quick Start

### 1. Visualization (CLI)
Run the real-time 3D visualization:
```bash
# Default grid (64x64)
python -m src.main --gui

# Custom size and duration
python -m src.main --gui --n 1024 --years 5.0
```

### 2. Headless Simulation (HPC)
Run large-scale simulations with configuration files:
```bash
python -m src.main --config experiments/10k_100yr.toml --output results/run1
```

### 3. Python API
```python
import taichi as ti
from src.simulation import Simulation
from src.params import SimulationParams

ti.init(arch=ti.gpu)

# Create simulation with synthetic terrain
params = SimulationParams(
    n=1000,
    dx=10.0,  # 10m resolution
)
sim = Simulation(params)

# Initialize tilted plane terrain
sim.initialize(slope=0.02)

# Run for 10 simulated years
sim.run(years=10)
```

## Development Status

**Current status:** Core simulation complete with GPU optimization. Mass conservation verified. See `docs/BENCHMARKS.md` for performance data.

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
