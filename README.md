# SACO-Flow

GPU-accelerated ecohydrological simulation using Taichi, targeting NVIDIA H100/B200.

## Setup

```bash
uv sync && source .venv/bin/activate
```

## Project Structure

```
saco-flow/
├── src/
│   ├── config.py              # Taichi backend init (CUDA/Vulkan/CPU)
│   ├── fields.py              # Field allocation and buffer swap
│   └── kernels/
│       └── diffusion.py       # Laplacian diffusion stencil
├── tests/
│   └── test_diffusion.py      # Mass conservation and correctness
├── benchmarks/
│   ├── harness.py             # Benchmark base class
│   ├── diffusion.py           # Stencil throughput benchmark
│   └── run.py                 # CLI runner
├── pyproject.toml
├── ruff.toml
└── AGENTS.md
```

## Usage

```bash
# Run tests
pytest

# Run benchmarks
python -m benchmarks.run

# Lint
ruff check --fix . && ruff format .
```

## References

1. Saco et al. (2007) — Eco-geomorphology of banded vegetation patterns
2. Saco et al. (2013) — Ecogeomorphic coevolution of semiarid hillslopes
