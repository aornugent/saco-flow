"""
Parameter management module for EcoHydro simulation.

This module provides:
- Validated, immutable parameter containers (schema.py)
- YAML loading utilities (loader.py)
- Taichi parameter injection (taichi_params.py)
"""

from src.params.schema import (
    GridParams,
    RainfallParams,
    InfiltrationParams,
    SoilParams,
    VegetationParams,
    RoutingParams,
    DrainageParams,
    TimestepParams,
    SimulationConfig,
)
from src.params.loader import load_config, save_config
from src.params.taichi_params import TaichiParams

__all__ = [
    # Schema classes
    "GridParams",
    "RainfallParams",
    "InfiltrationParams",
    "SoilParams",
    "VegetationParams",
    "RoutingParams",
    "DrainageParams",
    "TimestepParams",
    "SimulationConfig",
    # Loader functions
    "load_config",
    "save_config",
    # Taichi injection
    "TaichiParams",
]
