"""Taichi parameter injection.

Bridges Python dataclasses to Taichi scalar fields for kernel access.
"""

import taichi as ti

from src.core.dtypes import DTYPE
from src.params.schema import SimulationConfig


class TaichiParams:
    """Taichi-accessible parameter container."""

    def __init__(self) -> None:
        """Create Taichi parameter fields."""
        # Grid
        self.dx = ti.field(DTYPE, shape=())

        # Infiltration
        self.alpha = ti.field(DTYPE, shape=())
        self.k_P = ti.field(DTYPE, shape=())
        self.W_0 = ti.field(DTYPE, shape=())

        # Soil
        self.M_sat = ti.field(DTYPE, shape=())
        self.E_max = ti.field(DTYPE, shape=())
        self.k_ET = ti.field(DTYPE, shape=())
        self.beta_ET = ti.field(DTYPE, shape=())
        self.L_max = ti.field(DTYPE, shape=())
        self.D_M = ti.field(DTYPE, shape=())

        # Vegetation
        self.g_max = ti.field(DTYPE, shape=())
        self.k_G = ti.field(DTYPE, shape=())
        self.mu = ti.field(DTYPE, shape=())
        self.D_P = ti.field(DTYPE, shape=())

        # Routing
        self.manning_n = ti.field(DTYPE, shape=())

        # Drainage
        self.h_threshold = ti.field(DTYPE, shape=())
        self.drainage_time = ti.field(DTYPE, shape=())

        # Rainfall
        self.rain_depth = ti.field(DTYPE, shape=())
        self.storm_duration = ti.field(DTYPE, shape=())
        self.interstorm = ti.field(DTYPE, shape=())

        # Timestep
        self.dt_veg = ti.field(DTYPE, shape=())
        self.dt_soil = ti.field(DTYPE, shape=())

        self._loaded = False

    def load(self, config: SimulationConfig) -> None:
        """Load parameters from SimulationConfig."""
        self.dx[None] = config.grid.dx

        self.alpha[None] = config.infiltration.alpha
        self.k_P[None] = config.infiltration.k_P
        self.W_0[None] = config.infiltration.W_0

        self.M_sat[None] = config.soil.M_sat
        self.E_max[None] = config.soil.E_max
        self.k_ET[None] = config.soil.k_ET
        self.beta_ET[None] = config.soil.beta_ET
        self.L_max[None] = config.soil.L_max
        self.D_M[None] = config.soil.D_M

        self.g_max[None] = config.vegetation.g_max
        self.k_G[None] = config.vegetation.k_G
        self.mu[None] = config.vegetation.mu
        self.D_P[None] = config.vegetation.D_P

        self.manning_n[None] = config.routing.manning_n

        self.h_threshold[None] = config.drainage.h_threshold
        self.drainage_time[None] = config.drainage.drainage_time

        self.rain_depth[None] = config.rainfall.rain_depth
        self.storm_duration[None] = config.rainfall.storm_duration
        self.interstorm[None] = config.rainfall.interstorm

        self.dt_veg[None] = config.timestep.dt_veg
        self.dt_soil[None] = config.timestep.dt_soil

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def to_dict(self) -> dict[str, float]:
        """Extract current values as dictionary."""
        return {
            "dx": float(self.dx[None]),
            "alpha": float(self.alpha[None]),
            "k_P": float(self.k_P[None]),
            "W_0": float(self.W_0[None]),
            "M_sat": float(self.M_sat[None]),
            "E_max": float(self.E_max[None]),
            "k_ET": float(self.k_ET[None]),
            "beta_ET": float(self.beta_ET[None]),
            "L_max": float(self.L_max[None]),
            "D_M": float(self.D_M[None]),
            "g_max": float(self.g_max[None]),
            "k_G": float(self.k_G[None]),
            "mu": float(self.mu[None]),
            "D_P": float(self.D_P[None]),
            "manning_n": float(self.manning_n[None]),
            "h_threshold": float(self.h_threshold[None]),
            "drainage_time": float(self.drainage_time[None]),
            "rain_depth": float(self.rain_depth[None]),
            "storm_duration": float(self.storm_duration[None]),
            "interstorm": float(self.interstorm[None]),
            "dt_veg": float(self.dt_veg[None]),
            "dt_soil": float(self.dt_soil[None]),
        }


def create_taichi_params(config: SimulationConfig) -> TaichiParams:
    """Create and load TaichiParams from config."""
    params = TaichiParams()
    params.load(config)
    return params
