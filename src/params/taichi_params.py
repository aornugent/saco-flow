"""
Taichi parameter injection.

Bridges Python dataclasses to Taichi-accessible scalar fields.
This allows parameters to be read from kernels without recompilation.

Usage:
    config = SimulationConfig(grid=GridParams(n=128))
    ti_params = TaichiParams()
    ti_params.load(config)

    @ti.kernel
    def my_kernel():
        alpha = ti_params.alpha[None]  # Read from Taichi field
"""

import taichi as ti

from src.core.dtypes import DTYPE
from src.params.schema import SimulationConfig


class TaichiParams:
    """
    Taichi-accessible parameter container.

    Holds all simulation parameters as ti.field(shape=()) scalar fields
    that can be read from Taichi kernels. This approach has a small
    performance cost (one global memory read per parameter per kernel)
    but enables runtime parameter changes without recompilation.

    Attributes:
        All parameter fields from SimulationConfig are exposed as
        ti.field(dtype, shape=()) with matching names.
    """

    def __init__(self) -> None:
        """Create uninitialized Taichi parameter fields."""
        # Grid parameters
        self.dx = ti.field(DTYPE, shape=())

        # Infiltration parameters
        self.alpha = ti.field(DTYPE, shape=())
        self.alpha_i = ti.field(DTYPE, shape=())
        self.k_P = ti.field(DTYPE, shape=())
        self.W_0 = ti.field(DTYPE, shape=())
        self.k_sat = ti.field(DTYPE, shape=())

        # Soil parameters
        self.M_sat = ti.field(DTYPE, shape=())
        self.E_max = ti.field(DTYPE, shape=())
        self.k_ET = ti.field(DTYPE, shape=())
        self.beta_ET = ti.field(DTYPE, shape=())
        self.L_max = ti.field(DTYPE, shape=())
        self.D_M = ti.field(DTYPE, shape=())

        # Vegetation parameters
        self.g_max = ti.field(DTYPE, shape=())
        self.k_G = ti.field(DTYPE, shape=())
        self.mu = ti.field(DTYPE, shape=())
        self.D_P = ti.field(DTYPE, shape=())

        # Routing parameters
        self.manning_n = ti.field(DTYPE, shape=())
        self.min_slope = ti.field(DTYPE, shape=())

        # Drainage parameters
        self.h_threshold = ti.field(DTYPE, shape=())
        self.drainage_time = ti.field(DTYPE, shape=())

        # Rainfall parameters
        self.rain_depth = ti.field(DTYPE, shape=())
        self.storm_duration = ti.field(DTYPE, shape=())
        self.interstorm = ti.field(DTYPE, shape=())

        # Timestep parameters
        self.dt_veg = ti.field(DTYPE, shape=())
        self.dt_soil = ti.field(DTYPE, shape=())

        # Track if parameters have been loaded
        self._loaded = False

    def load(self, config: SimulationConfig) -> None:
        """
        Load parameters from SimulationConfig into Taichi fields.

        Args:
            config: Validated SimulationConfig instance
        """
        # Grid
        self.dx[None] = config.grid.dx

        # Infiltration
        self.alpha[None] = config.infiltration.alpha
        self.alpha_i[None] = config.infiltration.alpha_i
        self.k_P[None] = config.infiltration.k_P
        self.W_0[None] = config.infiltration.W_0
        self.k_sat[None] = config.infiltration.k_sat

        # Soil
        self.M_sat[None] = config.soil.M_sat
        self.E_max[None] = config.soil.E_max
        self.k_ET[None] = config.soil.k_ET
        self.beta_ET[None] = config.soil.beta_ET
        self.L_max[None] = config.soil.L_max
        self.D_M[None] = config.soil.D_M

        # Vegetation
        self.g_max[None] = config.vegetation.g_max
        self.k_G[None] = config.vegetation.k_G
        self.mu[None] = config.vegetation.mu
        self.D_P[None] = config.vegetation.D_P

        # Routing
        self.manning_n[None] = config.routing.manning_n
        self.min_slope[None] = config.routing.min_slope

        # Drainage
        self.h_threshold[None] = config.drainage.h_threshold
        self.drainage_time[None] = config.drainage.drainage_time

        # Rainfall
        self.rain_depth[None] = config.rainfall.rain_depth
        self.storm_duration[None] = config.rainfall.storm_duration
        self.interstorm[None] = config.rainfall.interstorm

        # Timesteps
        self.dt_veg[None] = config.timestep.dt_veg
        self.dt_soil[None] = config.timestep.dt_soil

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if parameters have been loaded."""
        return self._loaded

    def to_dict(self) -> dict[str, float]:
        """
        Extract current parameter values as a dictionary.

        Useful for debugging and logging.

        Returns:
            Dictionary of parameter names to values
        """
        return {
            # Grid
            "dx": float(self.dx[None]),
            # Infiltration
            "alpha": float(self.alpha[None]),
            "alpha_i": float(self.alpha_i[None]),
            "k_P": float(self.k_P[None]),
            "W_0": float(self.W_0[None]),
            "k_sat": float(self.k_sat[None]),
            # Soil
            "M_sat": float(self.M_sat[None]),
            "E_max": float(self.E_max[None]),
            "k_ET": float(self.k_ET[None]),
            "beta_ET": float(self.beta_ET[None]),
            "L_max": float(self.L_max[None]),
            "D_M": float(self.D_M[None]),
            # Vegetation
            "g_max": float(self.g_max[None]),
            "k_G": float(self.k_G[None]),
            "mu": float(self.mu[None]),
            "D_P": float(self.D_P[None]),
            # Routing
            "manning_n": float(self.manning_n[None]),
            "min_slope": float(self.min_slope[None]),
            # Drainage
            "h_threshold": float(self.h_threshold[None]),
            "drainage_time": float(self.drainage_time[None]),
            # Rainfall
            "rain_depth": float(self.rain_depth[None]),
            "storm_duration": float(self.storm_duration[None]),
            "interstorm": float(self.interstorm[None]),
            # Timesteps
            "dt_veg": float(self.dt_veg[None]),
            "dt_soil": float(self.dt_soil[None]),
        }


def create_taichi_params(config: SimulationConfig) -> TaichiParams:
    """
    Convenience factory to create and load TaichiParams.

    Args:
        config: Validated SimulationConfig instance

    Returns:
        Loaded TaichiParams instance
    """
    params = TaichiParams()
    params.load(config)
    return params
