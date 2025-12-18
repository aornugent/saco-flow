"""Simulation parameters with validation.

Simple dataclass with __post_init__ validation. All units documented.
Default values from Saco et al. (2013).

Units: meters, days throughout.
"""

from dataclasses import dataclass


def _check_positive(obj, *names):
    """Validate parameters are strictly positive."""
    for name in names:
        val = getattr(obj, name)
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")


def _check_non_negative(obj, *names):
    """Validate parameters are non-negative."""
    for name in names:
        val = getattr(obj, name)
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")


def _check_fraction(obj, *names):
    """Validate parameters are in [0, 1]."""
    for name in names:
        val = getattr(obj, name)
        if not 0 <= val <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass
class SimulationParams:
    """All simulation parameters with defaults.

    Attributes documented with units in comments.
    """

    # Grid
    n: int = 64  # grid size (n x n cells)
    dx: float = 1.0  # [m] cell size

    # Rainfall (intensity-based)
    rain_depth: float = 0.02  # [m] mean event depth (~20mm)
    storm_duration: float = 0.25  # [days] event duration (~6 hours)
    interstorm: float = 18.0  # [days] mean interstorm period (~20 events/year)

    # Infiltration
    alpha: float = 0.1  # [1/day] infiltration rate coefficient
    k_P: float = 1.0  # [kg/m^2] vegetation half-saturation for infiltration
    W_0: float = 0.2  # [-] bare soil infiltration factor
    M_sat: float = 0.4  # [m] saturated soil moisture

    # Soil moisture
    E_max: float = 0.005  # [m/day] max ET rate (~5mm/day)
    k_ET: float = 0.1  # [m] ET half-saturation moisture
    beta_ET: float = 0.5  # [-] vegetation enhancement of ET
    L_max: float = 0.002  # [1/day] deep leakage coefficient
    D_M: float = 0.1  # [m^2/day] soil moisture diffusivity

    # Vegetation
    g_max: float = 0.02  # [1/day] max growth rate
    k_G: float = 0.1  # [m] growth half-saturation moisture
    mu: float = 0.001  # [1/day] mortality rate
    D_P: float = 0.01  # [m^2/day] seed dispersal diffusivity

    # Surface routing
    manning_n: float = 0.03  # [-] Manning's roughness coefficient
    min_slope: float = 1e-6  # [-] minimum slope for flow

    # Drainage (for rainfall event completion)
    h_threshold: float = 1e-6  # [m] water depth threshold
    drainage_time: float = 1.0  # [days] extra drainage time after event

    # Timesteps
    dt_veg: float = 7.0  # [days] vegetation update timestep
    dt_soil: float = 1.0  # [days] soil moisture update timestep

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.n < 3:
            raise ValueError(f"Grid size n must be >= 3, got {self.n}")

        _check_positive(
            self,
            "dx",
            "rain_depth",
            "storm_duration",
            "interstorm",
            "alpha",
            "k_P",
            "M_sat",
            "E_max",
            "k_ET",
            "g_max",
            "k_G",
            "mu",
            "manning_n",
            "min_slope",
            "h_threshold",
            "dt_veg",
            "dt_soil",
        )

        _check_non_negative(
            self,
            "beta_ET",
            "L_max",
            "D_M",
            "D_P",
            "drainage_time",
        )

        _check_fraction(self, "W_0")

    @property
    def cell_area(self) -> float:
        """Cell area [m^2]."""
        return self.dx * self.dx

    @property
    def events_per_year(self) -> float:
        """Expected rainfall events per year."""
        return 365.0 / self.interstorm

    @property
    def annual_rainfall(self) -> float:
        """Expected annual rainfall [m/year]."""
        return self.rain_depth * self.events_per_year
