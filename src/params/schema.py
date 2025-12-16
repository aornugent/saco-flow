"""
Parameter schema definitions with validation.

All parameter containers are frozen dataclasses with __post_init__ validation.
Units are documented in docstrings and field comments.

Parameter categories mirror the physics domains:
- Grid: spatial discretization
- Rainfall: precipitation events
- Infiltration: surface-to-soil water transfer
- Soil: moisture dynamics (ET, leakage, diffusion)
- Vegetation: growth, mortality, dispersal
- Routing: surface water flow
- Drainage: event completion criteria
- Timestep: temporal discretization
"""

from dataclasses import dataclass, field, asdict
from typing import Any


class ValidationError(ValueError):
    """Raised when parameter validation fails."""

    pass


def _validate_positive(value: float, name: str) -> None:
    """Validate that a parameter is strictly positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a parameter is non-negative."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def _validate_fraction(value: float, name: str) -> None:
    """Validate that a parameter is in [0, 1]."""
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be in [0, 1], got {value}")


def _validate_range(value: float, name: str, low: float, high: float) -> None:
    """Validate that a parameter is in [low, high]."""
    if not low <= value <= high:
        raise ValidationError(f"{name} must be in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class GridParams:
    """
    Grid spatial parameters.

    Attributes:
        n: Grid size (n x n cells)
        dx: Cell size [m]
    """

    n: int = 64
    dx: float = 1.0  # [m]

    def __post_init__(self) -> None:
        if self.n < 3:
            raise ValidationError(f"Grid size n must be >= 3, got {self.n}")
        _validate_positive(self.dx, "dx")

    @property
    def total_area(self) -> float:
        """Total domain area [m²]."""
        return (self.n * self.dx) ** 2

    @property
    def cell_area(self) -> float:
        """Single cell area [m²]."""
        return self.dx * self.dx


@dataclass(frozen=True)
class RainfallParams:
    """
    Rainfall event parameters.

    Controls the stochastic rainfall generator using Poisson process.

    Attributes:
        rain_depth: Mean rainfall depth per event [m]
        storm_duration: Duration of storm event [days]
        interstorm: Mean time between storms [days]
    """

    rain_depth: float = 0.02  # [m] ~20mm per event
    storm_duration: float = 0.25  # [days] ~6 hours
    interstorm: float = 18.0  # [days] ~20 events/year

    def __post_init__(self) -> None:
        _validate_positive(self.rain_depth, "rain_depth")
        _validate_positive(self.storm_duration, "storm_duration")
        _validate_positive(self.interstorm, "interstorm")

    @property
    def events_per_year(self) -> float:
        """Expected rainfall events per year."""
        return 365.0 / self.interstorm

    @property
    def annual_rainfall(self) -> float:
        """Expected annual rainfall [m/year]."""
        return self.rain_depth * self.events_per_year


@dataclass(frozen=True)
class InfiltrationParams:
    """
    Infiltration parameters.

    Controls water transfer from surface to soil.
    Formula: I = alpha * h * (W_0 + (1 - W_0) * P / (P + k_P)) * (1 - M/M_sat)

    Attributes:
        alpha: Infiltration rate coefficient [1/day]
        alpha_i: Infiltration feedback strength [-] (unused in current kernels)
        k_P: Vegetation half-saturation for infiltration [kg/m²]
        W_0: Bare soil infiltration factor [-]
        k_sat: Saturated hydraulic conductivity [m/day] (reference)
    """

    alpha: float = 0.1  # [1/day]
    alpha_i: float = 2.0  # [-]
    k_P: float = 1.0  # [kg/m²]
    W_0: float = 0.2  # [-]
    k_sat: float = 0.24  # [m/day]

    def __post_init__(self) -> None:
        _validate_positive(self.alpha, "alpha")
        _validate_non_negative(self.alpha_i, "alpha_i")
        _validate_positive(self.k_P, "k_P")
        _validate_fraction(self.W_0, "W_0")
        _validate_positive(self.k_sat, "k_sat")


@dataclass(frozen=True)
class SoilParams:
    """
    Soil moisture dynamics parameters.

    Controls ET, leakage, and moisture diffusion.

    Attributes:
        M_sat: Saturated soil moisture content [m]
        E_max: Maximum evapotranspiration rate [m/day]
        k_ET: ET half-saturation moisture [m]
        beta_ET: Vegetation enhancement of ET [-]
        L_max: Deep leakage coefficient [1/day]
        D_M: Soil moisture diffusivity [m²/day]
    """

    M_sat: float = 0.4  # [m]
    E_max: float = 0.005  # [m/day] ~5mm/day
    k_ET: float = 0.1  # [m]
    beta_ET: float = 0.5  # [-]
    L_max: float = 0.002  # [1/day]
    D_M: float = 0.1  # [m²/day]

    def __post_init__(self) -> None:
        _validate_positive(self.M_sat, "M_sat")
        _validate_positive(self.E_max, "E_max")
        _validate_positive(self.k_ET, "k_ET")
        _validate_non_negative(self.beta_ET, "beta_ET")
        _validate_non_negative(self.L_max, "L_max")
        _validate_non_negative(self.D_M, "D_M")


@dataclass(frozen=True)
class VegetationParams:
    """
    Vegetation dynamics parameters.

    Controls growth, mortality, and seed dispersal.

    Attributes:
        g_max: Maximum growth rate [1/day]
        k_G: Growth half-saturation moisture [m]
        mu: Mortality rate [1/day]
        D_P: Seed dispersal diffusivity [m²/day]
    """

    g_max: float = 0.02  # [1/day]
    k_G: float = 0.1  # [m]
    mu: float = 0.001  # [1/day]
    D_P: float = 0.01  # [m²/day]

    def __post_init__(self) -> None:
        _validate_positive(self.g_max, "g_max")
        _validate_positive(self.k_G, "k_G")
        _validate_positive(self.mu, "mu")
        _validate_non_negative(self.D_P, "D_P")

    @property
    def turnover_time(self) -> float:
        """Expected vegetation turnover time [days]."""
        return 1.0 / self.mu


@dataclass(frozen=True)
class RoutingParams:
    """
    Surface water routing parameters.

    Controls Manning's equation flow routing.

    Attributes:
        manning_n: Manning's roughness coefficient [-]
        min_slope: Minimum slope for flow calculation [-]
    """

    manning_n: float = 0.03  # [-]
    min_slope: float = 1e-6  # [-]

    def __post_init__(self) -> None:
        _validate_positive(self.manning_n, "manning_n")
        _validate_positive(self.min_slope, "min_slope")


@dataclass(frozen=True)
class DrainageParams:
    """
    Drainage and event completion parameters.

    Controls when rainfall event processing terminates.

    Attributes:
        h_threshold: Water depth threshold for drainage completion [m]
        drainage_time: Extra drainage time after rainfall ends [days]
    """

    h_threshold: float = 1e-6  # [m]
    drainage_time: float = 1.0  # [days]

    def __post_init__(self) -> None:
        _validate_positive(self.h_threshold, "h_threshold")
        _validate_non_negative(self.drainage_time, "drainage_time")


@dataclass(frozen=True)
class TimestepParams:
    """
    Temporal discretization parameters.

    Attributes:
        dt_veg: Vegetation update timestep [days]
        dt_soil: Soil moisture update timestep [days]
    """

    dt_veg: float = 7.0  # [days]
    dt_soil: float = 1.0  # [days]

    def __post_init__(self) -> None:
        _validate_positive(self.dt_veg, "dt_veg")
        _validate_positive(self.dt_soil, "dt_soil")


@dataclass(frozen=True)
class SimulationConfig:
    """
    Complete simulation configuration aggregating all parameter groups.

    This is the top-level configuration object used to configure simulations.
    All sub-parameter groups are frozen dataclasses with validation.

    Example:
        config = SimulationConfig(
            grid=GridParams(n=128, dx=2.0),
            soil=SoilParams(M_sat=0.5),
        )
    """

    grid: GridParams = field(default_factory=GridParams)
    rainfall: RainfallParams = field(default_factory=RainfallParams)
    infiltration: InfiltrationParams = field(default_factory=InfiltrationParams)
    soil: SoilParams = field(default_factory=SoilParams)
    vegetation: VegetationParams = field(default_factory=VegetationParams)
    routing: RoutingParams = field(default_factory=RoutingParams)
    drainage: DrainageParams = field(default_factory=DrainageParams)
    timestep: TimestepParams = field(default_factory=TimestepParams)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return {
            "grid": asdict(self.grid),
            "rainfall": asdict(self.rainfall),
            "infiltration": asdict(self.infiltration),
            "soil": asdict(self.soil),
            "vegetation": asdict(self.vegetation),
            "routing": asdict(self.routing),
            "drainage": asdict(self.drainage),
            "timestep": asdict(self.timestep),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """
        Create configuration from nested dictionary.

        Args:
            data: Dictionary with parameter groups as keys

        Returns:
            SimulationConfig instance

        Raises:
            ValidationError: If any parameter validation fails
        """
        kwargs = {}
        param_classes = {
            "grid": GridParams,
            "rainfall": RainfallParams,
            "infiltration": InfiltrationParams,
            "soil": SoilParams,
            "vegetation": VegetationParams,
            "routing": RoutingParams,
            "drainage": DrainageParams,
            "timestep": TimestepParams,
        }

        for key, param_cls in param_classes.items():
            if key in data:
                kwargs[key] = param_cls(**data[key])

        return cls(**kwargs)

    def with_updates(self, **kwargs: Any) -> "SimulationConfig":
        """
        Create new config with updated parameter groups.

        Args:
            **kwargs: Parameter group names mapping to updated dataclasses
                     or dictionaries of parameter values

        Returns:
            New SimulationConfig with updates applied

        Example:
            new_config = config.with_updates(
                grid=GridParams(n=256),
                soil={"M_sat": 0.6}
            )
        """
        current = self.to_dict()

        for key, value in kwargs.items():
            if key not in current:
                raise ValidationError(f"Unknown parameter group: {key}")
            if isinstance(value, dict):
                current[key].update(value)
            else:
                # Assume it's a dataclass instance
                current[key] = asdict(value)

        return self.from_dict(current)

    # Convenience accessors for backward compatibility with SimulationParams
    @property
    def n(self) -> int:
        """Grid size (convenience accessor)."""
        return self.grid.n

    @property
    def dx(self) -> float:
        """Cell size (convenience accessor)."""
        return self.grid.dx

    @property
    def alpha(self) -> float:
        """Infiltration rate coefficient (convenience accessor)."""
        return self.infiltration.alpha

    @property
    def k_P(self) -> float:
        """Vegetation half-saturation for infiltration (convenience accessor)."""
        return self.infiltration.k_P

    @property
    def W_0(self) -> float:
        """Bare soil infiltration factor (convenience accessor)."""
        return self.infiltration.W_0

    @property
    def M_sat(self) -> float:
        """Saturated soil moisture (convenience accessor)."""
        return self.soil.M_sat

    @property
    def E_max(self) -> float:
        """Maximum ET rate (convenience accessor)."""
        return self.soil.E_max

    @property
    def k_ET(self) -> float:
        """ET half-saturation (convenience accessor)."""
        return self.soil.k_ET

    @property
    def beta_ET(self) -> float:
        """Vegetation enhancement of ET (convenience accessor)."""
        return self.soil.beta_ET

    @property
    def L_max(self) -> float:
        """Leakage coefficient (convenience accessor)."""
        return self.soil.L_max

    @property
    def D_M(self) -> float:
        """Soil moisture diffusivity (convenience accessor)."""
        return self.soil.D_M

    @property
    def g_max(self) -> float:
        """Maximum growth rate (convenience accessor)."""
        return self.vegetation.g_max

    @property
    def k_G(self) -> float:
        """Growth half-saturation (convenience accessor)."""
        return self.vegetation.k_G

    @property
    def mu(self) -> float:
        """Mortality rate (convenience accessor)."""
        return self.vegetation.mu

    @property
    def D_P(self) -> float:
        """Dispersal diffusivity (convenience accessor)."""
        return self.vegetation.D_P

    @property
    def manning_n(self) -> float:
        """Manning's roughness (convenience accessor)."""
        return self.routing.manning_n

    @property
    def h_threshold(self) -> float:
        """Drainage threshold (convenience accessor)."""
        return self.drainage.h_threshold

    @property
    def drainage_time_days(self) -> float:
        """Drainage time (convenience accessor)."""
        return self.drainage.drainage_time

    @property
    def dt_veg(self) -> float:
        """Vegetation timestep (convenience accessor)."""
        return self.timestep.dt_veg

    @property
    def dt_soil(self) -> float:
        """Soil timestep (convenience accessor)."""
        return self.timestep.dt_soil

    @property
    def rain_depth(self) -> float:
        """Rain depth (convenience accessor)."""
        return self.rainfall.rain_depth

    @property
    def storm_duration(self) -> float:
        """Storm duration (convenience accessor)."""
        return self.rainfall.storm_duration

    @property
    def interstorm(self) -> float:
        """Interstorm period (convenience accessor)."""
        return self.rainfall.interstorm
