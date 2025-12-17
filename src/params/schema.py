"""Parameter schema with validation. Units: meters, days, kg/m²."""

from dataclasses import dataclass, field, asdict
from typing import Any


class ValidationError(ValueError):
    """Parameter validation failed."""
    pass


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def _non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def _fraction(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class GridParams:
    """Grid: n (size), dx (cell size [m])."""
    n: int = 64
    dx: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 3:
            raise ValidationError(f"n must be >= 3, got {self.n}")
        _positive(self.dx, "dx")

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx


@dataclass(frozen=True)
class RainfallParams:
    """Rainfall: rain_depth [m], storm_duration [days], interstorm [days]."""
    rain_depth: float = 0.02
    storm_duration: float = 0.25
    interstorm: float = 18.0

    def __post_init__(self) -> None:
        _positive(self.rain_depth, "rain_depth")
        _positive(self.storm_duration, "storm_duration")
        _positive(self.interstorm, "interstorm")

    @property
    def events_per_year(self) -> float:
        return 365.0 / self.interstorm


@dataclass(frozen=True)
class InfiltrationParams:
    """Infiltration: alpha [1/day], k_P [kg/m²], W_0 [-]."""
    alpha: float = 0.1
    k_P: float = 1.0
    W_0: float = 0.2

    def __post_init__(self) -> None:
        _positive(self.alpha, "alpha")
        _positive(self.k_P, "k_P")
        _fraction(self.W_0, "W_0")


@dataclass(frozen=True)
class SoilParams:
    """Soil: M_sat [m], E_max [m/day], k_ET [m], beta_ET [-], L_max [1/day], D_M [m²/day]."""
    M_sat: float = 0.4
    E_max: float = 0.005
    k_ET: float = 0.1
    beta_ET: float = 0.5
    L_max: float = 0.002
    D_M: float = 0.1

    def __post_init__(self) -> None:
        _positive(self.M_sat, "M_sat")
        _positive(self.E_max, "E_max")
        _positive(self.k_ET, "k_ET")
        _non_negative(self.beta_ET, "beta_ET")
        _non_negative(self.L_max, "L_max")
        _non_negative(self.D_M, "D_M")


@dataclass(frozen=True)
class VegetationParams:
    """Vegetation: g_max [1/day], k_G [m], mu [1/day], D_P [m²/day]."""
    g_max: float = 0.02
    k_G: float = 0.1
    mu: float = 0.001
    D_P: float = 0.01

    def __post_init__(self) -> None:
        _positive(self.g_max, "g_max")
        _positive(self.k_G, "k_G")
        _positive(self.mu, "mu")
        _non_negative(self.D_P, "D_P")


@dataclass(frozen=True)
class RoutingParams:
    """Routing: manning_n [-]."""
    manning_n: float = 0.03

    def __post_init__(self) -> None:
        _positive(self.manning_n, "manning_n")


@dataclass(frozen=True)
class DrainageParams:
    """Drainage: h_threshold [m], drainage_time [days]."""
    h_threshold: float = 1e-6
    drainage_time: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.h_threshold, "h_threshold")
        _non_negative(self.drainage_time, "drainage_time")


@dataclass(frozen=True)
class TimestepParams:
    """Timestep: dt_veg [days], dt_soil [days]."""
    dt_veg: float = 7.0
    dt_soil: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.dt_veg, "dt_veg")
        _positive(self.dt_soil, "dt_soil")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulation configuration."""

    grid: GridParams = field(default_factory=GridParams)
    rainfall: RainfallParams = field(default_factory=RainfallParams)
    infiltration: InfiltrationParams = field(default_factory=InfiltrationParams)
    soil: SoilParams = field(default_factory=SoilParams)
    vegetation: VegetationParams = field(default_factory=VegetationParams)
    routing: RoutingParams = field(default_factory=RoutingParams)
    drainage: DrainageParams = field(default_factory=DrainageParams)
    timestep: TimestepParams = field(default_factory=TimestepParams)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary."""
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
        """Create from nested dictionary."""
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
        kwargs = {k: param_classes[k](**data[k]) for k in data if k in param_classes}
        return cls(**kwargs)

    def with_updates(self, **kwargs: Any) -> "SimulationConfig":
        """Create new config with updates."""
        current = self.to_dict()
        for key, value in kwargs.items():
            if key not in current:
                raise ValidationError(f"Unknown parameter group: {key}")
            if isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = asdict(value)
        return self.from_dict(current)

    # Convenience accessors
    @property
    def n(self) -> int:
        return self.grid.n

    @property
    def dx(self) -> float:
        return self.grid.dx
