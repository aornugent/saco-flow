"""
Kernel protocol definitions for swappable implementations.

Protocols define the interface that both naive and optimized kernels implement,
enabling variant selection at runtime without changing orchestration code.

Each kernel type has:
- step() method: Execute one timestep
- fields_read property: Fields read by this kernel (for dependency tracking)
- fields_written property: Fields written by this kernel

Result dataclasses capture flux totals for mass balance tracking.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable, Any


class KernelVariant(Enum):
    """Available kernel implementation variants."""

    NAIVE = auto()  # Reference implementation
    FUSED = auto()  # Memory-optimized fused kernels
    TEMPORAL = auto()  # Temporal blocking for diffusion


# Result dataclasses for kernel outputs


@dataclass(frozen=True)
class SoilFluxes:
    """Results from soil moisture kernel step.

    Attributes:
        total_et: Total evapotranspiration this step [m³]
        total_leakage: Total deep leakage this step [m³]
    """

    total_et: float
    total_leakage: float


@dataclass(frozen=True)
class VegetationFluxes:
    """Results from vegetation kernel step.

    Attributes:
        total_growth: Total vegetation growth this step [kg]
        total_mortality: Total vegetation mortality this step [kg]
    """

    total_growth: float
    total_mortality: float


@dataclass(frozen=True)
class InfiltrationFluxes:
    """Results from infiltration kernel step.

    Attributes:
        total_infiltration: Total water infiltrated this step [m³]
    """

    total_infiltration: float


@dataclass(frozen=True)
class RoutingFluxes:
    """Results from surface routing kernel step.

    Attributes:
        boundary_outflow: Total water leaving domain at boundaries [m³]
    """

    boundary_outflow: float


# Protocol definitions


@runtime_checkable
class SoilKernel(Protocol):
    """Protocol for soil moisture dynamics kernels.

    Handles: evapotranspiration, deep leakage, lateral diffusion.
    Equation: dM/dt = -E(M,P) - L(M) + D_M·nabla²M
    """

    def step(
        self,
        state: Any,  # StateFields
        static: Any,  # StaticFields
        params: Any,  # SoilParams or SimulationConfig
        dx: float,
        dt: float,
    ) -> SoilFluxes:
        """Execute one soil moisture timestep.

        Args:
            state: State fields container (m, p, and buffers)
            static: Static fields container (mask)
            params: Soil parameters (E_max, k_ET, beta_ET, L_max, M_sat, D_M)
            dx: Cell size [m]
            dt: Timestep [days]

        Returns:
            SoilFluxes with total_et and total_leakage
        """
        ...

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        ...

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        ...


@runtime_checkable
class VegetationKernel(Protocol):
    """Protocol for vegetation dynamics kernels.

    Handles: growth, mortality, seed dispersal.
    Equation: dP/dt = G(M)·P - mu·P + D_P·nabla²P
    """

    def step(
        self,
        state: Any,  # StateFields
        static: Any,  # StaticFields
        params: Any,  # VegetationParams or SimulationConfig
        dx: float,
        dt: float,
    ) -> VegetationFluxes:
        """Execute one vegetation timestep.

        Args:
            state: State fields container (m, p, and buffers)
            static: Static fields container (mask)
            params: Vegetation parameters (g_max, k_G, mu, D_P)
            dx: Cell size [m]
            dt: Timestep [days]

        Returns:
            VegetationFluxes with total_growth and total_mortality
        """
        ...

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        ...

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        ...


@runtime_checkable
class InfiltrationKernel(Protocol):
    """Protocol for infiltration kernels.

    Handles: surface water to soil moisture transfer.
    Equation: I = alpha·h·veg_factor·sat_factor
    """

    def step(
        self,
        state: Any,  # StateFields
        static: Any,  # StaticFields
        params: Any,  # InfiltrationParams or SimulationConfig
        dt: float,
    ) -> InfiltrationFluxes:
        """Execute one infiltration timestep.

        Args:
            state: State fields container (h, m, p)
            static: Static fields container (mask)
            params: Infiltration parameters (alpha, k_P, W_0, M_sat)
            dt: Timestep [days]

        Returns:
            InfiltrationFluxes with total_infiltration
        """
        ...

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        ...

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        ...


@runtime_checkable
class FlowKernel(Protocol):
    """Protocol for surface water routing kernels.

    Handles: MFD routing with kinematic wave approximation.
    Two-pass: compute outflow rates, then apply fluxes.
    """

    def step(
        self,
        state: Any,  # StateFields
        static: Any,  # StaticFields
        scratch: Any,  # ScratchFields
        params: Any,  # RoutingParams or SimulationConfig
        dx: float,
        dt: float,
    ) -> RoutingFluxes:
        """Execute one surface routing timestep.

        Args:
            state: State fields container (h)
            static: Static fields container (z, mask, flow_frac)
            scratch: Scratch fields container (q_out)
            params: Routing parameters (manning_n)
            dx: Cell size [m]
            dt: Timestep [days]

        Returns:
            RoutingFluxes with boundary_outflow
        """
        ...

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        ...

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        ...


@runtime_checkable
class FlowDirectionKernel(Protocol):
    """Protocol for flow direction computation.

    Computes MFD flow fractions based on terrain slope.
    Called once during initialization (static terrain assumption).
    """

    def compute(
        self,
        static: Any,  # StaticFields
        dx: float,
        p: float,
    ) -> None:
        """Compute flow directions from elevation.

        Args:
            static: Static fields container (z, mask, flow_frac)
            dx: Cell size [m]
            p: Flow exponent (1.0=diffuse, 1.5=default, >5=D8)
        """
        ...

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        ...

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        ...
