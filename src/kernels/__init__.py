"""
Taichi kernels for ecohydrological simulation.

This module provides kernel implementations and a registry for selecting
between naive (reference) and optimized implementations.

Usage:
    from src.kernels import KernelRegistry, KernelVariant

    registry = KernelRegistry()
    soil_kernel = registry.get_soil(KernelVariant.NAIVE)
    result = soil_kernel.step(state, static, params, dx, dt)

Submodules:
- naive: Reference implementations (correctness first)
- optimized: Performance-optimized implementations (future)
- protocol: Kernel interfaces and result types
"""

from typing import Type, TypeVar

from src.kernels.protocol import (
    KernelVariant,
    SoilKernel,
    VegetationKernel,
    InfiltrationKernel,
    FlowKernel,
    FlowDirectionKernel,
    SoilFluxes,
    VegetationFluxes,
    InfiltrationFluxes,
    RoutingFluxes,
)
from src.kernels.naive import (
    NaiveSoilKernel,
    NaiveVegetationKernel,
    NaiveInfiltrationKernel,
    NaiveFlowKernel,
    NaiveFlowDirectionKernel,
)


T = TypeVar("T")


class KernelRegistry:
    """Registry for kernel implementations with variant selection.

    Allows runtime selection of kernel implementations (naive, fused, etc.)
    without changing orchestration code. Useful for:
    - A/B testing between implementations
    - Gradual migration to optimized kernels
    - Equivalence testing

    Example:
        registry = KernelRegistry()

        # Get default (naive) implementations
        soil = registry.get_soil()
        veg = registry.get_vegetation()

        # Get specific variant
        soil_fused = registry.get_soil(KernelVariant.FUSED)

        # Register custom implementation
        registry.register_soil(KernelVariant.FUSED, MyFusedSoilKernel)
    """

    def __init__(self):
        """Initialize registry with naive implementations."""
        # Soil kernels
        self._soil: dict[KernelVariant, Type[SoilKernel]] = {
            KernelVariant.NAIVE: NaiveSoilKernel,
        }

        # Vegetation kernels
        self._vegetation: dict[KernelVariant, Type[VegetationKernel]] = {
            KernelVariant.NAIVE: NaiveVegetationKernel,
        }

        # Infiltration kernels
        self._infiltration: dict[KernelVariant, Type[InfiltrationKernel]] = {
            KernelVariant.NAIVE: NaiveInfiltrationKernel,
        }

        # Flow routing kernels
        self._flow: dict[KernelVariant, Type[FlowKernel]] = {
            KernelVariant.NAIVE: NaiveFlowKernel,
        }

        # Flow direction kernels
        self._flow_direction: dict[KernelVariant, Type[FlowDirectionKernel]] = {
            KernelVariant.NAIVE: NaiveFlowDirectionKernel,
        }

    # Get methods (return instances)

    def get_soil(self, variant: KernelVariant = KernelVariant.NAIVE) -> SoilKernel:
        """Get a soil kernel instance.

        Args:
            variant: Implementation variant (default: NAIVE)

        Returns:
            Soil kernel instance implementing SoilKernel protocol

        Raises:
            KeyError: If variant not registered
        """
        if variant not in self._soil:
            raise KeyError(
                f"No soil kernel registered for variant {variant}. "
                f"Available: {list(self._soil.keys())}"
            )
        return self._soil[variant]()

    def get_vegetation(
        self, variant: KernelVariant = KernelVariant.NAIVE
    ) -> VegetationKernel:
        """Get a vegetation kernel instance.

        Args:
            variant: Implementation variant (default: NAIVE)

        Returns:
            Vegetation kernel instance implementing VegetationKernel protocol

        Raises:
            KeyError: If variant not registered
        """
        if variant not in self._vegetation:
            raise KeyError(
                f"No vegetation kernel registered for variant {variant}. "
                f"Available: {list(self._vegetation.keys())}"
            )
        return self._vegetation[variant]()

    def get_infiltration(
        self, variant: KernelVariant = KernelVariant.NAIVE
    ) -> InfiltrationKernel:
        """Get an infiltration kernel instance.

        Args:
            variant: Implementation variant (default: NAIVE)

        Returns:
            Infiltration kernel instance implementing InfiltrationKernel protocol

        Raises:
            KeyError: If variant not registered
        """
        if variant not in self._infiltration:
            raise KeyError(
                f"No infiltration kernel registered for variant {variant}. "
                f"Available: {list(self._infiltration.keys())}"
            )
        return self._infiltration[variant]()

    def get_flow(self, variant: KernelVariant = KernelVariant.NAIVE) -> FlowKernel:
        """Get a flow routing kernel instance.

        Args:
            variant: Implementation variant (default: NAIVE)

        Returns:
            Flow kernel instance implementing FlowKernel protocol

        Raises:
            KeyError: If variant not registered
        """
        if variant not in self._flow:
            raise KeyError(
                f"No flow kernel registered for variant {variant}. "
                f"Available: {list(self._flow.keys())}"
            )
        return self._flow[variant]()

    def get_flow_direction(
        self, variant: KernelVariant = KernelVariant.NAIVE
    ) -> FlowDirectionKernel:
        """Get a flow direction kernel instance.

        Args:
            variant: Implementation variant (default: NAIVE)

        Returns:
            FlowDirection kernel instance implementing FlowDirectionKernel protocol

        Raises:
            KeyError: If variant not registered
        """
        if variant not in self._flow_direction:
            raise KeyError(
                f"No flow direction kernel registered for variant {variant}. "
                f"Available: {list(self._flow_direction.keys())}"
            )
        return self._flow_direction[variant]()

    # Register methods (for adding implementations)

    def register_soil(
        self, variant: KernelVariant, kernel_cls: Type[SoilKernel]
    ) -> None:
        """Register a soil kernel implementation.

        Args:
            variant: Variant to register under
            kernel_cls: Kernel class implementing SoilKernel protocol
        """
        self._soil[variant] = kernel_cls

    def register_vegetation(
        self, variant: KernelVariant, kernel_cls: Type[VegetationKernel]
    ) -> None:
        """Register a vegetation kernel implementation.

        Args:
            variant: Variant to register under
            kernel_cls: Kernel class implementing VegetationKernel protocol
        """
        self._vegetation[variant] = kernel_cls

    def register_infiltration(
        self, variant: KernelVariant, kernel_cls: Type[InfiltrationKernel]
    ) -> None:
        """Register an infiltration kernel implementation.

        Args:
            variant: Variant to register under
            kernel_cls: Kernel class implementing InfiltrationKernel protocol
        """
        self._infiltration[variant] = kernel_cls

    def register_flow(
        self, variant: KernelVariant, kernel_cls: Type[FlowKernel]
    ) -> None:
        """Register a flow kernel implementation.

        Args:
            variant: Variant to register under
            kernel_cls: Kernel class implementing FlowKernel protocol
        """
        self._flow[variant] = kernel_cls

    def register_flow_direction(
        self, variant: KernelVariant, kernel_cls: Type[FlowDirectionKernel]
    ) -> None:
        """Register a flow direction kernel implementation.

        Args:
            variant: Variant to register under
            kernel_cls: Kernel class implementing FlowDirectionKernel protocol
        """
        self._flow_direction[variant] = kernel_cls

    # Query methods

    def available_variants(self, kernel_type: str) -> list[KernelVariant]:
        """List available variants for a kernel type.

        Args:
            kernel_type: One of "soil", "vegetation", "infiltration", "flow",
                        "flow_direction"

        Returns:
            List of registered variants for that kernel type

        Raises:
            ValueError: If kernel_type is not recognized
        """
        registries = {
            "soil": self._soil,
            "vegetation": self._vegetation,
            "infiltration": self._infiltration,
            "flow": self._flow,
            "flow_direction": self._flow_direction,
        }
        if kernel_type not in registries:
            raise ValueError(
                f"Unknown kernel type: {kernel_type}. "
                f"Available: {list(registries.keys())}"
            )
        return list(registries[kernel_type].keys())


# Default registry instance for convenience
_default_registry = KernelRegistry()


def get_registry() -> KernelRegistry:
    """Get the default kernel registry.

    Returns:
        The global KernelRegistry instance
    """
    return _default_registry


__all__ = [
    # Registry
    "KernelRegistry",
    "get_registry",
    # Protocol types
    "KernelVariant",
    "SoilKernel",
    "VegetationKernel",
    "InfiltrationKernel",
    "FlowKernel",
    "FlowDirectionKernel",
    # Result types
    "SoilFluxes",
    "VegetationFluxes",
    "InfiltrationFluxes",
    "RoutingFluxes",
    # Naive implementations (for direct use)
    "NaiveSoilKernel",
    "NaiveVegetationKernel",
    "NaiveInfiltrationKernel",
    "NaiveFlowKernel",
    "NaiveFlowDirectionKernel",
]
