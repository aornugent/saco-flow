"""Static field specifications and factory.

Static fields are initialized once and remain constant during simulation:
- z: Terrain elevation [m]
- mask: Domain mask (1=interior, 0=boundary/outside)
- flow_frac: MFD flow fractions to 8 neighbors [-]

These fields define the simulation domain geometry and routing topology.
"""

from typing import Any

import numpy as np
import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import GridGeometry, NUM_NEIGHBORS
from src.fields.base import FieldContainer, FieldRole, FieldSpec


def create_static_specs(dtype: Any = DTYPE) -> list[FieldSpec]:
    """Create specifications for static (read-only) fields.

    Static fields are set during initialization and don't change:
    - z: Terrain elevation [m]
    - mask: Domain mask (1=interior, 0=boundary/outside)
    - flow_frac: MFD flow fractions to 8 neighbors [-]

    Args:
        dtype: Floating-point type for z and flow_frac

    Returns:
        List of FieldSpec for static fields
    """
    return [
        FieldSpec(
            name="z",
            dtype=dtype,
            role=FieldRole.STATIC,
            description="Terrain elevation [m]",
        ),
        FieldSpec(
            name="mask",
            dtype=ti.i8,
            role=FieldRole.STATIC,
            description="Domain mask (1=interior, 0=boundary)",
        ),
        FieldSpec(
            name="flow_frac",
            dtype=dtype,
            role=FieldRole.STATIC,
            extra_dims=(NUM_NEIGHBORS,),
            description="MFD flow fractions to 8 neighbors [-]",
        ),
    ]


class StaticFields:
    """Convenience wrapper for accessing static fields.

    Provides typed access to static fields with initialization helpers.

    Example:
        static = StaticFields(container)
        static.initialize_tilted_plane(slope=0.01, direction="south")
        z_field = static.z
        mask_field = static.mask
    """

    def __init__(self, container: FieldContainer):
        """Initialize with field container.

        Args:
            container: Allocated FieldContainer with static fields
        """
        self._container = container

    @property
    def z(self) -> Any:
        """Terrain elevation field [m]."""
        return self._container["z"]

    @property
    def mask(self) -> Any:
        """Domain mask field (1=interior, 0=boundary)."""
        return self._container["mask"]

    @property
    def flow_frac(self) -> Any:
        """MFD flow fractions field [-, shape (nx, ny, 8)]."""
        return self._container["flow_frac"]

    @property
    def geometry(self) -> GridGeometry:
        """Get the grid geometry."""
        return self._container.geometry

    def initialize_tilted_plane(
        self,
        slope: float = 0.01,
        direction: str = "south",
    ) -> None:
        """Initialize elevation as a tilted plane with boundary mask.

        Creates a simple synthetic terrain sloping in the specified direction.
        Also initializes the mask with boundaries at edges.

        Args:
            slope: Terrain slope [m/m] (default 0.01 = 1%)
            direction: Downslope direction ("south", "north", "east", "west")

        Raises:
            ValueError: If direction is not recognized
        """
        nx, ny = self.geometry.nx, self.geometry.ny
        rows = np.arange(nx, dtype=np.float32).reshape(-1, 1)
        cols = np.arange(ny, dtype=np.float32).reshape(1, -1)

        if direction == "south":
            z_np = (nx - 1 - rows) * slope * np.ones((1, ny), dtype=np.float32)
        elif direction == "north":
            z_np = rows * slope * np.ones((1, ny), dtype=np.float32)
        elif direction == "east":
            z_np = (ny - 1 - cols) * slope * np.ones((nx, 1), dtype=np.float32)
        elif direction == "west":
            z_np = cols * slope * np.ones((nx, 1), dtype=np.float32)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        self.z.from_numpy(z_np.astype(np.float32))

        # Initialize boundary mask (0 at edges, 1 interior)
        mask_np = np.ones((nx, ny), dtype=np.int8)
        mask_np[0, :] = 0
        mask_np[-1, :] = 0
        mask_np[:, 0] = 0
        mask_np[:, -1] = 0
        self.mask.from_numpy(mask_np)

    def initialize_from_dem(
        self,
        z_array: np.ndarray,
        mask_array: np.ndarray | None = None,
    ) -> None:
        """Initialize elevation from a numpy array (e.g., loaded DEM).

        Args:
            z_array: Elevation array [m], shape must match geometry
            mask_array: Optional mask array. If None, creates boundary mask.

        Raises:
            ValueError: If array shapes don't match geometry
        """
        nx, ny = self.geometry.nx, self.geometry.ny

        if z_array.shape != (nx, ny):
            raise ValueError(
                f"DEM shape {z_array.shape} doesn't match geometry ({nx}, {ny})"
            )

        self.z.from_numpy(z_array.astype(np.float32))

        if mask_array is not None:
            if mask_array.shape != (nx, ny):
                raise ValueError(
                    f"Mask shape {mask_array.shape} doesn't match geometry ({nx}, {ny})"
                )
            self.mask.from_numpy(mask_array.astype(np.int8))
        else:
            # Default boundary mask
            mask_np = np.ones((nx, ny), dtype=np.int8)
            mask_np[0, :] = 0
            mask_np[-1, :] = 0
            mask_np[:, 0] = 0
            mask_np[:, -1] = 0
            self.mask.from_numpy(mask_np)


def create_static_container(geometry: GridGeometry) -> FieldContainer:
    """Create a container with only static fields.

    Useful for testing static field behavior in isolation.

    Args:
        geometry: Grid dimensions and cell size

    Returns:
        Allocated FieldContainer with static fields only
    """
    container = FieldContainer(geometry)
    container.register_many(create_static_specs())
    container.allocate()
    return container
