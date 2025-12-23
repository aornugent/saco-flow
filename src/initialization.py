"""Initialization routines for simulation fields.
"""

from types import SimpleNamespace

import numpy as np


def initialize_mask(fields: SimpleNamespace) -> None:
    """Set boundary mask: boundaries=0, interior=1."""
    n = fields.n
    mask = np.ones((n, n), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)


def initialize_tilted_plane(
    fields: SimpleNamespace,
    slope: float = 0.01,
    direction: str = "south",
) -> None:
    """Initialize elevation as a tilted plane with boundary mask.

    Args:
        fields: SimpleNamespace containing Z and mask fields
        slope: Slope gradient [-]
        direction: 'south', 'north', 'east', or 'west'
    """
    n = fields.n
    rows = np.arange(n, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float32).reshape(1, -1)

    if direction == "south":
        Z = (n - 1 - rows) * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "north":
        Z = rows * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "east":
        Z = (n - 1 - cols) * slope * np.ones((n, 1), dtype=np.float32)
    elif direction == "west":
        Z = cols * slope * np.ones((n, 1), dtype=np.float32)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    fields.Z.from_numpy(Z.astype(np.float32))
    initialize_mask(fields)


def initialize_from_dem(fields: SimpleNamespace, dem: np.ndarray) -> None:
    """Initialize elevation from DEM array.

    Args:
        fields: SimpleNamespace containing Z and mask fields
        dem: 2D numpy array of elevations [m]
    """
    if dem.shape != (fields.n, fields.n):
        raise ValueError(f"DEM shape {dem.shape} doesn't match grid ({fields.n}, {fields.n})")

    fields.Z.from_numpy(dem.astype(np.float32))
    initialize_mask(fields)


def initialize_vegetation(
    fields: SimpleNamespace,
    mean: float = 0.5,
    std: float = 0.1,
    seed: int | None = None,
) -> None:
    """Initialize vegetation with random perturbation.

    Args:
        fields: SimpleNamespace containing P field
        mean: Mean vegetation biomass [kg/m^2]
        std: Standard deviation [kg/m^2]
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    n = fields.n
    P_np = np.random.normal(mean, std, (n, n)).astype(np.float32)
    P_np = np.clip(P_np, 0.0, None)  # Ensure non-negative
    fields.P.from_numpy(P_np)
