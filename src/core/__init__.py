"""Core infrastructure: types, geometry, and constants."""

from src.core.dtypes import DTYPE
from src.core.geometry import (
    NEIGHBOR_DI,
    NEIGHBOR_DJ,
    NEIGHBOR_DIST,
    NUM_NEIGHBORS,
    GridGeometry,
    get_neighbor,
    get_neighbor_distance,
    is_interior,
)

__all__ = [
    "DTYPE",
    "GridGeometry",
    "NEIGHBOR_DI",
    "NEIGHBOR_DJ",
    "NEIGHBOR_DIST",
    "NUM_NEIGHBORS",
    "is_interior",
    "get_neighbor",
    "get_neighbor_distance",
]
