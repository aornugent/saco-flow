"""Pytest fixtures and test utilities for SACO-Flow."""


import numpy as np
import pytest
import taichi as ti

from src.config import init_taichi
from src.fields import allocate
from src.initialization import initialize_mask


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi once per test session with CPU backend."""
    init_taichi(backend="cpu", debug=True)
    yield


@pytest.fixture
def grid_factory():
    """Factory for creating Taichi fields of various sizes."""
    return allocate


@pytest.fixture
def tilted_plane():
    """Generate tilted plane terrain."""
    return make_tilted_plane


def make_tilted_plane(fields, slope: float = 0.01, direction: str = "south"):
    """Create tilted plane elevation with boundary mask."""
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


@pytest.fixture
def flat_terrain():
    """Generate flat terrain."""
    return make_flat_terrain


def make_flat_terrain(fields):
    """Create flat terrain with boundary mask."""
    n = fields.n
    fields.Z.from_numpy(np.zeros((n, n), dtype=np.float32))
    initialize_mask(fields)


@pytest.fixture
def valley_terrain():
    """Generate V-shaped valley terrain."""
    return make_valley_terrain


def make_valley_terrain(fields, slope: float = 0.01, valley_depth: float = 0.5):
    """Create V-shaped valley terrain with boundary mask."""
    n = fields.n
    rows = np.arange(n, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float32).reshape(1, -1)
    center = n // 2

    # V-shaped cross-section centered on column center
    cross_section = valley_depth * np.abs(cols - center) / center
    # Add downslope gradient
    Z = (n - 1 - rows) * slope + cross_section
    fields.Z.from_numpy(Z.astype(np.float32))
    initialize_mask(fields)


@pytest.fixture
def assert_mass_conserved():
    """Assert mass conservation within tolerance."""
    return check_mass_conserved


def check_mass_conserved(
    initial: float,
    final: float,
    fluxes: dict[str, float] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-10,
):
    """Check mass conservation: final == initial - sum(fluxes)."""
    fluxes = fluxes or {}
    expected = initial - sum(fluxes.values())
    diff = abs(final - expected)
    tol = atol + rtol * abs(expected)

    if diff > tol:
        flux_str = ", ".join(f"{k}={v:.6e}" for k, v in fluxes.items())
        raise AssertionError(
            f"Mass not conserved!\n"
            f"  Initial: {initial:.10e}\n"
            f"  Final:   {final:.10e}\n"
            f"  Expected: {expected:.10e}\n"
            f"  Fluxes: {flux_str}\n"
            f"  Difference: {diff:.10e} (tolerance: {tol:.10e})"
        )


@pytest.fixture
def compute_total_mass():
    """Compute total mass in a field (considering mask)."""
    return total_mass


def total_mass(field: ti.Field, mask: ti.Field | None = None) -> float:
    """Sum field values, optionally masked."""
    arr = field.to_numpy()
    if mask is not None:
        return float(np.sum(arr * (mask.to_numpy() == 1)))
    return float(np.sum(arr))
