"""
Pytest fixtures and test utilities for SACO-Flow.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.config import DTYPE, init_taichi


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi once per test session with CPU backend."""
    init_taichi(backend="cpu", debug=True)
    yield


@pytest.fixture
def grid_factory():
    """Factory for creating Taichi fields of various sizes."""
    return create_fields


def create_fields(n: int = 32) -> SimpleNamespace:
    """Create a set of simulation fields."""
    fields = SimpleNamespace(n=n)

    # Primary state variables
    fields.h = ti.field(dtype=DTYPE, shape=(n, n))  # Surface water [m]
    fields.M = ti.field(dtype=DTYPE, shape=(n, n))  # Soil moisture [m]
    fields.P = ti.field(dtype=DTYPE, shape=(n, n))  # Vegetation [kg/m²]

    # Static fields
    fields.Z = ti.field(dtype=DTYPE, shape=(n, n))  # Elevation [m]
    fields.mask = ti.field(dtype=ti.i8, shape=(n, n))  # Domain mask

    # Flow directions (8 neighbors)
    fields.flow_frac = ti.field(dtype=DTYPE, shape=(n, n, 8))

    # Double buffers for stencil operations
    fields.h_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.M_new = ti.field(dtype=DTYPE, shape=(n, n))
    fields.P_new = ti.field(dtype=DTYPE, shape=(n, n))

    # Flow routing fields
    fields.q_out = ti.field(dtype=DTYPE, shape=(n, n))  # Outflow rate [m/day]
    fields.flow_acc = ti.field(dtype=DTYPE, shape=(n, n))  # Flow accumulation
    fields.flow_acc_new = ti.field(dtype=DTYPE, shape=(n, n))  # Flow acc buffer
    fields.local_source = ti.field(dtype=DTYPE, shape=(n, n))  # Local contribution

    return fields


def set_boundary_mask(fields):
    """Set mask with boundaries=0, interior=1."""
    n = fields.n
    mask = np.ones((n, n), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)


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
    set_boundary_mask(fields)


@pytest.fixture
def valley_terrain():
    """Generate V-shaped valley terrain."""
    return make_valley_terrain


def make_valley_terrain(fields, slope: float = 0.01, valley_depth: float = 0.1):
    """Create V-shaped valley converging to center column."""
    n = fields.n
    rows = np.arange(n, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float32).reshape(1, -1)
    center = n // 2

    down = (n - 1 - rows) * slope
    cross = valley_depth * np.abs(cols - center) / center
    Z = down + cross

    fields.Z.from_numpy(Z.astype(np.float32))
    set_boundary_mask(fields)


@pytest.fixture
def flat_terrain():
    """Generate flat terrain."""
    return make_flat_terrain


def make_flat_terrain(fields):
    """Create flat terrain with boundary mask."""
    n = fields.n
    fields.Z.from_numpy(np.zeros((n, n), dtype=np.float32))
    set_boundary_mask(fields)


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
