"""
Pytest configuration and fixtures for SACO-Flow tests.

Provides:
- Taichi initialization (CPU backend for CI)
- Grid factory for creating test fields
- Synthetic terrain generators
- Mass conservation assertion helpers
"""

import numpy as np
import pytest
import taichi as ti

from src.config import DTYPE, init_taichi

# ---------------------------------------------------------------------------
# Taichi Initialization
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi once per test session with CPU backend."""
    # Always use CPU for tests to ensure CI compatibility
    init_taichi(backend="cpu", debug=True)
    yield


# ---------------------------------------------------------------------------
# Grid Factory
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_factory():
    """
    Factory for creating Taichi fields of various sizes.

    Usage:
        def test_something(grid_factory):
            fields = grid_factory(n=64)
            # fields.h, fields.M, fields.P, fields.Z, fields.mask available
    """
    return _GridFactory()


class _GridFactory:
    """Creates simulation fields for testing."""

    def __call__(self, n: int = 32):
        """
        Create a set of simulation fields.

        Args:
            n: Grid size (n x n)

        Returns:
            SimpleNamespace with h, M, P, Z, mask, flow_frac fields
        """
        from types import SimpleNamespace

        fields = SimpleNamespace()

        # Primary state variables
        fields.h = ti.field(dtype=DTYPE, shape=(n, n))  # Surface water [m]
        fields.M = ti.field(dtype=DTYPE, shape=(n, n))  # Soil moisture [-]
        fields.P = ti.field(dtype=DTYPE, shape=(n, n))  # Vegetation [-]

        # Static fields
        fields.Z = ti.field(dtype=DTYPE, shape=(n, n))  # Elevation [m]
        fields.mask = ti.field(dtype=ti.i8, shape=(n, n))  # Domain mask

        # Flow directions (8 neighbors)
        fields.flow_frac = ti.field(dtype=DTYPE, shape=(n, n, 8))

        # Double buffer for stencil operations
        fields.h_new = ti.field(dtype=DTYPE, shape=(n, n))
        fields.M_new = ti.field(dtype=DTYPE, shape=(n, n))
        fields.P_new = ti.field(dtype=DTYPE, shape=(n, n))

        # Store grid size
        fields.n = n

        return fields


# ---------------------------------------------------------------------------
# Synthetic Terrain Generators
# ---------------------------------------------------------------------------

@pytest.fixture
def tilted_plane():
    """
    Generate tilted plane terrain (uniform slope in one direction).

    Usage:
        def test_flow(grid_factory, tilted_plane):
            fields = grid_factory(n=64)
            tilted_plane(fields, slope=0.01, direction='south')
    """
    return _tilted_plane


def _tilted_plane(fields, slope: float = 0.01, direction: str = "south"):
    """
    Create tilted plane elevation.

    Args:
        fields: Grid fields from grid_factory
        slope: Slope magnitude (rise/run)
        direction: 'north', 'south', 'east', 'west'
    """
    n = fields.n
    Z_np = np.zeros((n, n), dtype=np.float32)

    if direction == "south":
        # Higher in north (row 0), lower in south (row n-1)
        for i in range(n):
            Z_np[i, :] = (n - 1 - i) * slope
    elif direction == "north":
        for i in range(n):
            Z_np[i, :] = i * slope
    elif direction == "east":
        for j in range(n):
            Z_np[:, j] = (n - 1 - j) * slope
    elif direction == "west":
        for j in range(n):
            Z_np[:, j] = j * slope
    else:
        raise ValueError(f"Unknown direction: {direction}")

    fields.Z.from_numpy(Z_np)

    # Set mask: all cells active, interior only (boundary = 0)
    mask_np = np.ones((n, n), dtype=np.int8)
    mask_np[0, :] = 0  # North boundary
    mask_np[-1, :] = 0  # South boundary
    mask_np[:, 0] = 0  # West boundary
    mask_np[:, -1] = 0  # East boundary
    fields.mask.from_numpy(mask_np)


@pytest.fixture
def valley_terrain():
    """
    Generate V-shaped valley terrain.

    Usage:
        def test_convergent_flow(grid_factory, valley_terrain):
            fields = grid_factory(n=64)
            valley_terrain(fields, slope=0.01, valley_depth=0.1)
    """
    return _valley_terrain


def _valley_terrain(fields, slope: float = 0.01, valley_depth: float = 0.1):
    """
    Create V-shaped valley (converging to center column).

    Args:
        fields: Grid fields from grid_factory
        slope: Downslope gradient
        valley_depth: Cross-slope depth at edges
    """
    n = fields.n
    Z_np = np.zeros((n, n), dtype=np.float32)

    center_j = n // 2
    for i in range(n):
        for j in range(n):
            # Downslope component (flow toward high row numbers)
            down = (n - 1 - i) * slope
            # Cross-slope component (V shape)
            cross = valley_depth * abs(j - center_j) / center_j
            Z_np[i, j] = down + cross

    fields.Z.from_numpy(Z_np)

    # Set mask
    mask_np = np.ones((n, n), dtype=np.int8)
    mask_np[0, :] = 0
    mask_np[-1, :] = 0
    mask_np[:, 0] = 0
    mask_np[:, -1] = 0
    fields.mask.from_numpy(mask_np)


@pytest.fixture
def flat_terrain():
    """
    Generate flat terrain (for testing edge cases).

    Usage:
        def test_no_flow(grid_factory, flat_terrain):
            fields = grid_factory(n=32)
            flat_terrain(fields)
    """
    return _flat_terrain


def _flat_terrain(fields):
    """Create completely flat terrain."""
    n = fields.n
    Z_np = np.zeros((n, n), dtype=np.float32)
    fields.Z.from_numpy(Z_np)

    mask_np = np.ones((n, n), dtype=np.int8)
    mask_np[0, :] = 0
    mask_np[-1, :] = 0
    mask_np[:, 0] = 0
    mask_np[:, -1] = 0
    fields.mask.from_numpy(mask_np)


# ---------------------------------------------------------------------------
# Conservation Assertion Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_mass_conserved():
    """
    Assert that total mass is conserved within tolerance.

    Usage:
        def test_conservation(assert_mass_conserved, ...):
            initial = compute_total(fields)
            run_kernel(...)
            final = compute_total(fields)
            assert_mass_conserved(initial, final, fluxes={'outflow': out})
    """
    return _assert_mass_conserved


def _assert_mass_conserved(
    initial: float,
    final: float,
    fluxes: dict[str, float] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-10,
):
    """
    Check mass conservation.

    Args:
        initial: Total mass before operation
        final: Total mass after operation
        fluxes: Dict of flux names to values (positive = leaving system)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        AssertionError: If mass not conserved
    """
    fluxes = fluxes or {}
    total_flux = sum(fluxes.values())
    expected = initial - total_flux

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
    """
    Compute total mass in a field (considering mask).

    Usage:
        def test_conservation(compute_total_mass, ...):
            total_water = compute_total_mass(fields.h, fields.mask)
    """
    return _compute_total_mass


def _compute_total_mass(field: ti.Field, mask: ti.Field | None = None) -> float:
    """
    Sum all values in field, optionally masked.

    Args:
        field: Taichi field to sum
        mask: Optional mask (only count where mask == 1)

    Returns:
        Total mass (float)
    """
    arr = field.to_numpy()
    if mask is not None:
        mask_np = mask.to_numpy()
        return float(np.sum(arr * (mask_np == 1)))
    return float(np.sum(arr))


# ---------------------------------------------------------------------------
# Utility Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_field():
    """
    Fill a Taichi field with a uniform value.

    Usage:
        def test_uniform(uniform_field, grid_factory):
            fields = grid_factory(n=32)
            uniform_field(fields.h, value=0.01)
    """
    return _uniform_field


def _uniform_field(field: ti.Field, value: float):
    """Set all elements of field to value."""
    shape = field.shape
    arr = np.full(shape, value, dtype=np.float32)
    field.from_numpy(arr)


@pytest.fixture
def random_field():
    """
    Fill a Taichi field with random values.

    Usage:
        def test_random(random_field, grid_factory):
            fields = grid_factory(n=32)
            random_field(fields.P, min_val=0.0, max_val=1.0, seed=42)
    """
    return _random_field


def _random_field(
    field: ti.Field,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None,
):
    """Fill field with uniform random values in [min_val, max_val]."""
    if seed is not None:
        np.random.seed(seed)
    shape = field.shape
    arr = np.random.uniform(min_val, max_val, shape).astype(np.float32)
    field.from_numpy(arr)
