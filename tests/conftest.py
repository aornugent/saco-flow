"""
Pytest fixtures for SACO-Flow tests.

Provides:
- Taichi initialization
- Synthetic terrain generators
- Conservation assertion helpers
"""

import pytest
import numpy as np
import taichi as ti
from src.config import init_taichi


@pytest.fixture(scope="session", autouse=True)
def init_ti():
    """Initialize Taichi once for all tests."""
    config = init_taichi(debug=True)
    print(f"\nRunning tests on {config['arch']}")
    if config['device']:
        print(f"Device: {config['device']} ({config['compute_capability']})")
    return config


@pytest.fixture
def small_grid():
    """Factory for creating small test grids."""
    def _make_grid(n=32):
        """
        Create small grid of size n×n for testing.

        Args:
            n: Grid dimension (default 32×32)

        Returns:
            dict with:
                - n: grid size
                - dx: cell size (m)
                - mask: valid cells (1=active, 0=boundary)
        """
        return {
            'n': n,
            'dx': 10.0,  # 10m cells
            'mask': np.ones((n, n), dtype=np.int32),
        }
    return _make_grid


@pytest.fixture
def tilted_plane():
    """Synthetic terrain: tilted plane for flow testing."""
    def _make_terrain(n=32, slope=0.01, direction='south'):
        """
        Create tilted plane terrain.

        Args:
            n: Grid dimension
            slope: Gradient (m/m)
            direction: 'south', 'east', 'north', 'west'

        Returns:
            dict with:
                - z: elevation field (n×n)
                - n: grid size
                - dx: cell size
                - mask: valid cells
        """
        dx = 10.0
        z = np.zeros((n, n), dtype=np.float32)

        if direction == 'south':
            # Higher in north (i=0), lower in south (i=n-1)
            for i in range(n):
                z[i, :] = (n - 1 - i) * slope * dx
        elif direction == 'east':
            # Higher in west (j=0), lower in east (j=n-1)
            for j in range(n):
                z[:, j] = (n - 1 - j) * slope * dx
        elif direction == 'north':
            for i in range(n):
                z[i, :] = i * slope * dx
        elif direction == 'west':
            for j in range(n):
                z[:, j] = j * slope * dx
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return {
            'z': z,
            'n': n,
            'dx': dx,
            'mask': np.ones((n, n), dtype=np.int32),
        }
    return _make_terrain


@pytest.fixture
def valley_terrain():
    """Synthetic terrain: symmetric valley."""
    def _make_terrain(n=32, depth=5.0):
        """
        Create symmetric valley (parabolic cross-section).

        Args:
            n: Grid dimension
            depth: Valley depth at center (m)

        Returns:
            dict with z, n, dx, mask
        """
        dx = 10.0
        z = np.zeros((n, n), dtype=np.float32)
        center = n // 2

        # Parabolic valley in j-direction, tilted in i-direction
        for i in range(n):
            for j in range(n):
                dist_from_center = abs(j - center) / center
                z[i, j] = depth * dist_from_center ** 2 + (n - 1 - i) * 0.01 * dx

        return {
            'z': z,
            'n': n,
            'dx': dx,
            'mask': np.ones((n, n), dtype=np.int32),
        }
    return _make_terrain


@pytest.fixture
def hill_terrain():
    """Synthetic terrain: isolated hill."""
    def _make_terrain(n=32, height=10.0):
        """
        Create isolated conical hill.

        Args:
            n: Grid dimension
            height: Hill height (m)

        Returns:
            dict with z, n, dx, mask
        """
        dx = 10.0
        z = np.zeros((n, n), dtype=np.float32)
        center = n // 2

        for i in range(n):
            for j in range(n):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                max_dist = np.sqrt(2 * center ** 2)
                z[i, j] = height * max(0, 1 - dist / max_dist)

        return {
            'z': z,
            'n': n,
            'dx': dx,
            'mask': np.ones((n, n), dtype=np.int32),
        }
    return _make_terrain


class ConservationChecker:
    """Helper for mass conservation assertions."""

    @staticmethod
    def check_total_conserved(initial, final, flux_in=0.0, flux_out=0.0, rtol=1e-5):
        """
        Assert that mass is conserved: final = initial + flux_in - flux_out.

        Args:
            initial: Initial total mass
            final: Final total mass
            flux_in: Mass added to system
            flux_out: Mass removed from system
            rtol: Relative tolerance

        Raises:
            AssertionError if conservation violated
        """
        expected = initial + flux_in - flux_out
        error = abs(final - expected)
        rel_error = error / (abs(expected) + 1e-10)

        assert rel_error < rtol, (
            f"Mass not conserved:\n"
            f"  Initial: {initial:.6e}\n"
            f"  Final:   {final:.6e}\n"
            f"  Expected: {expected:.6e}\n"
            f"  Flux in:  {flux_in:.6e}\n"
            f"  Flux out: {flux_out:.6e}\n"
            f"  Error: {error:.6e} (rel: {rel_error:.6e})"
        )

    @staticmethod
    def check_field_sum(field, expected, rtol=1e-5):
        """
        Assert that sum of field values matches expected.

        Args:
            field: Numpy array or Taichi field
            expected: Expected sum
            rtol: Relative tolerance
        """
        if hasattr(field, 'to_numpy'):
            field = field.to_numpy()

        total = float(np.sum(field))
        error = abs(total - expected)
        rel_error = error / (abs(expected) + 1e-10)

        assert rel_error < rtol, (
            f"Field sum incorrect:\n"
            f"  Got:      {total:.6e}\n"
            f"  Expected: {expected:.6e}\n"
            f"  Error: {error:.6e} (rel: {rel_error:.6e})"
        )


@pytest.fixture
def conservation():
    """Fixture providing conservation assertion helpers."""
    return ConservationChecker()
