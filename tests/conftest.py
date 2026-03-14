"""Pytest configuration — initialize Taichi once per session."""

import numpy as np
import pytest
import taichi as ti

from src.fields import allocate
from src.flow import compute_flow_fractions


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi with CPU backend for testing."""
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)
    yield


@pytest.fixture
def grid():
    """Factory: allocate fields with boundary mask (edges=0, interior=1)."""
    def _make(n):
        fields = allocate(n)
        mask = np.ones((n, n), dtype=np.int32)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        fields.mask.from_numpy(mask)
        return fields
    return _make


@pytest.fixture
def slope_grid(grid):
    """Factory: allocate fields with linear slope and precomputed flow fractions."""
    def _make(n, dx=1.0, p=1.0, step=1.0):
        fields = grid(n)
        z = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            z[i, :] = float(n - 1 - i) * step
        fields.z.from_numpy(z)
        compute_flow_fractions(fields.z, fields.mask, fields.flow_frac, dx, p)
        return fields
    return _make
