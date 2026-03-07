"""Pytest configuration — initialize Taichi once per session."""

import pytest

from src.config import init_taichi


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi with CPU backend for testing."""
    init_taichi(backend="cpu", debug=True)
    yield
