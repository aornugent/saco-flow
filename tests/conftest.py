"""Pytest configuration — initialize Taichi once per session."""

import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialize Taichi with CPU backend for testing."""
    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)
    yield
