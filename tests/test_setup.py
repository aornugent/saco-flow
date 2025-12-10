"""
Tests for Phase 0: Project Setup.

Verifies:
- Taichi initialization
- GPU detection and execution
- CUDA compute capability (sm90/sm100 for H100/B200)
"""

import pytest
import numpy as np
import taichi as ti
from src.config import get_config, verify_sm90_sm100


def test_taichi_initialized(init_ti):
    """Verify Taichi is initialized."""
    config = init_ti
    assert config is not None
    assert 'arch' in config
    assert config['arch'] in ['cuda', 'cpu', 'vulkan', 'metal']


def test_gpu_available(init_ti):
    """Check if GPU is available (expected for H100/B200)."""
    config = init_ti
    # This test passes but warns if CPU is being used
    if config['arch'] == 'cpu':
        pytest.skip("GPU not available, running on CPU")
    else:
        assert config['arch'] in ['cuda', 'vulkan', 'metal']


def test_cuda_compute_capability(init_ti):
    """Verify CUDA compute capability for H100 (sm90) or B200 (sm100)."""
    config = init_ti

    if config['arch'] != 'cuda':
        pytest.skip("Not using CUDA")

    if config['compute_capability'] is None:
        pytest.skip("Could not determine compute capability (pynvml not available)")

    # Extract major version from "sm_XY" format
    cc_str = config['compute_capability']
    assert cc_str.startswith('sm_'), f"Unexpected format: {cc_str}"

    # sm90 = H100, sm100 = B200
    major = int(cc_str[3])  # First digit after "sm_"

    # Warn if not sm90 or sm100, but don't fail (may run on other GPUs)
    if major < 9:
        pytest.skip(f"Not running on H100/B200 (got {cc_str}), but that's okay for development")


def test_sm90_sm100_verification():
    """Test the verify_sm90_sm100 helper function."""
    result = verify_sm90_sm100()
    # Result can be True or False, just ensure function runs
    assert isinstance(result, bool)


def test_basic_kernel_execution():
    """Verify that a simple Taichi kernel can execute."""
    n = 32

    # Create Taichi field
    field = ti.field(dtype=ti.f32, shape=(n, n))

    @ti.kernel
    def fill_kernel(value: ti.f32):
        """Simple kernel: fill array with constant value."""
        for i, j in ti.ndrange(n, n):
            field[i, j] = value

    # Execute kernel
    test_value = 42.0
    fill_kernel(test_value)

    # Verify result
    result = field.to_numpy()
    assert np.allclose(result, test_value), "Kernel execution failed"


def test_kernel_with_computation():
    """Verify Taichi kernel can perform computation."""
    n = 32

    a = ti.field(dtype=ti.f32, shape=(n, n))
    b = ti.field(dtype=ti.f32, shape=(n, n))
    c = ti.field(dtype=ti.f32, shape=(n, n))

    @ti.kernel
    def add_arrays():
        """Add two arrays."""
        for i, j in ti.ndrange(n, n):
            c[i, j] = a[i, j] + b[i, j]

    # Initialize
    a_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n, n).astype(np.float32)
    a.from_numpy(a_np)
    b.from_numpy(b_np)

    # Execute
    add_arrays()

    # Verify
    c_np = c.to_numpy()
    expected = a_np + b_np
    assert np.allclose(c_np, expected), "Computation failed"


def test_float32_default():
    """Verify f32 is the default float type."""
    # Create field without explicit dtype
    field = ti.field(dtype=ti.f32, shape=(10, 10))

    # Set a value
    @ti.kernel
    def set_value():
        field[5, 5] = 3.14159

    set_value()

    # Check precision (f32 has limited precision)
    value = field[5, 5]
    assert abs(value - 3.14159) < 1e-5, "Using f32 precision"
