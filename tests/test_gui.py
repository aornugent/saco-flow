"""Tests for the visualization system (src/gui.py).

Verifies Visualizer3D instantiation, mesh generation, and color mapping logic.
"""

import pytest
import taichi as ti
import numpy as np

from src.gui import Visualizer3D
from src.fields import allocate, fill_field

@pytest.fixture
def headless_ti():
    """Initialize Taichi in headless mode for testing."""
    # Try different backends if needed, but cpu is safest for logic tests
    # Note: GGUI might require gpu. 
    ti.init(arch=ti.cpu)
    yield
    ti.reset()

def test_visualizer_instantiation(headless_ti):
    """Test that Visualizer3D can be instantiated."""
    # Headless mode should work without display
    vis = Visualizer3D(vis_n=32, window_title="Test", headless=True)
    assert vis.vis_n == 32
    assert vis.vis_verts.shape == (32 * 32,)
    
    # Verify indices init
    indices = vis.vis_indices.to_numpy()
    assert len(indices) == (31 * 31 * 6)

def test_color_mapping(headless_ti):
    """Test the Update Mesh and Color Mapping logic.
    
    We verify the mathematical correctness of color blending:
    - Deep water should be blue.
    - Dry soil should be pale sand.
    """
    # Create fields
    n = 32
    Z = ti.field(float, shape=(n, n))
    h = ti.field(float, shape=(n, n))
    M = ti.field(float, shape=(n, n))
    P = ti.field(float, shape=(n, n))
    
    # Setup test case:
    # (0,0): Deep water -> Blue
    # (0,1): Dry soil -> Sand
    # (0,2): Dense Veg -> Green
    
    Z.fill(0.0)
    h.fill(0.0)
    M.fill(0.0)
    P.fill(0.0)
    
    # Set specific values
    h[0, 0] = 5.0  # Deep water
    
    M[0, 1] = 0.0  # Dry
    
    P[0, 2] = 2.0  # Very dense veg
    M[0, 2] = 0.5  # Moderate moisture
    
    # We need to instantiate Vis to access update_mesh. 
    # If instantiation fails, we can't test. 
    # Possible workaround: Extract kernel or mock window? 
    # Taichi doesn't easily allow mocking the Window class since it's C++ bound.
    
    # Instantiate Vis in headless mode
    vis = Visualizer3D(vis_n=n, window_title="Test", headless=True)

    # Call update_mesh
    M_sat = 1.0
    vis.update_mesh(Z, h, M, P, M_sat)
    
    colors = vis.vis_colors.to_numpy()
    
    # (0,0) index in vis_mesh (assuming vis_n matches n or mapping works)
    idx_00 = 0
    c_00 = colors[idx_00]
    
    # Expected: Blue (0.10, 0.40, 0.90) (approx due to interpolation/alpha)
    # Water alpha for h=5.0 should be saturated (1.0 or close)
    # The code says if h > 1e-4: water_alpha = 0.3 + 0.7 * min(h/0.1, 1.0)
    # h=5.0 -> min(50, 1.0)=1.0 -> alpha = 0.3 + 0.7 = 1.0
    # So strictly water color.
    expected_blue = np.array([0.10, 0.40, 0.90])
    np.testing.assert_allclose(c_00, expected_blue, atol=0.01, err_msg="Deep water should be blue")
    
    # (0,1) index
    idx_01 = 1
    c_01 = colors[idx_01]
    # Expected: Dry Soil. saturation = 0. soil_color = color_dry (0.80, 0.75, 0.65)
    # veg = 0. water = 0.
    expected_sand = np.array([0.80, 0.75, 0.65])
    np.testing.assert_allclose(c_01, expected_sand, atol=0.01, err_msg="Dry soil should be sand color")

    # (0,2) index
    idx_02 = 2
    c_02 = colors[idx_02]
    # Expected: Dense Veg.
    # veg_alpha: min(P*0.8, 1.0) -> min(1.6, 1) = 1.0.
    # So purely veg color.
    # Veg color: min(P, 1.0) = 1.0 -> Dense (0.05, 0.40, 0.05)
    expected_green = np.array([0.05, 0.40, 0.05])
    np.testing.assert_allclose(c_02, expected_green, atol=0.01, err_msg="Dense veg should be green")

