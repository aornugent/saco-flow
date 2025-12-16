"""Tests for core geometry module.

Tests GridGeometry dataclass, neighbor vectors, and helper functions.
"""

from math import sqrt

import numpy as np
import pytest
import taichi as ti

from src.core.dtypes import DTYPE
from src.core.geometry import (
    NEIGHBOR_DI,
    NEIGHBOR_DJ,
    NEIGHBOR_DIST,
    NUM_NEIGHBORS,
    SQRT2,
    GridGeometry,
    get_neighbor,
    get_neighbor_distance,
    is_interior,
)


class TestGridGeometry:
    """Tests for GridGeometry dataclass."""

    def test_basic_creation(self):
        """Test basic grid creation with defaults."""
        grid = GridGeometry(nx=100, ny=100)
        assert grid.nx == 100
        assert grid.ny == 100
        assert grid.dx == 1.0

    def test_custom_dx(self):
        """Test grid creation with custom cell size."""
        grid = GridGeometry(nx=50, ny=50, dx=2.5)
        assert grid.dx == 2.5

    def test_immutability(self):
        """Test that GridGeometry is frozen (immutable)."""
        grid = GridGeometry(nx=10, ny=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            grid.nx = 20

    def test_n_cells(self):
        """Test total cell count computation."""
        grid = GridGeometry(nx=10, ny=20)
        assert grid.n_cells == 200

    def test_n_interior(self):
        """Test interior cell count (excludes boundaries)."""
        grid = GridGeometry(nx=10, ny=20)
        # Interior is (10-2) * (20-2) = 8 * 18 = 144
        assert grid.n_interior == 144

    def test_domain_dimensions(self):
        """Test physical domain size computation."""
        grid = GridGeometry(nx=100, ny=200, dx=0.5)
        # Width = ny * dx = 200 * 0.5 = 100m
        # Height = nx * dx = 100 * 0.5 = 50m
        assert grid.domain_width == 100.0
        assert grid.domain_height == 50.0

    def test_shape_property(self):
        """Test shape tuple property."""
        grid = GridGeometry(nx=15, ny=25)
        assert grid.shape == (15, 25)

    def test_neighbor_distance_cardinal(self):
        """Test neighbor distance for cardinal directions."""
        grid = GridGeometry(nx=10, ny=10, dx=2.0)
        # Cardinal directions (0, 2, 4, 6) have distance = dx
        assert grid.neighbor_distance_m(0) == 2.0  # East
        assert grid.neighbor_distance_m(2) == 2.0  # South
        assert grid.neighbor_distance_m(4) == 2.0  # West
        assert grid.neighbor_distance_m(6) == 2.0  # North

    def test_neighbor_distance_diagonal(self):
        """Test neighbor distance for diagonal directions."""
        grid = GridGeometry(nx=10, ny=10, dx=2.0)
        # Diagonal directions (1, 3, 5, 7) have distance = dx * sqrt(2)
        expected = 2.0 * sqrt(2)
        assert abs(grid.neighbor_distance_m(1) - expected) < 1e-10  # SE
        assert abs(grid.neighbor_distance_m(3) - expected) < 1e-10  # SW
        assert abs(grid.neighbor_distance_m(5) - expected) < 1e-10  # NW
        assert abs(grid.neighbor_distance_m(7) - expected) < 1e-10  # NE

    def test_invalid_direction(self):
        """Test error on invalid neighbor direction."""
        grid = GridGeometry(nx=10, ny=10)
        with pytest.raises(ValueError):
            grid.neighbor_distance_m(-1)
        with pytest.raises(ValueError):
            grid.neighbor_distance_m(8)

    def test_validation_nx_too_small(self):
        """Test validation rejects nx < 3."""
        with pytest.raises(ValueError, match="nx must be >= 3"):
            GridGeometry(nx=2, ny=10)

    def test_validation_ny_too_small(self):
        """Test validation rejects ny < 3."""
        with pytest.raises(ValueError, match="ny must be >= 3"):
            GridGeometry(nx=10, ny=2)

    def test_validation_dx_nonpositive(self):
        """Test validation rejects non-positive dx."""
        with pytest.raises(ValueError, match="dx must be > 0"):
            GridGeometry(nx=10, ny=10, dx=0.0)
        with pytest.raises(ValueError, match="dx must be > 0"):
            GridGeometry(nx=10, ny=10, dx=-1.0)


class TestNeighborVectors:
    """Tests for neighbor offset vectors."""

    def test_num_neighbors(self):
        """Test 8-connectivity constant."""
        assert NUM_NEIGHBORS == 8

    def test_sqrt2_constant(self):
        """Test sqrt(2) constant accuracy."""
        assert abs(SQRT2 - sqrt(2)) < 1e-10

    def test_neighbor_di_values(self):
        """Test row offsets match 8-connectivity pattern."""
        # Expected: [0, 1, 1, 1, 0, -1, -1, -1] (E, SE, S, SW, W, NW, N, NE)
        expected = [0, 1, 1, 1, 0, -1, -1, -1]
        for k in range(8):
            assert NEIGHBOR_DI[k] == expected[k], f"DI[{k}] mismatch"

    def test_neighbor_dj_values(self):
        """Test column offsets match 8-connectivity pattern."""
        # Expected: [1, 1, 0, -1, -1, -1, 0, 1] (E, SE, S, SW, W, NW, N, NE)
        expected = [1, 1, 0, -1, -1, -1, 0, 1]
        for k in range(8):
            assert NEIGHBOR_DJ[k] == expected[k], f"DJ[{k}] mismatch"

    def test_neighbor_dist_values(self):
        """Test neighbor distances are correct."""
        for k in range(8):
            expected = SQRT2 if k % 2 == 1 else 1.0
            assert abs(NEIGHBOR_DIST[k] - expected) < 1e-6, f"DIST[{k}] mismatch"

    def test_directions_opposite(self):
        """Test that direction k and (k+4)%8 are opposites."""
        for k in range(4):
            opp = (k + 4) % 8
            # Opposite directions should have negated offsets
            assert NEIGHBOR_DI[k] == -NEIGHBOR_DI[opp]
            assert NEIGHBOR_DJ[k] == -NEIGHBOR_DJ[opp]


class TestTaichiFunctions:
    """Tests for Taichi helper functions (require Taichi initialization)."""

    def test_is_interior_center(self):
        """Test is_interior returns True for center cells."""

        @ti.kernel
        def check_center() -> ti.i32:
            return ti.cast(is_interior(5, 5, 10, 10), ti.i32)

        result = check_center()
        assert result == 1

    def test_is_interior_boundaries(self):
        """Test is_interior returns False for boundary cells."""

        @ti.kernel
        def check_boundaries() -> ti.i32:
            # Check all boundary cases
            count = 0
            # Top row
            if not is_interior(0, 5, 10, 10):
                count += 1
            # Bottom row
            if not is_interior(9, 5, 10, 10):
                count += 1
            # Left column
            if not is_interior(5, 0, 10, 10):
                count += 1
            # Right column
            if not is_interior(5, 9, 10, 10):
                count += 1
            # Corners
            if not is_interior(0, 0, 10, 10):
                count += 1
            if not is_interior(0, 9, 10, 10):
                count += 1
            if not is_interior(9, 0, 10, 10):
                count += 1
            if not is_interior(9, 9, 10, 10):
                count += 1
            return count

        result = check_boundaries()
        assert result == 8, "All 8 boundary cases should return False"

    def test_is_interior_edge_cases(self):
        """Test is_interior at row/col index 1 and n-2."""

        @ti.kernel
        def check_edges() -> ti.i32:
            count = 0
            # Just inside boundary should be interior
            if is_interior(1, 1, 10, 10):
                count += 1
            if is_interior(1, 8, 10, 10):
                count += 1
            if is_interior(8, 1, 10, 10):
                count += 1
            if is_interior(8, 8, 10, 10):
                count += 1
            return count

        result = check_edges()
        assert result == 4, "All 4 edge interior cells should return True"

    def test_get_neighbor_east(self):
        """Test get_neighbor for East direction (k=0)."""

        @ti.kernel
        def check_east() -> ti.i32:
            nb = get_neighbor(5, 5, 0)
            return nb[0] * 100 + nb[1]  # Encode as i*100+j

        result = check_east()
        assert result == 506, "East neighbor of (5,5) should be (5,6)"

    def test_get_neighbor_south(self):
        """Test get_neighbor for South direction (k=2)."""

        @ti.kernel
        def check_south() -> ti.i32:
            nb = get_neighbor(5, 5, 2)
            return nb[0] * 100 + nb[1]

        result = check_south()
        assert result == 605, "South neighbor of (5,5) should be (6,5)"

    def test_get_neighbor_all_directions(self):
        """Test get_neighbor returns correct coordinates for all 8 directions."""

        @ti.kernel
        def check_all() -> ti.i32:
            # Start from center (5, 5), check all neighbors
            errors = 0
            expected_i = ti.Vector([5, 6, 6, 6, 5, 4, 4, 4])
            expected_j = ti.Vector([6, 6, 5, 4, 4, 4, 5, 6])

            for k in ti.static(range(8)):
                nb = get_neighbor(5, 5, k)
                if nb[0] != expected_i[k] or nb[1] != expected_j[k]:
                    errors += 1
            return errors

        result = check_all()
        assert result == 0, "All 8 neighbors should match expected coordinates"

    def test_get_neighbor_distance_cardinal(self):
        """Test get_neighbor_distance for cardinal directions."""

        @ti.kernel
        def check_cardinal_dist() -> ti.f32:
            dx = ti.cast(2.0, ti.f32)
            total = ti.cast(0.0, ti.f32)
            for k in ti.static([0, 2, 4, 6]):  # Cardinal directions
                total += get_neighbor_distance(k, dx)
            return total

        result = check_cardinal_dist()
        # 4 cardinal directions * 2.0 = 8.0
        assert abs(result - 8.0) < 1e-5

    def test_get_neighbor_distance_diagonal(self):
        """Test get_neighbor_distance for diagonal directions."""

        @ti.kernel
        def check_diagonal_dist() -> ti.f32:
            dx = ti.cast(1.0, ti.f32)
            total = ti.cast(0.0, ti.f32)
            for k in ti.static([1, 3, 5, 7]):  # Diagonal directions
                total += get_neighbor_distance(k, dx)
            return total

        result = check_diagonal_dist()
        # 4 diagonal directions * sqrt(2) ≈ 5.657
        expected = 4 * sqrt(2)
        assert abs(result - expected) < 1e-5


class TestNeighborConsistency:
    """Tests verifying neighbor indexing consistency with existing code."""

    def test_neighbor_loop_all_cells(self):
        """Test that neighbor iteration covers all 8 surrounding cells."""
        n = 10

        @ti.kernel
        def count_neighbors() -> ti.i32:
            center_i, center_j = 5, 5
            count = 0
            for k in ti.static(range(8)):
                nb = get_neighbor(center_i, center_j, k)
                ni, nj = nb[0], nb[1]
                # Check neighbor is adjacent (distance ≤ sqrt(2))
                di = ni - center_i
                dj = nj - center_j
                if di * di + dj * dj <= 2:
                    count += 1
            return count

        result = count_neighbors()
        assert result == 8, "Should find exactly 8 neighbors"

    def test_neighbor_distances_scale_with_dx(self):
        """Test that physical distances scale correctly with cell size."""

        @ti.kernel
        def compute_total_distance(dx: ti.f32) -> ti.f32:
            total = ti.cast(0.0, ti.f32)
            for k in ti.static(range(8)):
                total += get_neighbor_distance(k, dx)
            return total

        dist_1m = compute_total_distance(1.0)
        dist_2m = compute_total_distance(2.0)

        # Distance should double when dx doubles
        assert abs(dist_2m - 2 * dist_1m) < 1e-5
