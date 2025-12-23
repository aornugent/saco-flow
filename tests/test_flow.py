"""
Tests for flow direction, accumulation, and surface water routing.
"""

import numpy as np
import pytest

from src.diagnostics import compute_total
from src.fields import fill_field
from src.kernels.flow import (
    FLOW_EXPONENT,
    compute_cfl_timestep,
    compute_flow_accumulation,
    compute_flow_directions,
    route_surface_water,
)
from src.params import SimulationParams

params = SimulationParams()


class TestFlowDirections:
    """Test MFD flow direction computation."""

    def test_tilted_plane_south_all_flow_downslope(self, grid_factory, tilted_plane):
        """On southward slope, interior cells flow south."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        frac = fields.flow_frac.to_numpy()

        # Check cells NOT adjacent to south boundary (rows 1 to n-3)
        # Row n-2 can't flow south because southern neighbors are boundary
        for i in range(1, n - 2):
            for j in range(2, n - 2):  # Also avoid east/west edges
                # Directions 1,2,3 are south-facing (SE, S, SW)
                south_flow = frac[i, j, 1] + frac[i, j, 2] + frac[i, j, 3]
                assert south_flow > 0.99, f"Cell ({i},{j}): south_flow={south_flow}"

    def test_tilted_plane_east_all_flow_downslope(self, grid_factory, tilted_plane):
        """On eastward slope, interior cells flow east."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="east")

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        frac = fields.flow_frac.to_numpy()

        # Check cells NOT adjacent to east boundary (cols 1 to n-3)
        for i in range(2, n - 2):
            for j in range(1, n - 2):
                # Directions 0,1,7 are east-facing (E, SE, NE)
                east_flow = frac[i, j, 0] + frac[i, j, 1] + frac[i, j, 7]
                assert east_flow > 0.99, f"Cell ({i},{j}): east_flow={east_flow}"

    def test_symmetric_valley_symmetric_flow(self, grid_factory, valley_terrain):
        """Valley terrain should have symmetric flow at ridge."""
        n = 32
        fields = grid_factory(n=n)
        valley_terrain(fields, slope=0.01, valley_depth=0.5)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        frac = fields.flow_frac.to_numpy()
        center = n // 2

        # Check cells away from boundaries
        for i in range(2, n - 2):
            left_j = center - 2
            right_j = center + 2

            # Cell on left should flow toward center (directions 0,1,7 = E/SE/NE)
            left_to_center = frac[i, left_j, 0] + frac[i, left_j, 1] + frac[i, left_j, 7]
            # Cell on right should flow toward center (directions 3,4,5 = SW/W/NW)
            right_to_center = (
                frac[i, right_j, 3] + frac[i, right_j, 4] + frac[i, right_j, 5]
            )

            assert left_to_center > 0.3, f"Row {i}: left_to_center={left_to_center}"
            assert right_to_center > 0.3, f"Row {i}: right_to_center={right_to_center}"

    def test_flat_terrain_flagged(self, grid_factory, flat_terrain):
        """Flat terrain should flag cells with flow_frac[i,j,0] = -1."""
        n = 16
        fields = grid_factory(n=n)
        flat_terrain(fields)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        frac = fields.flow_frac.to_numpy()
        mask = fields.mask.to_numpy()

        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if mask[i, j] == 1:
                    assert frac[i, j, 0] == -1.0, f"Cell ({i},{j}) not flagged"

    def test_flow_fractions_sum_to_one(self, grid_factory, tilted_plane):
        """Non-flat cells should have flow fractions summing to 1."""
        n = 16
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        frac = fields.flow_frac.to_numpy()
        mask = fields.mask.to_numpy()

        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if mask[i, j] == 0 or frac[i, j, 0] == -1.0:
                    continue
                total = sum(frac[i, j, k] for k in range(8))
                assert abs(total - 1.0) < 1e-5, f"Cell ({i},{j}): sum={total}"


class TestFlowAccumulation:
    """Test iterative flow accumulation."""

    def test_accumulation_conservation(self, grid_factory, tilted_plane):
        """Total accumulation at outlet should equal total input."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        fill_field(fields.local_source, 1.0)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        compute_flow_accumulation(
            fields.local_source, fields.flow_acc, fields.flow_acc_new,
            fields.flow_frac, fields.mask,
            max_iters=100, tol=1e-6,
        )

        acc = fields.flow_acc.to_numpy()
        mask = fields.mask.to_numpy()

        total_input = np.sum(mask == 1)
        bottom_acc = np.sum(acc[n - 2, :] * (mask[n - 2, :] == 1))

        assert bottom_acc > 0.5 * total_input, f"Bottom acc={bottom_acc}, input={total_input}"

    def test_accumulation_increases_downslope(self, grid_factory, tilted_plane):
        """Accumulation should increase going downslope."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        fill_field(fields.local_source, 1.0)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        compute_flow_accumulation(
            fields.local_source, fields.flow_acc, fields.flow_acc_new,
            fields.flow_frac, fields.mask,
            max_iters=100, tol=1e-6,
        )

        acc = fields.flow_acc.to_numpy()
        col = n // 2

        for i in range(3, n - 3):
            assert acc[i, col] <= acc[i + 1, col] + 0.1, f"Row {i}: not increasing"


class TestSurfaceRouting:
    """Test kinematic wave surface water routing."""

    def test_routing_mass_conservation(self, grid_factory, tilted_plane):
        """Mass balance: initial = final + boundary outflow."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        fill_field(fields.h, 0.01)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        initial_mass = compute_total(fields.h, fields.mask)

        dx = params.dx
        manning_n = params.manning_n
        dt = 0.1

        total_boundary_outflow = 0.0
        for _ in range(10):
            boundary_out = route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, dx, dt, manning_n
            )
            total_boundary_outflow += boundary_out

        final_mass = compute_total(fields.h, fields.mask)
        balance = abs(initial_mass - (final_mass + total_boundary_outflow))

        # Relaxed tolerance for floating point accumulation
        assert balance < 1e-4, f"Mass imbalance: {balance}"

    def test_water_flows_downslope(self, grid_factory, tilted_plane):
        """Water should move downslope over time."""
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        # Add water in middle rows
        h_np = np.zeros((n, n), dtype=np.float32)
        h_np[5:10, 5:27] = 0.1
        fields.h.from_numpy(h_np)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        # Compute initial center of mass (row-wise)
        mask = fields.mask.to_numpy()
        h_init = fields.h.to_numpy()
        total_init = np.sum(h_init * (mask == 1))
        row_weights = np.arange(n).reshape(-1, 1)
        com_init = np.sum(h_init * row_weights * (mask == 1)) / total_init

        dx = params.dx
        manning_n = params.manning_n

        # Route with CFL timestep for sufficient time
        for _ in range(200):
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask, dx, manning_n, cfl=0.5
            )
            if dt == float("inf"):
                break
            dt = min(dt, 0.5)
            route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, dx, dt, manning_n
            )

        # Compute final center of mass
        h_final = fields.h.to_numpy()
        total_final = np.sum(h_final * (mask == 1))
        if total_final > 1e-6:
            com_final = np.sum(h_final * row_weights * (mask == 1)) / total_final
            # Center of mass should have moved south (higher row number)
            assert com_final > com_init, f"COM: {com_init} -> {com_final}"

    def test_no_flow_on_flat_terrain(self, grid_factory, flat_terrain):
        """No flow should occur on flat terrain."""
        n = 16
        fields = grid_factory(n=n)
        flat_terrain(fields)

        fill_field(fields.h, 0.01)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        initial_mass = compute_total(fields.h, fields.mask)

        for _ in range(10):
            route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask, fields.q_out,
                params.dx, 0.1, params.manning_n
            )

        final_mass = compute_total(fields.h, fields.mask)
        assert abs(initial_mass - final_mass) < 1e-8


class TestCFLTimestep:
    """Test CFL timestep calculation."""

    def test_cfl_timestep_finite(self, grid_factory, tilted_plane):
        """CFL timestep should be finite with water present."""
        fields = grid_factory(n=32)
        tilted_plane(fields, slope=0.1, direction="south")
        fill_field(fields.h, 0.01)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        dt = compute_cfl_timestep(
            fields.h, fields.Z, fields.flow_frac, fields.mask,
            params.dx, params.manning_n, cfl=0.5,
        )

        assert 0 < dt < float("inf"), f"dt={dt}"

    def test_cfl_timestep_infinite_no_water(self, grid_factory, tilted_plane):
        """CFL timestep should be infinite with no water."""
        fields = grid_factory(n=16)
        tilted_plane(fields, slope=0.1, direction="south")
        fill_field(fields.h, 0.0)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        dt = compute_cfl_timestep(
            fields.h, fields.Z, fields.flow_frac, fields.mask,
            params.dx, params.manning_n,
        )

        assert dt == float("inf")

    def test_stability_with_cfl_timestep(self, grid_factory, tilted_plane):
        """Routing should be stable when using CFL-computed timestep."""
        fields = grid_factory(n=32)
        tilted_plane(fields, slope=0.1, direction="south")

        fill_field(fields.h, 0.05)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        dx = params.dx
        manning_n = params.manning_n

        for _ in range(50):
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask, dx, manning_n, cfl=0.5
            )
            if dt == float("inf"):
                break
            dt = min(dt, 1.0)
            route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, dx, dt, manning_n
            )

        h_final = fields.h.to_numpy()
        assert not np.any(np.isnan(h_final)), "NaN values in h"
        assert not np.any(h_final < -1e-10), "Negative values in h"


@pytest.mark.slow
class TestBoundaryOutflow:
    """Test that boundary outflow tracking is accurate."""

    def test_boundary_outflow_matches_mass_loss(self, grid_factory, tilted_plane):
        """
        Boundary outflow returned by route_surface_water should match
        the actual mass loss from the domain.

        On a tilted plane, water flows to boundary and exits. The returned
        outflow value should account for this exactly.
        """
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.1, direction="south")

        # Uniform initial water
        h_initial = 0.02
        fill_field(fields.h, h_initial)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        initial_mass = compute_total(fields.h, fields.mask)

        dx = params.dx
        manning_n = params.manning_n
        total_reported_outflow = 0.0

        # Route until drained
        for _ in range(200):
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                dx, manning_n, cfl=0.5
            )
            if dt == float("inf"):
                break
            dt = min(dt, 0.5)

            boundary_out = route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, dx, dt, manning_n
            )
            total_reported_outflow += boundary_out

        final_mass = compute_total(fields.h, fields.mask)
        actual_mass_loss = initial_mass - final_mass

        # Reported outflow should match actual mass loss
        assert abs(total_reported_outflow - actual_mass_loss) < 1e-4, (
            f"Boundary outflow mismatch: reported={total_reported_outflow:.6f}, "
            f"actual loss={actual_mass_loss:.6f}"
        )

    def test_all_water_eventually_drains(self, grid_factory, tilted_plane):
        """
        On a tilted plane with no infiltration, all water should eventually
        exit through boundaries.
        """
        n = 32
        fields = grid_factory(n=n)
        tilted_plane(fields, slope=0.05, direction="south")

        fill_field(fields.h, 0.01)

        compute_flow_directions(
            fields.Z, fields.mask, fields.flow_frac, 1.0, FLOW_EXPONENT
        )

        initial_mass = compute_total(fields.h, fields.mask)

        dx = params.dx
        manning_n = params.manning_n
        total_outflow = 0.0

        # Route for many steps
        for _ in range(1500):
            dt = compute_cfl_timestep(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                dx, manning_n, cfl=0.5
            )
            if dt == float("inf"):
                break
            dt = min(dt, 0.5)

            boundary_out = route_surface_water(
                fields.h, fields.Z, fields.flow_frac, fields.mask,
                fields.q_out, dx, dt, manning_n
            )
            total_outflow += boundary_out

        final_mass = compute_total(fields.h, fields.mask)

        # Most water should have drained (>99%)
        drained_fraction = total_outflow / initial_mass
        assert drained_fraction > 0.99, (
            f"Water didn't fully drain: {drained_fraction:.1%} drained, "
            f"remaining={final_mass:.6f}"
        )
