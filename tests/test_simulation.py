"""Integration tests for the simulation loop."""

import tempfile

import numpy as np
import pytest

from src.config import init_taichi
from src.params import SimulationParams
from src.diagnostics import MassBalance
from src.fields import allocate, initialize_tilted_plane, initialize_vegetation, fill_field
from src.simulation import Simulation, SimulationState
from src.output import save_simulation_output, HAS_RASTERIO, HAS_MATPLOTLIB


class TestMassBalance:
    """Test mass balance tracking."""

    def test_expected_water_calculation(self):
        """Expected water should account for all fluxes."""
        mb = MassBalance(
            initial_water=100.0,
            cumulative_rain=20.0,
            cumulative_et=5.0,
            cumulative_leakage=3.0,
            cumulative_outflow=10.0,
        )

        expected = 100.0 + 20.0 - 5.0 - 3.0 - 10.0
        assert mb.expected_water() == expected

    def test_check_passes_when_balanced(self):
        """Check should pass when actual matches expected."""
        mb = MassBalance(initial_water=100.0)
        error = mb.check(100.0)
        assert error < 1e-6

    def test_check_fails_when_unbalanced(self):
        """Check should raise AssertionError when unbalanced."""
        mb = MassBalance(initial_water=100.0)
        with pytest.raises(AssertionError, match="Mass conservation violated"):
            mb.check(50.0)


class TestSimulationInitialization:
    """Test simulation initialization."""

    def test_create_simulation_fields(self, taichi_init):
        """Should create all required fields."""
        fields = allocate(n=32)

        assert hasattr(fields, "h")
        assert hasattr(fields, "M")
        assert hasattr(fields, "P")
        assert hasattr(fields, "Z")
        assert hasattr(fields, "mask")
        assert hasattr(fields, "flow_frac")
        assert hasattr(fields, "q_out")

    def test_initialize_tilted_plane(self, taichi_init):
        """Tilted plane should have correct slope direction."""
        fields = allocate(n=32)
        initialize_tilted_plane(fields, slope=0.1, direction="south")

        Z_np = fields.Z.to_numpy()

        # North should be higher than south
        assert Z_np[1, 16] > Z_np[30, 16]

    def test_initialize_vegetation(self, taichi_init):
        """Vegetation should be initialized with random values."""
        fields = allocate(n=32)
        initialize_tilted_plane(fields)
        initialize_vegetation(fields, mean=1.0, std=0.2, seed=42)

        P_np = fields.P.to_numpy()

        assert np.mean(P_np) > 0.5  # Should be around mean
        assert np.std(P_np) > 0.1  # Should have variation
        assert np.all(P_np >= 0)  # Should be non-negative


class TestSimulationRun:
    """Test simulation execution."""

    def test_simulation_runs_without_crash(self, taichi_init):
        """Simulation should complete without errors."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(seed=42)

        # Run for a short time
        sim.run(years=0.1, check_mass_balance=True, verbose=False)

        assert sim.state.current_day > 0

    def test_simulation_mass_conservation(self, taichi_init):
        """Mass should be conserved throughout simulation."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        sim.initialize(seed=42)

        # Run simulation
        sim.run(years=0.2, check_mass_balance=True, verbose=False)

        # Final mass balance check
        error = sim.check_mass_balance()
        assert error < 1e-3, f"Mass balance error: {error}"

    def test_rainfall_increases_water(self, taichi_init):
        """Rainfall should increase total water in system."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        state = sim.initialize(seed=42)

        # Run a single rainfall event
        sim.run_rainfall_event(depth=0.02, duration=0.25)

        # Water should have increased (minus any outflow)
        assert state.mass_balance.cumulative_rain > 0

    def test_vegetation_responds_to_moisture(self, taichi_init):
        """Vegetation should change in response to soil moisture."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        state = sim.initialize(initial_moisture=0.2, seed=42)

        initial_P = state.fields.P.to_numpy().copy()

        # Run for some time with rainfall
        sim.run(years=0.5, check_mass_balance=False, verbose=False)

        final_P = state.fields.P.to_numpy()

        # Vegetation should have changed
        assert not np.allclose(initial_P, final_P)

    def test_water_drains_without_rainfall(self, taichi_init):
        """Water should drain and evaporate without new rainfall."""
        params = SimulationParams(n=32, interstorm=1000.0)  # Very rare rainfall
        sim = Simulation(params)
        state = sim.initialize(initial_moisture=0.3, seed=42)

        # Add some surface water
        fill_field(state.fields.h, 0.01)

        # Run without rainfall
        for _ in range(30):
            sim.step_soil(1.0)

        # M should have decreased from ET
        assert state.mass_balance.cumulative_et > 0


class TestSimulationOutput:
    """Test output file generation."""

    @pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
    def test_geotiff_output_created(self, taichi_init):
        """GeoTIFF files should be created."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        state = sim.initialize(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = save_simulation_output(
                state.fields,
                output_dir=tmpdir,
                prefix="test",
                dx=params.dx,
                day=0,
            )

            assert outputs["Z_tif"].exists()
            assert outputs["P_tif"].exists()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_thumbnail_output_created(self, taichi_init):
        """PNG thumbnails should be created."""
        params = SimulationParams(n=32)
        sim = Simulation(params)
        state = sim.initialize(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = save_simulation_output(
                state.fields,
                output_dir=tmpdir,
                prefix="test",
                dx=params.dx,
                day=0,
            )

            assert outputs["Z_png"].exists()
            assert outputs["P_png"].exists()

    @pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
    def test_geotiff_has_correct_crs(self, taichi_init):
        """GeoTIFF should have EPSG:3577 CRS."""
        import rasterio

        params = SimulationParams(n=32)
        sim = Simulation(params)
        state = sim.initialize(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = save_simulation_output(
                state.fields,
                output_dir=tmpdir,
                prefix="test",
                dx=params.dx,
            )

            with rasterio.open(outputs["Z_tif"]) as src:
                assert src.crs.to_epsg() == 3577


class TestSimulationParams:
    """Test parameter handling."""

    def test_default_params_valid(self):
        """Default parameters should be valid."""
        params = SimulationParams()

        assert params.n > 0
        assert params.dx > 0
        assert params.dt_veg > 0
        assert params.dt_soil > 0

    def test_custom_params_used(self, taichi_init):
        """Custom parameters should be used in simulation."""
        params = SimulationParams(n=48, dx=2.0)
        sim = Simulation(params)
        state = sim.initialize()

        assert state.fields.n == 48
        assert state.dx == 2.0


# Note: taichi_init fixture is provided by conftest.py (session-scoped)
