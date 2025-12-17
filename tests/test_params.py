"""Tests for parameter management module.

Tests schema validation, YAML loading, and Taichi parameter injection.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from src.params.schema import (
    DrainageParams,
    GridParams,
    InfiltrationParams,
    RainfallParams,
    RoutingParams,
    SimulationConfig,
    SoilParams,
    TimestepParams,
    ValidationError,
    VegetationParams,
)
from src.params.loader import (
    load_config,
    load_config_with_overrides,
    merge_configs,
    save_config,
)
from src.params.taichi_params import TaichiParams, create_taichi_params


class TestGridParams:
    """Tests for GridParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = GridParams()
        assert params.n == 64
        assert params.dx == 1.0

    def test_custom_values(self):
        """Test custom parameter values."""
        params = GridParams(n=128, dx=2.0)
        assert params.n == 128
        assert params.dx == 2.0

    def test_total_area(self):
        """Test total area calculation."""
        params = GridParams(n=100, dx=10.0)
        assert params.total_area == (100 * 10.0) ** 2

    def test_cell_area(self):
        """Test cell area calculation."""
        params = GridParams(n=64, dx=5.0)
        assert params.cell_area == 25.0

    def test_validation_n_too_small(self):
        """Test validation rejects n < 3."""
        with pytest.raises(ValidationError, match="n must be >= 3"):
            GridParams(n=2)

    def test_validation_negative_dx(self):
        """Test validation rejects negative dx."""
        with pytest.raises(ValidationError, match="dx must be positive"):
            GridParams(dx=-1.0)

    def test_validation_zero_dx(self):
        """Test validation rejects zero dx."""
        with pytest.raises(ValidationError, match="dx must be positive"):
            GridParams(dx=0.0)

    def test_immutability(self):
        """Test that GridParams is frozen."""
        params = GridParams()
        with pytest.raises(Exception):
            params.n = 128


class TestRainfallParams:
    """Tests for RainfallParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = RainfallParams()
        assert params.rain_depth == 0.02
        assert params.storm_duration == 0.25
        assert params.interstorm == 18.0

    def test_events_per_year(self):
        """Test events per year calculation."""
        params = RainfallParams(interstorm=18.25)  # 365/20 = 18.25
        assert abs(params.events_per_year - 20.0) < 0.01

    def test_annual_rainfall(self):
        """Test annual rainfall calculation."""
        params = RainfallParams(rain_depth=0.02, interstorm=18.25)
        # ~20 events * 0.02m = 0.4m/year
        assert abs(params.annual_rainfall - 0.4) < 0.01

    def test_validation_negative_rain_depth(self):
        """Test validation rejects negative rain depth."""
        with pytest.raises(ValidationError, match="rain_depth must be positive"):
            RainfallParams(rain_depth=-0.01)


class TestInfiltrationParams:
    """Tests for InfiltrationParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = InfiltrationParams()
        assert params.alpha == 0.1
        assert params.k_P == 1.0
        assert params.W_0 == 0.2

    def test_validation_W_0_fraction(self):
        """Test W_0 must be in [0, 1]."""
        with pytest.raises(ValidationError, match="W_0 must be in"):
            InfiltrationParams(W_0=1.5)

    def test_W_0_boundaries(self):
        """Test W_0 at boundary values."""
        params_zero = InfiltrationParams(W_0=0.0)
        assert params_zero.W_0 == 0.0
        params_one = InfiltrationParams(W_0=1.0)
        assert params_one.W_0 == 1.0


class TestSoilParams:
    """Tests for SoilParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = SoilParams()
        assert params.M_sat == 0.4
        assert params.E_max == 0.005
        assert params.D_M == 0.1

    def test_validation_negative_leakage(self):
        """Test validation allows zero leakage."""
        params = SoilParams(L_max=0.0)
        assert params.L_max == 0.0

    def test_validation_negative_diffusivity(self):
        """Test validation allows zero diffusivity."""
        params = SoilParams(D_M=0.0)
        assert params.D_M == 0.0


class TestVegetationParams:
    """Tests for VegetationParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = VegetationParams()
        assert params.g_max == 0.02
        assert params.mu == 0.001
        assert params.D_P == 0.01

    def test_turnover_time(self):
        """Test turnover time calculation."""
        params = VegetationParams(mu=0.01)
        assert params.turnover_time == 100.0  # 1/0.01 = 100 days


class TestRoutingParams:
    """Tests for RoutingParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = RoutingParams()
        assert params.manning_n == 0.03
        assert params.min_slope == 1e-6


class TestDrainageParams:
    """Tests for DrainageParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = DrainageParams()
        assert params.h_threshold == 1e-6
        assert params.drainage_time == 1.0


class TestTimestepParams:
    """Tests for TimestepParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = TimestepParams()
        assert params.dt_veg == 7.0
        assert params.dt_soil == 1.0

    def test_validation_negative_timestep(self):
        """Test validation rejects negative timesteps."""
        with pytest.raises(ValidationError, match="dt_veg must be positive"):
            TimestepParams(dt_veg=-1.0)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_creation(self):
        """Test default configuration creation."""
        config = SimulationConfig()
        assert config.grid.n == 64
        assert config.soil.M_sat == 0.4

    def test_custom_nested_params(self):
        """Test configuration with custom nested parameters."""
        config = SimulationConfig(
            grid=GridParams(n=128, dx=2.0),
            soil=SoilParams(M_sat=0.5),
        )
        assert config.grid.n == 128
        assert config.grid.dx == 2.0
        assert config.soil.M_sat == 0.5
        # Other params should be default
        assert config.vegetation.g_max == 0.02

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimulationConfig()
        d = config.to_dict()
        assert "grid" in d
        assert "soil" in d
        assert d["grid"]["n"] == 64
        assert d["soil"]["M_sat"] == 0.4

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "grid": {"n": 256, "dx": 0.5},
            "soil": {"M_sat": 0.6},
        }
        config = SimulationConfig.from_dict(data)
        assert config.grid.n == 256
        assert config.grid.dx == 0.5
        assert config.soil.M_sat == 0.6
        # Other params should be default
        assert config.vegetation.g_max == 0.02

    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        data = {"grid": {"n": 128}}
        config = SimulationConfig.from_dict(data)
        assert config.grid.n == 128
        assert config.grid.dx == 1.0  # default

    def test_with_updates_dataclass(self):
        """Test with_updates using dataclass."""
        config = SimulationConfig()
        new_config = config.with_updates(grid=GridParams(n=256))
        assert new_config.grid.n == 256
        # Original unchanged
        assert config.grid.n == 64

    def test_with_updates_dict(self):
        """Test with_updates using dictionary."""
        config = SimulationConfig()
        new_config = config.with_updates(soil={"M_sat": 0.6})
        assert new_config.soil.M_sat == 0.6
        # Original unchanged
        assert config.soil.M_sat == 0.4

    def test_convenience_accessors(self):
        """Test convenience property accessors."""
        config = SimulationConfig(
            grid=GridParams(n=100, dx=2.0),
            infiltration=InfiltrationParams(alpha=0.5),
        )
        assert config.n == 100
        assert config.dx == 2.0
        assert config.alpha == 0.5

    def test_roundtrip_dict(self):
        """Test dictionary roundtrip preserves values."""
        original = SimulationConfig(
            grid=GridParams(n=128, dx=2.0),
            soil=SoilParams(M_sat=0.5),
        )
        d = original.to_dict()
        restored = SimulationConfig.from_dict(d)

        assert restored.grid.n == original.grid.n
        assert restored.grid.dx == original.grid.dx
        assert restored.soil.M_sat == original.soil.M_sat


class TestYamlLoader:
    """Tests for YAML loading and saving."""

    def test_save_and_load(self, tmp_path):
        """Test YAML roundtrip."""
        config = SimulationConfig(
            grid=GridParams(n=128),
            soil=SoilParams(M_sat=0.5),
        )
        yaml_path = tmp_path / "config.yaml"

        save_config(config, yaml_path)
        loaded = load_config(yaml_path)

        assert loaded.grid.n == 128
        assert loaded.soil.M_sat == 0.5

    def test_load_empty_file(self, tmp_path):
        """Test loading empty YAML file uses defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")

        config = load_config(yaml_path)
        assert config.grid.n == 64  # default

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_with_overrides(self, tmp_path):
        """Test loading with overrides."""
        yaml_path = tmp_path / "base.yaml"
        save_config(SimulationConfig(grid=GridParams(n=64)), yaml_path)

        config = load_config_with_overrides(
            path=yaml_path,
            overrides={"grid": {"n": 256}},
        )
        assert config.grid.n == 256

    def test_load_with_overrides_no_file(self):
        """Test overrides without base file."""
        config = load_config_with_overrides(
            path=None,
            overrides={"grid": {"n": 512}},
        )
        assert config.grid.n == 512

    def test_merge_configs(self):
        """Test merging two configurations."""
        base = SimulationConfig(
            grid=GridParams(n=64, dx=1.0),
            soil=SoilParams(M_sat=0.4),
        )
        override = SimulationConfig(
            grid=GridParams(n=128, dx=1.0),  # Change n
        )

        merged = merge_configs(base, override)
        assert merged.grid.n == 128
        # M_sat should come from override (has default value)
        assert merged.soil.M_sat == 0.4  # base value preserved where override has same


class TestTaichiParams:
    """Tests for Taichi parameter injection."""

    def test_creation(self):
        """Test TaichiParams creation."""
        params = TaichiParams()
        assert params.is_loaded is False

    def test_load_from_config(self):
        """Test loading from SimulationConfig."""
        config = SimulationConfig(
            grid=GridParams(dx=2.0),
            infiltration=InfiltrationParams(alpha=0.5),
            soil=SoilParams(M_sat=0.6),
        )

        params = TaichiParams()
        params.load(config)

        assert params.is_loaded is True
        assert abs(float(params.dx[None]) - 2.0) < 1e-6
        assert abs(float(params.alpha[None]) - 0.5) < 1e-6
        assert abs(float(params.M_sat[None]) - 0.6) < 1e-6

    def test_factory_function(self):
        """Test create_taichi_params factory."""
        config = SimulationConfig(grid=GridParams(dx=5.0))
        params = create_taichi_params(config)

        assert params.is_loaded is True
        assert abs(float(params.dx[None]) - 5.0) < 1e-6

    def test_to_dict(self):
        """Test extracting values as dictionary."""
        config = SimulationConfig(
            infiltration=InfiltrationParams(alpha=0.3),
        )
        params = create_taichi_params(config)

        d = params.to_dict()
        assert abs(d["alpha"] - 0.3) < 1e-6
        assert "dx" in d
        assert "M_sat" in d

    def test_all_params_loaded(self):
        """Test all parameters are properly loaded."""
        config = SimulationConfig()
        params = create_taichi_params(config)
        d = params.to_dict()

        # Check a representative sample from each category
        # Use approximate comparison due to f32 precision
        assert abs(d["dx"] - config.grid.dx) < 1e-6
        assert abs(d["alpha"] - config.infiltration.alpha) < 1e-6
        assert abs(d["M_sat"] - config.soil.M_sat) < 1e-6
        assert abs(d["g_max"] - config.vegetation.g_max) < 1e-6
        assert abs(d["manning_n"] - config.routing.manning_n) < 1e-6
        assert abs(d["h_threshold"] - config.drainage.h_threshold) < 1e-6
        assert abs(d["dt_veg"] - config.timestep.dt_veg) < 1e-6

    def test_reload_params(self):
        """Test reloading with different config."""
        config1 = SimulationConfig(soil=SoilParams(M_sat=0.4))
        config2 = SimulationConfig(soil=SoilParams(M_sat=0.8))

        params = create_taichi_params(config1)
        assert abs(float(params.M_sat[None]) - 0.4) < 1e-6

        params.load(config2)
        assert abs(float(params.M_sat[None]) - 0.8) < 1e-6


class TestValidationErrors:
    """Tests for parameter validation error handling."""

    def test_negative_parameter(self):
        """Test error message for negative parameter."""
        with pytest.raises(ValidationError) as excinfo:
            GridParams(dx=-1.0)
        assert "dx must be positive" in str(excinfo.value)

    def test_fraction_out_of_range(self):
        """Test error message for fraction out of range."""
        with pytest.raises(ValidationError) as excinfo:
            InfiltrationParams(W_0=2.0)
        assert "W_0 must be in [0, 1]" in str(excinfo.value)

    def test_invalid_from_dict(self):
        """Test validation during from_dict."""
        with pytest.raises(ValidationError):
            SimulationConfig.from_dict({"grid": {"n": 1}})

    def test_invalid_with_updates(self):
        """Test validation during with_updates."""
        config = SimulationConfig()
        with pytest.raises(ValidationError):
            config.with_updates(grid={"dx": -1.0})


class TestKernelReadability:
    """Tests verifying Taichi params can be read from kernels."""

    def test_kernel_can_read_params(self):
        """Test that kernels can read parameter values."""
        config = SimulationConfig(
            infiltration=InfiltrationParams(alpha=0.123),
        )
        params = create_taichi_params(config)

        # Create a simple field to store result
        result = ti.field(ti.f32, shape=())

        @ti.kernel
        def read_alpha():
            result[None] = params.alpha[None]

        read_alpha()

        assert abs(float(result[None]) - 0.123) < 1e-6

    def test_kernel_uses_updated_params(self):
        """Test that kernels see updated parameter values."""
        config1 = SimulationConfig(soil=SoilParams(M_sat=0.3))
        config2 = SimulationConfig(soil=SoilParams(M_sat=0.7))

        params = create_taichi_params(config1)
        result = ti.field(ti.f32, shape=())

        @ti.kernel
        def read_m_sat():
            result[None] = params.M_sat[None]

        read_m_sat()
        assert abs(float(result[None]) - 0.3) < 1e-6

        # Reload with new config
        params.load(config2)
        read_m_sat()
        assert abs(float(result[None]) - 0.7) < 1e-6
