"""
YAML configuration loading and saving.

Provides utilities to load SimulationConfig from YAML files
and save configurations for reproducibility.
"""

from pathlib import Path
from typing import Any

import yaml

from src.params.schema import SimulationConfig, ValidationError


def load_config(path: str | Path) -> SimulationConfig:
    """
    Load simulation configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        SimulationConfig instance with validated parameters

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValidationError: If any parameter validation fails
        yaml.YAMLError: If the YAML is malformed

    Example:
        config = load_config("config/simulation.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValidationError(f"Configuration must be a dictionary, got {type(data)}")

    return SimulationConfig.from_dict(data)


def save_config(config: SimulationConfig, path: str | Path) -> None:
    """
    Save simulation configuration to a YAML file.

    Args:
        config: SimulationConfig instance to save
        path: Path to write YAML file

    Example:
        save_config(config, "config/simulation.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config_with_overrides(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SimulationConfig:
    """
    Load configuration with optional overrides.

    Useful for command-line parameter overrides or test configurations.

    Args:
        path: Optional path to base YAML file (uses defaults if None)
        overrides: Dictionary of parameter overrides to apply

    Returns:
        SimulationConfig with overrides applied

    Example:
        config = load_config_with_overrides(
            path="config/base.yaml",
            overrides={"grid": {"n": 256}, "soil": {"M_sat": 0.5}}
        )
    """
    if path is not None:
        config = load_config(path)
    else:
        config = SimulationConfig()

    if overrides:
        config = config.with_updates(**overrides)

    return config


def merge_configs(base: SimulationConfig, override: SimulationConfig) -> SimulationConfig:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration (non-default values take precedence)

    Returns:
        Merged SimulationConfig
    """
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    # Deep merge: override values replace base values
    for group in override_dict:
        for key, value in override_dict[group].items():
            base_dict[group][key] = value

    return SimulationConfig.from_dict(base_dict)
