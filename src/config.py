"""
Taichi configuration and initialization.

Environment variables:
    SACO_BACKEND: 'cuda', 'vulkan', 'cpu', or 'auto' (default)
    SACO_DEBUG: '1' to enable debug mode

GPU targets: H100 (sm_90), B200 (sm_100). Falls back to CPU if unavailable.
"""

import os
import subprocess

import taichi as ti

DTYPE = ti.f32


def get_backend() -> str:
    """Determine Taichi backend: check env var, then auto-detect."""
    env = os.environ.get("SACO_BACKEND", "auto").lower()

    if env in ("cuda", "vulkan", "cpu"):
        return env
    if env != "auto":
        raise ValueError(f"Invalid SACO_BACKEND: {env}")

    # Auto-detect CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu"


def init_taichi(backend: str | None = None, debug: bool | None = None) -> str:
    """Initialize Taichi with specified or auto-detected backend."""
    if backend is None:
        backend = get_backend()
    if debug is None:
        debug = os.environ.get("SACO_DEBUG", "0") == "1"

    arch = {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}.get(backend)
    if arch is None:
        raise ValueError(f"Unknown backend: {backend}")

    ti.init(arch=arch, default_fp=DTYPE, debug=debug, offline_cache=True, random_seed=42)
    return backend


class DefaultParams:
    """Default simulation parameters from Saco et al. (2013). Units: meters, days."""

    # Grid
    DX: float = 1.0  # cell size [m]

    # Rainfall (intensity-based)
    R_MEAN: float = 0.02  # mean event depth [m] (~20mm)
    STORM_DURATION: float = 0.25  # event duration [days] (~6 hours)
    INTERSTORM: float = 18.0  # mean interstorm period [days] (~365/20 events/year)

    # Infiltration
    K_SAT: float = 0.24  # saturated hydraulic conductivity [m/day]
    ALPHA_I: float = 2.0  # infiltration feedback strength [-]
    K_P: float = 1.0  # vegetation half-saturation for infiltration [kg/m²]
    W_0: float = 0.2  # bare soil infiltration factor [-]

    # Soil moisture
    M_SAT: float = 0.4  # saturated moisture [m]
    ET_MAX: float = 0.005  # max ET rate [m/day] (~5mm/day)
    K_ET: float = 0.1  # ET half-saturation moisture [m]
    BETA_ET: float = 0.5  # vegetation enhancement of ET [-]
    LEAKAGE: float = 0.002  # deep leakage coefficient [1/day]
    D_SOIL: float = 0.1  # soil moisture diffusivity [m²/day]

    # Vegetation
    G_MAX: float = 0.02  # max growth rate [1/day]
    K_G: float = 0.1  # growth half-saturation moisture [m]
    MORTALITY: float = 0.001  # mortality rate [1/day]
    D_VEG: float = 0.01  # seed dispersal diffusivity [m²/day]

    # Surface routing
    MANNING_N: float = 0.03  # Manning's roughness coefficient [-]
    MIN_SLOPE: float = 1e-6  # minimum slope for flow [-]

    # Drainage (for rainfall event completion)
    H_THRESHOLD: float = 1e-6  # water depth threshold [m]
    DRAINAGE_TIME: float = 1.0  # extra drainage time after event [days]
