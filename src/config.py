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
    """Default simulation parameters from Saco et al. (2013). Units: meters, hours."""

    # Grid
    DX: float = 1.0  # cell size [m]

    # Rainfall
    R_MEAN: float = 0.01  # intensity [m/hr]
    STORM_DURATION: float = 2.0  # [hr]
    INTERSTORM: float = 100.0  # mean interstorm period [hr]

    # Infiltration
    K_SAT: float = 0.01  # saturated hydraulic conductivity [m/hr]
    ALPHA_I: float = 2.0  # infiltration feedback strength [-]

    # Soil moisture
    M_SAT: float = 0.4  # saturated moisture [-]
    ET_MAX: float = 0.001  # max ET rate [m/hr]
    LEAKAGE: float = 0.0001  # deep leakage [1/hr]
    D_SOIL: float = 0.01  # diffusivity [m²/hr]

    # Vegetation
    C_GROWTH: float = 0.1  # growth coefficient [1/hr]
    K_HALF: float = 0.1  # half-saturation moisture [-]
    MORTALITY: float = 0.01  # mortality rate [1/hr]
    D_VEG: float = 0.1  # seed dispersal [m²/hr]

    # Surface routing
    MANNING_N: float = 0.03  # roughness [-]
    MIN_SLOPE: float = 1e-6  # minimum slope [-]
