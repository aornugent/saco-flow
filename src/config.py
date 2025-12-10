"""
Taichi configuration and initialization.

Handles backend detection, GPU/CPU fallback, and simulation parameters.

Environment variables:
    SACO_BACKEND: Force backend selection ('cuda', 'vulkan', 'cpu', 'auto')
    SACO_DEBUG: Enable debug mode ('1' for on)

GPU Compatibility:
    - NVIDIA H100: sm_90 (Hopper architecture)
    - NVIDIA B200: sm_100 (Blackwell architecture)
    - Falls back to CPU if no compatible GPU found
"""

import os

import taichi as ti

# Default simulation dtype
DTYPE = ti.f32

# Backend preference order for auto-detection
_BACKEND_PRIORITY = ["cuda", "vulkan", "cpu"]


def get_backend() -> str:
    """
    Determine the Taichi backend to use.

    Returns:
        Backend string: 'cuda', 'vulkan', or 'cpu'
    """
    env_backend = os.environ.get("SACO_BACKEND", "auto").lower()

    if env_backend != "auto":
        if env_backend in ("cuda", "vulkan", "cpu"):
            return env_backend
        raise ValueError(
            f"Invalid SACO_BACKEND: {env_backend}. "
            "Use 'cuda', 'vulkan', 'cpu', or 'auto'."
        )

    # Auto-detect: try backends in priority order
    for backend in _BACKEND_PRIORITY:
        if _backend_available(backend):
            return backend

    return "cpu"


def _backend_available(backend: str) -> bool:
    """Check if a backend is available without initializing Taichi."""
    if backend == "cpu":
        return True
    elif backend == "cuda":
        # Check for NVIDIA GPU via environment or nvidia-smi
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and "GPU" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    elif backend == "vulkan":
        # Vulkan detection is complex; let Taichi handle it
        # Return False to prefer CUDA over Vulkan when available
        return False
    return False


def init_taichi(backend: str | None = None, debug: bool | None = None) -> str:
    """
    Initialize Taichi with appropriate backend.

    Args:
        backend: Force specific backend, or None for auto-detection
        debug: Enable debug mode, or None to check SACO_DEBUG env var

    Returns:
        The backend that was initialized
    """
    if backend is None:
        backend = get_backend()

    if debug is None:
        debug = os.environ.get("SACO_DEBUG", "0") == "1"

    # Map backend string to Taichi arch
    arch_map = {
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "cpu": ti.cpu,
    }

    arch = arch_map.get(backend)
    if arch is None:
        raise ValueError(f"Unknown backend: {backend}")

    ti.init(
        arch=arch,
        default_fp=DTYPE,
        debug=debug,
        # Offline cache speeds up repeated runs
        offline_cache=True,
        # Random seed for reproducibility in tests
        random_seed=42,
    )

    return backend


def get_device_info() -> dict:
    """
    Get information about the current Taichi device.

    Returns:
        Dictionary with device information
    """
    # This must be called after ti.init()
    impl = ti.lang.impl.current_cfg()
    return {
        "arch": str(impl.arch),
        "default_fp": str(impl.default_fp),
        "debug": impl.debug,
    }


# Physical constants and default parameters
# Units: length in meters, time in hours, water depth in meters

class DefaultParams:
    """Default simulation parameters from Saco et al. (2013)."""

    # Grid parameters
    DX: float = 1.0  # Grid cell size [m]

    # Rainfall
    R_MEAN: float = 0.01  # Mean rainfall intensity [m/hr]
    STORM_DURATION: float = 2.0  # Storm duration [hr]
    INTERSTORM: float = 100.0  # Mean interstorm period [hr]

    # Infiltration
    K_SAT: float = 0.01  # Saturated hydraulic conductivity [m/hr]
    ALPHA_I: float = 2.0  # Infiltration feedback strength [-]

    # Soil moisture
    M_SAT: float = 0.4  # Saturated soil moisture [-]
    ET_MAX: float = 0.001  # Max evapotranspiration rate [m/hr]
    LEAKAGE: float = 0.0001  # Deep leakage rate [1/hr]
    D_SOIL: float = 0.01  # Soil moisture diffusivity [m^2/hr]

    # Vegetation
    C_GROWTH: float = 0.1  # Growth rate coefficient [1/hr]
    K_HALF: float = 0.1  # Half-saturation moisture [-]
    MORTALITY: float = 0.01  # Mortality rate [1/hr]
    D_VEG: float = 0.1  # Seed dispersal diffusivity [m^2/hr]

    # Surface routing
    MANNING_N: float = 0.03  # Manning's roughness [-]
    MIN_SLOPE: float = 1e-6  # Minimum slope for routing [-]
