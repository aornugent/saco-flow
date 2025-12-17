"""Taichi configuration and initialization.

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
    """Determine Taichi backend from env var or auto-detect."""
    env = os.environ.get("SACO_BACKEND", "auto").lower()

    if env in ("cuda", "vulkan", "cpu"):
        return env
    if env != "auto":
        raise ValueError(f"Invalid SACO_BACKEND: {env}")

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
