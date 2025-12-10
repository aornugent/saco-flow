"""
Taichi initialization and GPU configuration.

Handles GPU detection, CPU fallback, and compute architecture verification.
"""

import taichi as ti
import sys


def init_taichi(arch=None, debug=False):
    """
    Initialize Taichi with GPU detection and CPU fallback.

    Args:
        arch: Optional architecture override ('cuda', 'cpu', 'vulkan', 'metal')
        debug: Enable debug mode for kernel development

    Returns:
        dict: Configuration info with 'arch', 'device', 'compute_capability'
    """
    # Determine architecture
    if arch is None:
        # Auto-detect: prefer CUDA, fallback to CPU
        try:
            ti.init(arch=ti.cuda, default_fp=ti.f32, debug=debug)
            arch_used = 'cuda'
        except Exception as e:
            print(f"CUDA initialization failed: {e}", file=sys.stderr)
            print("Falling back to CPU", file=sys.stderr)
            ti.init(arch=ti.cpu, default_fp=ti.f32, debug=debug)
            arch_used = 'cpu'
    else:
        # Use specified architecture
        arch_map = {
            'cuda': ti.cuda,
            'cpu': ti.cpu,
            'vulkan': ti.vulkan,
            'metal': ti.metal,
            'gpu': ti.gpu,
        }
        if arch not in arch_map:
            raise ValueError(f"Unknown architecture: {arch}")

        ti.init(arch=arch_map[arch], default_fp=ti.f32, debug=debug)
        arch_used = arch

    # Gather device info
    config_info = {
        'arch': arch_used,
        'device': None,
        'compute_capability': None,
    }

    # Get CUDA compute capability if available
    if arch_used == 'cuda':
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            device_name = pynvml.nvmlDeviceGetName(handle)
            compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            config_info['device'] = device_name
            config_info['compute_capability'] = f"sm_{compute_cap[0]}{compute_cap[1]}"
            pynvml.nvmlShutdown()
        except ImportError:
            print("pynvml not available, cannot query GPU details", file=sys.stderr)
        except Exception as e:
            print(f"Failed to query GPU details: {e}", file=sys.stderr)

    return config_info


def verify_sm90_sm100():
    """
    Verify compatibility with NVIDIA H100 (sm90) or B200 (sm100).

    Returns:
        bool: True if running on sm90/sm100, False otherwise
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()

        # sm90 = compute capability 9.0, sm100 = 10.0
        return major >= 9
    except Exception:
        return False


# Global configuration
_config = None


def get_config():
    """Get current Taichi configuration."""
    global _config
    if _config is None:
        _config = init_taichi()
    return _config
