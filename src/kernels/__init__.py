"""
Taichi kernels for ecohydrological processes.

All kernels follow these conventions:
- Use @ti.kernel decorator
- Document physics with equation in docstring
- Check mask before processing cells
- Clamp outputs to physical bounds
"""
