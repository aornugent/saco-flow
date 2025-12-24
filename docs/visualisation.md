# Visualisation

The simulation includes a real-time 3D visualisation system built on Taichi's GGUI (`ti.ui`). This system provides immediate visual feedback on flow routing, vegetation patterning, and hydrological states without requiring external post-processing tools.


### Quickstart

```bash
# Run with 3D GUI (default 64x64 grid)
python -m src.main --gui

# Run with GUI on larger physical grid (vis decoupled at 512x512)
python -m src.main --gui --n 2048 --years 2.0

# Run headless with reproducible config
python -m src.main --config experiments/10k_100yr.toml --output results/run1
```

### Decoupled Resolution
To maintain interactive frame rates (60fps) even when the physics grid is massive ($10^4 \times 10^4$ cells), the visualisation mesh resolution ($M \times M$) is decoupled from the physics grid resolution ($N \times N$).

*   **Physics Grid ($N$):** Defined by `SimulationParams.n`. Can be arbitrarily large.
*   **Vis Mesh ($M$):** Fixed at instantiation (default $512$).
*   **Sampling:** A dedicated kernel samples the high-resolution physics fields to update the lower-resolution visualisation mesh.

### Zero-Copy Rendering
All rendering data remains on the GPU. The `update_mesh` kernel reads directly from the simulation state fields (`Z`, `h`, `P`, `M`) and writes to the visualisation vertex/color buffers. No data is transferred to the CPU during the render loop.

## Visual Language

The visualisation renders the landscape using a layered material system to distinguish between terrain, soil moisture, vegetation, and surface water.

### Geometry (Dimensionality)
The vertical position of vertices is determined by:

$$ Y_{vis} = (Z \times S_{vert}) + \delta_{water} $$

*   **Vertical Exaggeration ($S_{vert}$):** Terrain elevation $Z$ is scaled by **2.0x** to emphasize topography.
*   **Water Depth ($\delta_{water}$):** Surface water $h$ is non-linearly scaled to ensure visibility.
    *   If $h < 1mm$: $\delta_{water} = 0$
    *   If $h \ge 1mm$: $\delta_{water} = 0.1 + 0.5 \times \min(h, 1.0)$
    *   This renders thin sheets of runoff as a distinct physical layer floating above the terrain.

### Materiality (Color)
Colors are calculated by blending three distinct material layers:

1.  **Soil Layer (Base):**
    *   Interpolates between **Pale Sand** (Dry) and **Dark Mud** (Saturated) based on soil moisture saturation ($M / M_{sat}$).
    *   Dry: `(0.80, 0.75, 0.65)`
    *   Wet: `(0.35, 0.25, 0.15)`

2.  **Vegetation Layer (Overlay):**
    *   Acts as a semi-transparent coverage layer on top of the soil.
    *   Interpolates between **Sparse** (Yellow-Green) and **Dense** (Forest Green) based on biomass $P$.
    *   Sparse: `(0.60, 0.70, 0.20)`
    *   Dense: `(0.05, 0.40, 0.05)`
    *   Opacity scales with biomass.

3.  **Water Layer (Top):**
    *   Acts as a glossy, transmissive layer on top of vegetation and soil.
    *   Color: **Electric Blue** `(0.10, 0.40, 0.90)`
    *   Opacity scales with water depth $h$ (deeper water is more opaque).

## Controls

*   **RMB + Drag:** Orbit camera
*   **Wheel:** Zoom in/out
*   **ESC:** Close window and terminate simulation