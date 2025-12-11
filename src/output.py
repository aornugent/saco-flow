"""
Output functions for saving simulation results.

Supports GeoTIFF with EPSG:3577 (Australian Albers) and PNG thumbnails.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# EPSG:3577 - Australian Albers Equal Area projection
EPSG_3577 = "EPSG:3577"


def create_vegetation_colormap():
    """
    Create a brown-green colormap for vegetation visualization.

    Brown (bare soil) -> Yellow-green (sparse) -> Dark green (dense)
    """
    colors = [
        (0.6, 0.4, 0.2),  # Brown (bare)
        (0.8, 0.7, 0.3),  # Tan
        (0.7, 0.8, 0.3),  # Yellow-green
        (0.4, 0.7, 0.3),  # Light green
        (0.2, 0.5, 0.2),  # Medium green
        (0.1, 0.3, 0.1),  # Dark green (dense)
    ]
    return LinearSegmentedColormap.from_list("vegetation", colors, N=256)


def create_elevation_colormap():
    """
    Create a terrain colormap for elevation visualization.

    Low (blue-green) -> Mid (yellow-tan) -> High (brown-white)
    """
    colors = [
        (0.2, 0.4, 0.3),  # Low (green-blue)
        (0.5, 0.6, 0.4),  # Mid-low
        (0.8, 0.7, 0.5),  # Mid (tan)
        (0.6, 0.5, 0.4),  # Mid-high (brown)
        (0.8, 0.8, 0.8),  # High (light gray)
    ]
    return LinearSegmentedColormap.from_list("terrain_elev", colors, N=256)


def get_transform(n: int, dx: float, origin_x: float = 0.0, origin_y: float = 0.0):
    """
    Create affine transform for GeoTIFF.

    Args:
        n: Grid size
        dx: Cell size [m]
        origin_x: X coordinate of lower-left corner
        origin_y: Y coordinate of lower-left corner

    Returns:
        Affine transform
    """
    # Calculate bounds
    width = n * dx
    height = n * dx

    # Note: rasterio uses top-left origin, so we flip y
    return from_bounds(
        origin_x,
        origin_y,
        origin_x + width,
        origin_y + height,
        n,
        n,
    )


def save_geotiff(
    data: np.ndarray,
    filepath: str | Path,
    dx: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    crs: str = EPSG_3577,
    nodata: float | None = None,
) -> None:
    """
    Save 2D array as GeoTIFF with specified CRS.

    Args:
        data: 2D numpy array
        filepath: Output path
        dx: Cell size [m]
        origin_x: X coordinate of lower-left corner
        origin_y: Y coordinate of lower-left corner
        crs: Coordinate reference system (default: EPSG:3577)
        nodata: NoData value
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF output")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n = data.shape[0]
    transform = get_transform(n, dx, origin_x, origin_y)

    # Flip data vertically for correct orientation (rasterio uses top-left origin)
    data_flipped = np.flipud(data)

    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": n,
        "height": n,
        "count": 1,
        "crs": CRS.from_string(crs),
        "transform": transform,
        "compress": "lzw",
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(data_flipped, 1)


def save_thumbnail(
    data: np.ndarray,
    filepath: str | Path,
    colormap: str | LinearSegmentedColormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 100,
) -> None:
    """
    Save 2D array as PNG thumbnail with colormap.

    Args:
        data: 2D numpy array
        filepath: Output path
        colormap: Matplotlib colormap name or object
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        title: Optional title
        figsize: Figure size in inches
        dpi: Resolution
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for thumbnail output")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(im, ax=ax, shrink=0.8)

    if title:
        ax.set_title(title)

    ax.set_xlabel("X [cells]")
    ax.set_ylabel("Y [cells]")

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_simulation_output(
    fields: SimpleNamespace,
    output_dir: str | Path,
    prefix: str = "sim",
    dx: float = 1.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    day: float | None = None,
) -> dict[str, Path]:
    """
    Save simulation fields (Z, P) as GeoTIFF and PNG thumbnails.

    Args:
        fields: SimpleNamespace containing Taichi fields
        output_dir: Output directory
        prefix: Filename prefix
        dx: Cell size [m]
        origin_x: X coordinate of lower-left corner
        origin_y: Y coordinate of lower-left corner
        day: Optional day number for filename

    Returns:
        Dictionary mapping field names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename suffix
    suffix = f"_day{int(day):06d}" if day is not None else ""

    outputs = {}

    # Extract numpy arrays
    Z_np = fields.Z.to_numpy().astype(np.float32)
    P_np = fields.P.to_numpy().astype(np.float32)
    mask_np = fields.mask.to_numpy()

    # Apply mask (set boundaries to NaN for visualization)
    Z_masked = np.where(mask_np == 1, Z_np, np.nan)
    P_masked = np.where(mask_np == 1, P_np, np.nan)

    # Save elevation GeoTIFF
    Z_path = output_dir / f"{prefix}_Z{suffix}.tif"
    save_geotiff(Z_np, Z_path, dx=dx, origin_x=origin_x, origin_y=origin_y, nodata=-9999)
    outputs["Z_tif"] = Z_path

    # Save vegetation GeoTIFF
    P_path = output_dir / f"{prefix}_P{suffix}.tif"
    save_geotiff(P_np, P_path, dx=dx, origin_x=origin_x, origin_y=origin_y, nodata=-9999)
    outputs["P_tif"] = P_path

    # Save thumbnails if matplotlib available
    if HAS_MATPLOTLIB:
        # Elevation thumbnail
        Z_png_path = output_dir / f"{prefix}_Z{suffix}.png"
        save_thumbnail(
            Z_masked,
            Z_png_path,
            colormap=create_elevation_colormap(),
            title=f"Elevation [m]{f' (day {int(day)})' if day else ''}",
        )
        outputs["Z_png"] = Z_png_path

        # Vegetation thumbnail
        P_png_path = output_dir / f"{prefix}_P{suffix}.png"
        save_thumbnail(
            P_masked,
            P_png_path,
            colormap=create_vegetation_colormap(),
            vmin=0.0,
            title=f"Vegetation [kg/m²]{f' (day {int(day)})' if day else ''}",
        )
        outputs["P_png"] = P_png_path

    return outputs
