"""All Taichi fields with ping-pong buffer management.

Fields are organized by purpose:
- State fields: h, M, P (double-buffered for stencil operations)
- Static fields: Z, mask, flow_frac (read-only during simulation)
- Intermediate: q_out, flow_acc (for routing)
"""

from types import SimpleNamespace

import numpy as np
import taichi as ti

from src.geometry import DTYPE, NUM_NEIGHBORS


def allocate(n: int) -> SimpleNamespace:
    """Allocate all fields for given grid size.

    Args:
        n: Grid size (n x n cells)

    Returns:
        SimpleNamespace containing all fields with attribute access
    """
    fields = SimpleNamespace(n=n)

    # State fields (double-buffered for stencil operations)
    fields.h = ti.field(dtype=DTYPE, shape=(n, n))  # [m] surface water depth
    fields.h_new = ti.field(dtype=DTYPE, shape=(n, n))  # h buffer
    fields.M = ti.field(dtype=DTYPE, shape=(n, n))  # [m] soil moisture
    fields.M_new = ti.field(dtype=DTYPE, shape=(n, n))  # M buffer
    fields.P = ti.field(dtype=DTYPE, shape=(n, n))  # [kg/m^2] vegetation biomass
    fields.P_new = ti.field(dtype=DTYPE, shape=(n, n))  # P buffer

    # Static fields (read-only during simulation)
    fields.Z = ti.field(dtype=DTYPE, shape=(n, n))  # [m] elevation
    fields.mask = ti.field(dtype=ti.i8, shape=(n, n))  # 1=active, 0=boundary
    fields.flow_frac = ti.field(dtype=DTYPE, shape=(n, n, NUM_NEIGHBORS))  # D8 fractions

    # Intermediate fields for routing
    fields.q_out = ti.field(dtype=DTYPE, shape=(n, n))  # [m/day] outflow rate
    fields.flow_acc = ti.field(dtype=DTYPE, shape=(n, n))  # flow accumulation
    fields.flow_acc_new = ti.field(dtype=DTYPE, shape=(n, n))  # flow_acc buffer
    fields.local_source = ti.field(dtype=DTYPE, shape=(n, n))  # local contribution

    return fields


def swap_buffers(fields: SimpleNamespace, which: str) -> None:
    """Swap a field with its double buffer (O(1) pointer swap).

    Args:
        fields: SimpleNamespace containing all fields
        which: Field name to swap ('h', 'M', 'P', or 'flow_acc')
    """
    if which == "h":
        fields.h, fields.h_new = fields.h_new, fields.h
    elif which == "M":
        fields.M, fields.M_new = fields.M_new, fields.M
    elif which == "P":
        fields.P, fields.P_new = fields.P_new, fields.P
    elif which == "flow_acc":
        fields.flow_acc, fields.flow_acc_new = fields.flow_acc_new, fields.flow_acc
    else:
        raise ValueError(f"Unknown field: {which}")


def initialize_mask(fields: SimpleNamespace) -> None:
    """Set boundary mask: boundaries=0, interior=1."""
    n = fields.n
    mask = np.ones((n, n), dtype=np.int8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    fields.mask.from_numpy(mask)


def initialize_tilted_plane(
    fields: SimpleNamespace,
    slope: float = 0.01,
    direction: str = "south",
) -> None:
    """Initialize elevation as a tilted plane with boundary mask.

    Args:
        fields: SimpleNamespace containing Z and mask fields
        slope: Slope gradient [-]
        direction: 'south', 'north', 'east', or 'west'
    """
    n = fields.n
    rows = np.arange(n, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(n, dtype=np.float32).reshape(1, -1)

    if direction == "south":
        Z = (n - 1 - rows) * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "north":
        Z = rows * slope * np.ones((1, n), dtype=np.float32)
    elif direction == "east":
        Z = (n - 1 - cols) * slope * np.ones((n, 1), dtype=np.float32)
    elif direction == "west":
        Z = cols * slope * np.ones((n, 1), dtype=np.float32)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    fields.Z.from_numpy(Z.astype(np.float32))
    initialize_mask(fields)


def initialize_from_dem(fields: SimpleNamespace, dem: np.ndarray) -> None:
    """Initialize elevation from DEM array.

    Args:
        fields: SimpleNamespace containing Z and mask fields
        dem: 2D numpy array of elevations [m]
    """
    if dem.shape != (fields.n, fields.n):
        raise ValueError(f"DEM shape {dem.shape} doesn't match grid ({fields.n}, {fields.n})")

    fields.Z.from_numpy(dem.astype(np.float32))
    initialize_mask(fields)


def initialize_vegetation(
    fields: SimpleNamespace,
    mean: float = 0.5,
    std: float = 0.1,
    seed: int | None = None,
) -> None:
    """Initialize vegetation with random perturbation.

    Args:
        fields: SimpleNamespace containing P field
        mean: Mean vegetation biomass [kg/m^2]
        std: Standard deviation [kg/m^2]
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    n = fields.n
    P_np = np.random.normal(mean, std, (n, n)).astype(np.float32)
    P_np = np.clip(P_np, 0.0, None)  # Ensure non-negative
    fields.P.from_numpy(P_np)


@ti.kernel
def fill_field(field: ti.template(), value: DTYPE):
    """Set all field values to a constant."""
    for I in ti.grouped(field):
        field[I] = value


@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    """Copy src to dst."""
    for I in ti.grouped(src):
        dst[I] = src[I]


@ti.kernel
def add_uniform(field: ti.template(), mask: ti.template(), value: DTYPE):
    """Add value to field where mask == 1."""
    for I in ti.grouped(field):
        if mask[I] == 1:
            field[I] += value


@ti.kernel
def clamp_field(field: ti.template(), min_val: DTYPE, max_val: DTYPE):
    """Clamp field values to [min_val, max_val]."""
    for I in ti.grouped(field):
        field[I] = ti.max(min_val, ti.min(max_val, field[I]))
