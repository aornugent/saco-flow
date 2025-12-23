"""All Taichi fields with ping-pong buffer management.

Fields are organized by purpose:
- State fields: h, M, P (double-buffered for stencil operations)
- Static fields: Z, mask, flow_frac (read-only during simulation)
- Intermediate: q_out, flow_acc (for routing)
"""

from types import SimpleNamespace

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
