"""Taichi field allocation and buffer management."""

from dataclasses import dataclass

import taichi as ti


@dataclass
class Fields:
    """Container for simulation fields on a square grid."""

    n: int
    z: ti.Field  # elevation [m]
    M: ti.Field  # soil moisture read [m]
    M_new: ti.Field  # soil moisture write [m]
    V: ti.Field  # vegetation density read [%]
    V_new: ti.Field  # vegetation density write [%]
    Q_out: ti.Field  # outgoing discharge read [m^3/day]
    Q_out_new: ti.Field  # outgoing discharge write [m^3/day]
    Q_daily: ti.Field  # cell-average discharge (Q_in+Q_out)/2 [m^3/day]
    h: ti.Field  # flow depth [m]
    I_inf: ti.Field  # infiltration rate [m/day]
    R: ti.Field  # rainfall rate [m/day]
    S: ti.Field  # sediment flux read [kg/m/day]
    S_new: ti.Field  # sediment flux write [kg/m/day]
    mask: ti.Field  # active cell mask (1=active, 0=boundary)
    flow_frac: ti.Field  # MFD flow fractions to 8 neighbors (n, n, 8)

    def swap(self, name: str):
        """Swap read/write buffers in Python scope (no copy kernel)."""
        a = getattr(self, name)
        b = getattr(self, f"{name}_new")
        setattr(self, name, b)
        setattr(self, f"{name}_new", a)


def allocate(n: int) -> Fields:
    """Allocate all Taichi fields for an n x n grid.

    Double-buffered fields: M, V, Q_out, S (stencil/gather ops).
    Single fields: z, h, I_inf, R (point-wise or recomputed each step).
    """
    return Fields(
        n=n,
        z=ti.field(ti.f32, shape=(n, n)),
        M=ti.field(ti.f32, shape=(n, n)),
        M_new=ti.field(ti.f32, shape=(n, n)),
        V=ti.field(ti.f32, shape=(n, n)),
        V_new=ti.field(ti.f32, shape=(n, n)),
        Q_out=ti.field(ti.f32, shape=(n, n)),
        Q_out_new=ti.field(ti.f32, shape=(n, n)),
        Q_daily=ti.field(ti.f32, shape=(n, n)),
        h=ti.field(ti.f32, shape=(n, n)),
        I_inf=ti.field(ti.f32, shape=(n, n)),
        R=ti.field(ti.f32, shape=(n, n)),
        S=ti.field(ti.f32, shape=(n, n)),
        S_new=ti.field(ti.f32, shape=(n, n)),
        mask=ti.field(ti.i32, shape=(n, n)),
        flow_frac=ti.field(ti.f32, shape=(n, n, 8)),
    )
