"""Taichi field allocation and buffer management."""

from dataclasses import dataclass

import taichi as ti


@dataclass
class Fields:
    """Container for simulation fields on a square grid."""

    n: int
    z: ti.Field  # elevation [m]
    M: ti.Field  # soil moisture [mm]  (point-wise update, no buffer needed)
    V: ti.Field  # vegetation density read [g/m^2]
    V_new: ti.Field  # vegetation density write [g/m^2]
    Q_out: ti.Field  # outgoing discharge, single-buffered [mm*m/day]
    Q_daily: ti.Field  # cell-average discharge (Q_in+Q_out)/2 [mm*m/day]
    Q_annual: ti.Field  # accumulated annual discharge [mm*m/yr]
    I_inf: ti.Field  # infiltration rate [mm/day]
    R: ti.Field  # rainfall rate [mm/day]
    S: ti.Field  # sediment flux read [kg/m/day]
    S_new: ti.Field  # sediment flux write [kg/m/day]
    mask: ti.Field  # active cell mask (1=active, 0=boundary)
    flow_frac: ti.Field  # MFD flow fractions to 8 neighbors (n, n, 8)
    sorted_idx: ti.Field  # flat cell indices in topological order (n*n,)
    n_active: int = 0  # number of active cells in sorted_idx

    def swap(self, name: str):
        """Swap read/write buffers in Python scope (no copy kernel)."""
        a = getattr(self, name)
        b = getattr(self, f"{name}_new")
        setattr(self, name, b)
        setattr(self, f"{name}_new", a)


def allocate(n: int) -> Fields:
    """Allocate all Taichi fields for an n x n grid.

    Double-buffered fields: V, S (stencil/gather ops).
    Single-buffered: z, M, Q_out, I_inf, R (point-wise, wavefront, or recomputed).
    """
    return Fields(
        n=n,
        z=ti.field(ti.f32, shape=(n, n)),
        M=ti.field(ti.f32, shape=(n, n)),
        V=ti.field(ti.f32, shape=(n, n)),
        V_new=ti.field(ti.f32, shape=(n, n)),
        Q_out=ti.field(ti.f32, shape=(n, n)),
        Q_daily=ti.field(ti.f32, shape=(n, n)),
        Q_annual=ti.field(ti.f32, shape=(n, n)),
        I_inf=ti.field(ti.f32, shape=(n, n)),
        R=ti.field(ti.f32, shape=(n, n)),
        S=ti.field(ti.f32, shape=(n, n)),
        S_new=ti.field(ti.f32, shape=(n, n)),
        mask=ti.field(ti.i32, shape=(n, n)),
        flow_frac=ti.field(ti.f32, shape=(n, n, 8)),
        sorted_idx=ti.field(ti.i32, shape=(n * n,)),
    )
