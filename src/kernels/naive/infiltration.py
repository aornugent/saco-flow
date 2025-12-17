"""
Naive infiltration kernel.

Implements vegetation-enhanced infiltration following Saco et al. (2013):
I = alpha · h · [(P + k_P·W_0)/(P + k_P)] · (1 - M/M_sat)⁺
"""

import taichi as ti

from src.core.dtypes import DTYPE
from src.kernels.protocol import InfiltrationKernel, InfiltrationFluxes


@ti.kernel
def infiltration_step(
    h: ti.template(),
    M: ti.template(),
    P: ti.template(),
    mask: ti.template(),
    alpha: DTYPE,
    k_P: DTYPE,
    W_0: DTYPE,
    M_sat: DTYPE,
    dt: DTYPE,
) -> DTYPE:
    """
    Compute infiltration from surface water to soil moisture.

    I = alpha · h · veg_factor · sat_factor · dt

    Where:
        veg_factor = (P + k_P·W_0) / (P + k_P)  -- vegetation enhancement
        sat_factor = max(0, 1 - M/M_sat)        -- saturation limit

    Returns total infiltrated volume for mass balance tracking.
    """
    total_infiltration = ti.cast(0.0, DTYPE)
    n = h.shape[0]

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if mask[i, j] == 0:
            continue

        h_local = h[i, j]
        M_local = M[i, j]
        P_local = P[i, j]

        if h_local <= 0 or M_local >= M_sat:
            continue

        # Vegetation enhancement factor
        # Bare soil: P=0 -> factor = W_0 (reduced infiltration)
        # Dense veg: P>>k_P -> factor approx 1 (full infiltration)
        veg_factor = (P_local + k_P * W_0) / (P_local + k_P)

        # Saturation factor (0 when saturated)
        sat_factor = ti.max(0.0, 1.0 - M_local / M_sat)

        # Potential infiltration
        I_potential = alpha * h_local * veg_factor * sat_factor * dt

        # Limit by available water and remaining capacity
        I_actual = ti.min(I_potential, h_local, M_sat - M_local)

        # Update fields
        h[i, j] = h_local - I_actual
        M[i, j] = M_local + I_actual

        ti.atomic_add(total_infiltration, I_actual)

    return total_infiltration


# Protocol-compliant wrapper


class NaiveInfiltrationKernel:
    """Naive implementation of infiltration.

    Implements the InfiltrationKernel protocol. Reference implementation
    for equivalence testing against optimized variants.
    """

    def step(self, state, static, params, dt: float) -> InfiltrationFluxes:
        """Execute one infiltration timestep.

        Args:
            state: State fields (h, m, p)
            static: Static fields (mask)
            params: Infiltration parameters or SimulationConfig
            dt: Timestep [days]

        Returns:
            InfiltrationFluxes with total_infiltration
        """
        # Extract parameters (handle both InfiltrationParams and SimulationConfig)
        if hasattr(params, "infiltration"):
            # SimulationConfig
            alpha = params.infiltration.alpha
            k_P = params.infiltration.k_P
            W_0 = params.infiltration.W_0
            M_sat = params.soil.M_sat  # M_sat is in soil params
        else:
            # Direct InfiltrationParams (need M_sat from somewhere)
            alpha = params.alpha
            k_P = params.k_P
            W_0 = params.W_0
            M_sat = getattr(params, "M_sat", 0.4)  # Default if not provided

        total_infiltration = infiltration_step(
            state.h,
            state.m,
            state.p,
            static.mask,
            alpha,
            k_P,
            W_0,
            M_sat,
            dt,
        )

        return InfiltrationFluxes(total_infiltration=float(total_infiltration))

    @property
    def fields_read(self) -> set[str]:
        """Fields read by this kernel."""
        return {"h", "m", "p", "mask"}

    @property
    def fields_written(self) -> set[str]:
        """Fields written by this kernel."""
        return {"h", "m"}
