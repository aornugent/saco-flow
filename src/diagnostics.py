"""Conservation checks and timing utilities.

Simple functions for mass conservation verification.
"""

from dataclasses import dataclass

import taichi as ti

from src.geometry import DTYPE


@dataclass
class MassBalance:
    """Tracks cumulative fluxes for mass conservation verification."""

    initial_water: float = 0.0  # h + M at start [m^3]
    cumulative_rain: float = 0.0  # total rainfall [m^3]
    cumulative_et: float = 0.0  # total evapotranspiration [m^3]
    cumulative_leakage: float = 0.0  # total deep leakage [m^3]
    cumulative_outflow: float = 0.0  # total boundary outflow [m^3]

    def expected_water(self) -> float:
        """Compute expected total water based on fluxes."""
        return (
            self.initial_water
            + self.cumulative_rain
            - self.cumulative_et
            - self.cumulative_leakage
            - self.cumulative_outflow
        )

    def check(self, actual: float, rtol: float = 1e-4, atol: float = 1e-8) -> float:
        """Check mass conservation and return relative error.

        Args:
            actual: Current total water (h + M) [m^3]
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Relative error

        Raises:
            AssertionError: If mass balance violated beyond tolerance
        """
        expected = self.expected_water()
        error = abs(actual - expected)
        tol = atol + rtol * abs(expected)

        if error > tol:
            raise AssertionError(
                f"Mass conservation violated!\n"
                f"  Expected: {expected:.6e} m^3\n"
                f"  Actual:   {actual:.6e} m^3\n"
                f"  Error:    {error:.6e} (tolerance: {tol:.6e})\n"
                f"  Rain:     {self.cumulative_rain:.6e}\n"
                f"  ET:       {self.cumulative_et:.6e}\n"
                f"  Leakage:  {self.cumulative_leakage:.6e}\n"
                f"  Outflow:  {self.cumulative_outflow:.6e}"
            )

        return error / max(expected, 1e-10)


@ti.kernel
def compute_total(field: ti.template(), mask: ti.template()) -> DTYPE:
    """Sum field values where mask == 1."""
    total = ti.cast(0.0, DTYPE)
    for I in ti.grouped(field):
        if mask[I] == 1:
            total += field[I]
    return total


def check_conservation(
    initial: float,
    final: float,
    fluxes: dict[str, float] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-10,
) -> None:
    """Check mass conservation: final == initial - sum(fluxes).

    Args:
        initial: Initial total mass
        final: Final total mass
        fluxes: Dict of flux name -> value (positive = loss)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        AssertionError: If conservation violated
    """
    fluxes = fluxes or {}
    expected = initial - sum(fluxes.values())
    diff = abs(final - expected)
    tol = atol + rtol * abs(expected)

    if diff > tol:
        flux_str = ", ".join(f"{k}={v:.6e}" for k, v in fluxes.items())
        raise AssertionError(
            f"Mass not conserved!\n"
            f"  Initial: {initial:.10e}\n"
            f"  Final:   {final:.10e}\n"
            f"  Expected: {expected:.10e}\n"
            f"  Fluxes: {flux_str}\n"
            f"  Difference: {diff:.10e} (tolerance: {tol:.10e})"
        )
