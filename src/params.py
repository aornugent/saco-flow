"""Physical constants for the Baartman et al. (2018) ecohydrological model.

Defaults correspond to Table I / Table II of the paper.
"""

from dataclasses import dataclass


@dataclass
class Params:
    """Physical constants driving the ecohydrological simulation.

    All parameters are named and unitised following Baartman et al. (2018).
    """

    # Domain
    dx: float = 5.0  # cell spacing [m]

    # Flow routing (Table I)
    p: float = 2.0  # MFD convergence exponent [-]
    n_manning: float = 0.05  # Manning's roughness [s/m^(1/3)]
    cn: float = 86400.0  # kinematic wave constant [m^(2/3) mm d⁻¹]

    # Infiltration (Table I / Table II)
    alpha: float = 8.0  # infiltration capacity [d⁻¹]
    k2: float = 18.0  # vegetation half-saturation for infiltration [g/m²]
    W0: float = 0.05  # bare-soil infiltration fraction [-]

    # Soil moisture (Table I, mm-based — I_inf converted m→mm in step_day)
    g_max: float = 0.05  # maximum growth rate [mm·m²/(g·d)]
    k1: float = 5.0  # moisture half-saturation [mm]
    rw: float = 0.19  # soil moisture loss rate [d⁻¹]

    # Vegetation (Table I / Table II, mm-based)
    c: float = 10.0  # vegetation growth scaling [g/(mm·m²)]
    d: float = 0.13  # mortality rate [d⁻¹]
    Dp: float = 0.0007  # isotropic seed diffusion [m²/d]
    c1: float = 0.005  # flow dispersal coefficient [mm⁻¹]
    c2: float = 0.0005  # flow dispersal saturation [m/d]

    # Sediment transport (Table I + "Model coupling" section)
    gamma: float = 1.0  # transport coefficient [-]
    m_exp: float = 1.65  # discharge exponent [-]
    n_exp: float = 1.65  # slope exponent [-]
    K_max: float = 0.05  # erosion coefficient, bare soil [m⁻¹]
    K_min: float = 5e-5  # erosion coefficient, vegetated [m⁻¹]  (K × 0.001)
    P_min: float = 0.05  # deposition coefficient, bare soil [m⁻¹]
    P_max: float = 50.0  # deposition coefficient, vegetated [m⁻¹]  (P × 1000)
    v_low: float = 5.0  # vegetation threshold for max erosion [g/m²]
    v_high: float = 20.0  # vegetation threshold for min erosion [g/m²]
