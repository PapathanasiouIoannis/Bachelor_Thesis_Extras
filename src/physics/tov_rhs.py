# src/physics/tov_rhs.py

"""
  Computes the derivatives for the Tolman-Oppenheimer-Volkoff (TOV) equations
  coupled with the Tidal Deformability Riccati equation.

Refactored:
  - PURE GRAVITY ENGINE: Decoupled entirely from specific EoS physics.
    All Hadronic mixing, Generalized CFL root-finding, and crust transitions
    have been moved out. It now purely evaluates General Relativity.
  - SMOOTH INTEGRATION: By receiving a pre-clamped, C^1 continuous `eos_callable`
    from the workers, we eliminate the C^0 continuous shockwaves (if cs2 > 1: clamp)
    that previously caused Runge-Kutta step-size collapse and stiffness.
"""

from src.const import CONSTANTS


def tov_rhs(r, y_state, eos_callable):
    """
    Computes the derivatives for the TOV and Tidal Deformability equations.

    Parameters:
    - r: Current radius (integration variable) [km]
    - y_state: Array containing[Mass (M_sun), Pressure (MeV/fm^3), y_tidal (dimensionless)]
    - eos_callable: A function/closure that takes Pressure and returns
                    (Energy_Density [MeV/fm^3], Sound_Speed_Squared).

    Returns:
    - [dm_dr, dP_dr, dy_dr]
    """
    m, P, y_tidal = y_state

    # Ensure P is never strictly zero or negative to avoid numerical instability
    P_safe = max(P, CONSTANTS["TOV_P_MIN_SAFE"])

    # ==========================================
    # 1. MICROPHYSICS (Thermodynamics)
    # ==========================================
    # Evaluate the agnostic EoS callable (Spline for Hadronic, Algebraic for Quark)
    epsilon, cs2_local = eos_callable(P_safe)

    # Terminate integration safely if density becomes unphysical or at singularity origin
    if r < 1e-4 or epsilon <= 0:
        return [0.0, 0.0, 0.0]

    # Protect against divisions by zero in the Riccati equation
    if cs2_local < 1e-10:
        cs2_local = 1e-10

    # ==========================================
    # 2. MACROPHYSICS (General Relativity)
    # ==========================================

    # --- A. TOV EQUATIONS (Structure) ---
    term_1 = epsilon + P_safe

    # G_CONV includes 4*pi*G/c^4 factors for converting MeV/fm^3 to geometric terms
    term_2 = m + (r**3 * P_safe * CONSTANTS["G_CONV"])

    # A_CONV converts solar masses to km (G*M_sun/c^2)
    term_3 = r * (r - 2.0 * m * CONSTANTS["A_CONV"])

    # Singularity / Horizon protection
    if abs(term_3) < 1e-5:
        return [0.0, 0.0, 0.0]

    # dP/dr (Hydrostatic equilibrium curve)
    dP_dr = -CONSTANTS["A_CONV"] * (term_1 * term_2) / term_3

    # dm/dr (Mass continuity equation)
    dm_dr = (r**2) * epsilon * CONSTANTS["G_CONV"]

    # --- B. RICCATI EQUATION (Tidal Deformability) ---
    # exp_lambda is the metric component e^lambda (radial stretching)
    exp_lambda = 1.0 / (1.0 - 2.0 * CONSTANTS["A_CONV"] * m / r)

    # Q term (incorporates the speed of sound cs2_local)
    Q = (
        CONSTANTS["A_CONV"]
        * CONSTANTS["G_CONV"]
        * (5.0 * epsilon + 9.0 * P_safe + (epsilon + P_safe) / cs2_local)
        * (r**2)
    )
    Q -= 6.0 * exp_lambda

    # F term
    F = (
        1.0 - CONSTANTS["A_CONV"] * CONSTANTS["G_CONV"] * (r**2) * (epsilon - P_safe)
    ) * exp_lambda

    # dy/dr (Tidal structural evolution)
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r

    return [dm_dr, dP_dr, dy_dr]
