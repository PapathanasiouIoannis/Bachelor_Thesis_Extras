# src/physics/solve_sequence.py

r"""
  Integrates the TOV equations over a range of central pressures to generate
  a full Mass-Radius-Lambda sequence (an EoS curve).

Refactored:
  - MORPHOLOGY FIX: Expanded the central pressure search grid from 1e-3 down
    to 1e-6 MeV/fm^3. This allows the solver to find the extremely low-mass
    ($\sim 0.1 M_\odot$) stars with highly expanded crusts, completely
    eliminating the "floating cut-off" visual artifact.
  - INCREASED RESOLUTION: Bumped n_points to 150 for smoother rendering.
  - SENTINEL CHECKS: Safely skips unphysical branches returned by the EoS callable.
"""

import numpy as np
from scipy.integrate import solve_ivp
from src.const import CONSTANTS
from src.physics.tov_rhs import tov_rhs


def solve_sequence(eos_callable, is_quark=False):
    """
    Integrates TOV for a sequence of central pressures to form a full star curve.

    Parameters:
    - eos_callable: A function/closure that takes Pressure [MeV/fm^3] and
                    returns (Energy_Density [MeV/fm^3], Sound_Speed_Squared).
    - is_quark: Boolean flag used strictly to optimize the pressure search grid.

    Returns:
    - curve_data: List of[Mass, Radius, Lambda, Pc, Eps_c, CS2_c, Eps_surf]
    - max_m: The maximum mass found in this sequence.
    """
    r_min = CONSTANTS["TOV_R_MIN"]

    # ---------------------------------------------------------
    # PRESSURE GRID (The Morphology Fix)
    # ---------------------------------------------------------
    n_points = 150
    if is_quark:
        # Quark stars are self-bound and have higher central pressures even at low masses
        pressures = np.logspace(-1.0, 4.2, n_points)
    else:
        # Hadronic stars need extremely low central pressures to trace
        # the expanded crusts of 0.1 M_sun stars.
        pressures = np.logspace(-6.0, 4.2, n_points)

    curve_data = []
    max_m = 0.0

    # Extract surface density (Default to 0.0 for Hadronic models)
    eps_surf = getattr(eos_callable, "eps_surf", 0.0)

    for pc in pressures:
        # ==============================================================
        # 1. INITIALIZATION (Get Core Microphysics)
        # ==============================================================
        eps_init, cs2_init = eos_callable(pc)

        # Sentinel value check for unphysical roots/branches
        if eps_init < 0:
            continue

        # ==============================================================
        # 2. INTEGRATION (TOV Solver)
        # ==============================================================
        # Initial Mass (Approximation for small r_min)
        m_init = (r_min**3) * eps_init * (CONSTANTS["G_CONV"] / 3.0)

        # State Vector:[Mass, Pressure, y_tidal]
        y0 = [m_init, pc, 2.0]

        # Event to detect surface (Pressure -> 0)
        def surface_event(t, y):
            return y[1]

        surface_event.terminal = True
        surface_event.direction = -1

        try:
            # Integrate from r_min out to the boundary where P = 0
            # Note: TOV_R_MAX was increased to 50.0 in const.py so thick crusts aren't truncated
            sol = solve_ivp(
                fun=lambda r, y: tov_rhs(r, y, eos_callable),
                t_span=(r_min, CONSTANTS["TOV_R_MAX"]),
                y0=y0,
                events=surface_event,
                method="RK45",
                rtol=1e-10,  # Slightly loosened for speed given the dense grid
                atol=1e-12,
            )

            # Check if the integration successfully hit the surface
            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]
                M = sol.y_events[0][0][0]
                yR = sol.y_events[0][0][2]

                # Filter unphysical results
                if R < 3.0 or M < 0.05:
                    continue

                # ==============================================================
                # 3. MACROPHYSICS (Tidal Deformability)
                # ==============================================================
                # Calculate Compactness
                C = (M * CONSTANTS["A_CONV"]) / R

                # STRICT BUCHDAHL LIMIT (C < 4/9)
                if C >= CONSTANTS["BUCHDAHL_LIMIT"]:
                    continue

                # Complex Tidal Love Number (k2) formula (Hinderer et al. 2008)
                num = (
                    (8.0 / 5.0)
                    * (1.0 - 2.0 * C) ** 2
                    * C**5
                    * (2.0 * C * (yR - 1.0) - yR + 2.0)
                )

                den_term1 = 2.0 * C * (6.0 - 3.0 * yR + 3.0 * C * (5.0 * yR - 8.0))
                den_term2 = (
                    4.0
                    * (C**3)
                    * (
                        13.0
                        - 11.0 * yR
                        + C * (3.0 * yR - 2.0)
                        + 2.0 * (C**2) * (1.0 + yR)
                    )
                )
                den_term3 = (
                    3.0
                    * (1.0 - 2.0 * C) ** 2
                    * (2.0 - yR + 2.0 * C * (yR - 1.0))
                    * np.log(1.0 - 2.0 * C)
                )

                den = den_term1 + den_term2 + den_term3

                if abs(den) < 1e-10:
                    continue

                k2 = num / den

                # Dimensionless Tidal Deformability
                Lam = (2.0 / 3.0) * k2 * (C**-5)

                if M <= 0.0:
                    break

                # Peak detection: Stop tracking if mass starts decreasing (Unstable branch)
                if M < max_m:
                    break
                if M > max_m:
                    max_m = M

                # Record stable point
                curve_data.append([M, R, Lam, pc, eps_init, cs2_init, eps_surf])

        except Exception:
            # Trap integration faults
            continue

    return curve_data, max_m
