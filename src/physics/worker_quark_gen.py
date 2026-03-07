# src/physics/worker_quark_gen.py

"""
  Generates Quark Star EoS using the Generalized CFL model.

Refactored:
  - FIXED UNITS MISMATCH: `eff_gap_sq` was previously calculated in MeV^2 and
    fed into geometric equations expecting fm^-2, causing values to blow up.
  - FIXED SOUND SPEED DERIVATIVE: The fraction for c_s^2 = (dP/dmu) / (dEps/dmu)
    was accidentally inverted in a previous pass. The correct physical formula
    is c_s^2 = (mu^2 + 2*Delta_eff^2) / (3*mu^2 + 2*Delta_eff^2). This perfectly
    restores the smooth asymptotic conformal limit (1/3) for all Quark stars!
"""

import numpy as np
from src.const import CONSTANTS
from src.physics.solve_sequence import solve_sequence
from src.physics.feature_extraction import extract_features


def worker_quark_gen(n_curves_to_gen, seed_offset, batch_idx):
    """
    Worker process for generating bounded Quark Star EoS curves via natural sampling.
    """
    np.random.seed(seed_offset)
    valid_data = []
    curves_found = 0
    attempts = 0

    hc = CONSTANTS["HC"]
    m_n = CONSTANTS["M_N"]

    m_min_save = CONSTANTS["M_MIN_SAVE"]
    m_max_lower = CONSTANTS["M_MAX_LOWER_BOUND"]
    m_max_upper = CONSTANTS["Q_M_MAX_UPPER"]

    max_attempts = n_curves_to_gen * 10000

    while curves_found < n_curves_to_gen and attempts < max_attempts:
        attempts += 1

        # ==============================================================
        # 1. UNIFORM MICROPHYSICS SAMPLING (MeV Space)
        # ==============================================================
        ms_MeV = np.random.uniform(*CONSTANTS["Q_MS_RANGE"])
        Delta_MeV = np.random.uniform(*CONSTANTS["Q_DELTA_RANGE"])

        # Stability relative to neutron decay (E/A < 939 MeV) -> mu <= 313 MeV
        mu_limit = m_n / 3.0
        eff_gap_sq_MeV = Delta_MeV**2 - (ms_MeV**2 / 4.0)

        term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)
        term2 = (3.0 / np.pi**2) * eff_gap_sq_MeV * (mu_limit**2)

        B_max = (term1 + term2) / (hc**3)
        B_min = CONSTANTS["Q_B_MIN"]

        if B_max <= B_min:
            continue

        # Pure natural forward sampling of the Vacuum Bag Constant
        real_upper = min(B_max, CONSTANTS["Q_B_ABS_MAX"])
        B = np.random.uniform(B_min, real_upper)

        # ==============================================================
        # 2. CONVERT TO GEOMETRIC UNITS (fm space)
        # ==============================================================
        Delta_geom = Delta_MeV / hc
        ms_geom = ms_MeV / hc
        B_geom = B / hc

        # CRITICAL BUG FIX: Compute effective gap squared in geometric units!
        eff_gap_sq_geom = Delta_geom**2 - (ms_geom**2 / 4.0)

        # ==============================================================
        # 3. BUILD EOS CALLABLE
        # ==============================================================
        coeff_a = 3.0 / (4.0 * np.pi**2)
        coeff_b = 3.0 * eff_gap_sq_geom / (np.pi**2)

        def eos_callable(p):
            p_safe = max(p, 0.0)
            p_geom = p_safe / hc
            coeff_c = -(p_geom + B_geom)

            # Solve quadratic for mu^2
            det = coeff_b**2 - 4 * coeff_a * coeff_c
            if det < 0:
                return -1.0, 1e-5  # Sentinel for unphysical root

            mu2 = (-coeff_b + np.sqrt(det)) / (2 * coeff_a)

            # Energy Density
            eps_geom = 3.0 * coeff_a * (mu2**2) + coeff_b * mu2 + B_geom
            eps = eps_geom * hc

            # Speed of Sound Squared
            # cs^2 = (dP/dmu) / (dEps/dmu)
            # cs^2 = (3*mu^2 + 6*Delta_eff^2) / (9*mu^2 + 6*Delta_eff^2)
            # cs^2 = (mu^2 + 2*Delta_eff^2) / (3*mu^2 + 2*Delta_eff^2)
            term_shift = 2.0 * eff_gap_sq_geom
            num = mu2 + term_shift
            den = 3.0 * mu2 + term_shift

            if den > 1e-10:
                cs2 = num / den
            else:
                cs2 = 1.0 / 3.0

            cs2 = np.clip(cs2, 1e-5, 1.0)
            return float(eps), float(cs2)

        # Calculate analytic surface density (P=0)
        coeff_c_surf = -B_geom
        det_surf = coeff_b**2 - 4 * coeff_a * coeff_c_surf
        if det_surf < 0:
            continue

        mu2_surf = (-coeff_b + np.sqrt(det_surf)) / (2 * coeff_a)
        eps_surf_geom = 3.0 * coeff_a * (mu2_surf**2) + coeff_b * mu2_surf + B_geom
        eps_surf = float(eps_surf_geom * hc)

        eos_callable.eps_surf = eps_surf

        # ==============================================================
        # 4. SOLVE STRUCTURE & VIABILITY CUTS
        # ==============================================================
        curve, max_m = solve_sequence(eos_callable, is_quark=True)

        # Reject models that collapse before J0740+6620 (2.08 M_sun)
        if max_m < m_max_lower or max_m > m_max_upper:
            continue

        c_arr = np.array(curve)

        # Ensure sequence properly traced the low mass tail
        if len(c_arr) == 0 or c_arr[0, 0] > 0.5:
            continue

        # ==============================================================
        # 5. FEATURE EXTRACTION & OBSERVATIONAL FILTERS
        # ==============================================================
        features = extract_features(c_arr, max_m)
        if features is None:
            continue

        # Quark stars can be slightly more compact, floor is 8.5 km.
        if features["r_14"] > 14.5 or features["r_14"] < 8.5:
            continue

        # ==============================================================
        # 6. SAVE DATA
        # ==============================================================
        curves_found += 1
        curve_id = f"Q_{batch_idx}_{attempts}"

        for pt in curve:
            m_val = pt[0]
            # Save points down to M_MIN_SAVE for full morphology plots
            if m_val >= m_min_save and m_val <= max_m:
                valid_data.append(
                    [
                        m_val,  # Mass
                        pt[1],  # Radius
                        pt[2],  # Lambda
                        1,  # Label (1 = Quark)
                        curve_id,  # Group ID
                        pt[3],  # P_Central
                        pt[4],  # Eps_Central
                        eps_surf,  # Eps_Surface
                        pt[5],  # CS2_Central
                        features["cs2_at_14"],
                        features["r_14"],
                        features["slopes"].get(1.4, np.nan),
                        features["slopes"].get(1.6, np.nan),
                        features["slopes"].get(1.8, np.nan),
                        features["slopes"].get(2.0, np.nan),
                        B,  # Bag_B
                        Delta_MeV,  # Gap_Delta
                        ms_MeV,  # Mass_Strange
                    ]
                )

    return valid_data
