"""
  Generates a single high-resolution EoS curve and its corresponding P-Epsilon
  grid specifically for theoretical plotting and visualization.

Cleaned & Refactored:
  - PURE ANALYTICAL MATH: Synced with `worker_hadronic_gen.py` to completely
    eliminate splines. The `eos_callable` now returns exact analytical
    derivatives (Chain Rule) for both Hadronic and Quark models.
  - ZERO-WIGGLE PLOTS: Because the math is analytically exact, the plotting
    grids (`eos_arr`) will perfectly match the integration curves without
    any Runge's Phenomenon artifacts.
  - DRY PRINCIPLES: Leverages `CONSTANTS` for all physical parameters.
"""

import numpy as np
from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence


def worker_get_plot_curve(mode, baselines, seed):
    """
    Generates a single high-resolution EoS curve and its corresponding P-Epsilon grid.

    Parameters:
    - mode: 'hadronic' or 'quark'
    - baselines: Dictionary of hadronic max masses (only used if mode='hadronic')
    - seed: Random seed for reproducibility

    Returns:
    - (curve_data, eos_grid_data)
      * curve_data: Numpy array of the TOV sequence
      * eos_grid_data: Numpy array of [EnergyDensity, Pressure]
    """
    np.random.seed(seed)

    # Physical Constants
    hc = CONSTANTS["HC"]
    m_n = CONSTANTS["M_N"]

    if mode == "hadronic":
        # ==========================================
        # HADRONIC LOGIC (Pure Analytical)
        # ==========================================
        core_lib, crust_funcs = get_eos_library()
        model_names = list(baselines.keys())

        transition_map = {
            name: CONSTANTS["P_TRANS_PS"]
            if name == "PS"
            else CONSTANTS["P_TRANS_DEFAULT"]
            for name in model_names
        }

        m_min_target = CONSTANTS["H_M_MIN_TARGET"]
        m_max_target = CONSTANTS["H_M_MAX_TARGET"]
        delta_limit = CONSTANTS["H_DELTA_LIMIT"]

        while True:
            # 1. SMART PARAMETER SELECTION
            nA, nB = np.random.choice(model_names, 2, replace=False)
            w = np.random.uniform(
                CONSTANTS["H_MIX_WEIGHT_LOWER"], CONSTANTS["H_MIX_WEIGHT_UPPER"]
            )

            base_max_m = w * baselines[nA] + (1.0 - w) * baselines[nB]

            req_delta_min = (m_min_target / base_max_m) - 1.0
            req_delta_max = (m_max_target / base_max_m) - 1.0

            effective_min = max(req_delta_min, -delta_limit)
            effective_max = min(req_delta_max, delta_limit)

            if effective_min >= effective_max:
                continue

            delta = np.random.uniform(effective_min, effective_max)

            target_m = base_max_m * (1.0 + delta)
            alpha = (base_max_m / target_m) ** 2

            # Retrieve both Energy Density (0) and Analytical Derivative (1)
            fA_e, fA_de = core_lib[nA]
            fB_e, fB_de = core_lib[nB]

            p_base_mix = w * transition_map[nA] + (1.0 - w) * transition_map[nB]
            p_jitter = np.random.uniform(
                CONSTANTS["H_JITTER_LOWER"], CONSTANTS["H_JITTER_UPPER"]
            )
            p_trans_mix = p_base_mix * p_jitter

            # 2. PURE ANALYTICAL EOS CALLABLE
            def eos_callable(P_query):
                if P_query > p_trans_mix:
                    p_base = P_query / alpha
                    vA = fA_e(p_base)
                    vB = fB_e(p_base)

                    if vA <= 0 or vB <= 0:
                        raise ValueError("Negative energy density")

                    eps = ((vA**w) * (vB ** (1.0 - w))) * alpha

                    dedpA, dedpB = fA_de(p_base), fB_de(p_base)
                    termA = w * dedpA / vA
                    termB = (1.0 - w) * dedpB / vB
                    deps_dp = eps * (termA + termB) / alpha
                else:
                    if P_query > CONSTANTS["P_C1"]:
                        eps = crust_funcs["c1"][0](P_query)
                        deps_dp = crust_funcs["c1"][1](P_query)
                    elif P_query > CONSTANTS["P_C2"]:
                        eps = crust_funcs["c2"][0](P_query)
                        deps_dp = crust_funcs["c2"][1](P_query)
                    elif P_query > CONSTANTS["P_C3"]:
                        eps = crust_funcs["c3"][0](P_query)
                        deps_dp = crust_funcs["c3"][1](P_query)
                    else:
                        eps = crust_funcs["c4"][0](P_query)
                        deps_dp = crust_funcs["c4"][1](P_query)

                if deps_dp < 1.0:
                    deps_dp = 1.0

                cs2 = 1.0 / deps_dp if deps_dp > 0 else 0.0
                return float(eps), float(cs2)

            # 3. SOLVE STRUCTURE
            try:
                curve, max_m = solve_sequence(eos_callable, is_quark=False)
            except Exception:
                continue

            if max_m < m_min_target or max_m > m_max_target:
                continue

            c = np.array(curve)
            if len(c) < 2:
                continue

            violation_mask = (c[:, 5] >= 0.999) & (
                c[:, 4] < CONSTANTS["CAUSALITY_EPS_LIMIT"]
            )
            if np.any(violation_mask):
                continue

            mask_canonical = c[:, 0] > CONSTANTS["M_CANONICAL"]
            if np.any(mask_canonical):
                if np.max(c[mask_canonical, 1]) > 14.0:
                    continue

            if c[0, 0] > CONSTANTS["M_CANONICAL"]:
                continue

            # 4. GENERATE DENSE EOS GRID FOR PLOTTING
            try:
                p_plot_grid = np.logspace(-4.0, np.log10(3000.0), 200)
                eos_arr = []
                for p_val in p_plot_grid:
                    try:
                        eps_val, _ = eos_callable(p_val)
                        if eps_val > 0:
                            eos_arr.append([eps_val, p_val])
                    except Exception:
                        pass

                if len(eos_arr) == 0:
                    continue

                return (c, np.array(eos_arr))
            except Exception:
                continue

    else:
        # ==========================================
        # QUARK LOGIC (Analytic CFL)
        # ==========================================
        while True:
            # 1. PARAMETER SELECTION
            ms_MeV = np.random.uniform(*CONSTANTS["Q_MS_RANGE"])
            Delta_MeV = np.random.uniform(*CONSTANTS["Q_DELTA_RANGE"])

            mu_limit = m_n / 3.0
            term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)
            eff_gap_sq = Delta_MeV**2 - (ms_MeV**2 / 4.0)
            term2 = (3.0 / np.pi**2) * eff_gap_sq * (mu_limit**2)

            B_max = (term1 + term2) / (hc**3)
            B_min = CONSTANTS["Q_B_MIN"]

            if B_max <= B_min:
                continue

            target_m_max = np.random.uniform(1.0, CONSTANTS["Q_M_MAX_UPPER"])
            B_guess = CONSTANTS["Q_B_GUESS_BASE"] * (2.03 / target_m_max) ** 2
            delta_stiffness_factor = 1.0 + (
                CONSTANTS["Q_B_STIFFNESS_FACTOR"] * (Delta_MeV / 500.0)
            )
            B_guess_corrected = B_guess * delta_stiffness_factor
            noise = np.random.uniform(
                CONSTANTS["Q_NOISE_LOWER"], CONSTANTS["Q_NOISE_UPPER"]
            )
            B_target = B_guess_corrected * noise

            real_upper = min(B_max, CONSTANTS["Q_B_ABS_MAX"])
            if B_target < B_min:
                B_target = B_min
            if B_target > real_upper:
                B_target = real_upper

            B = B_target

            # 2. EOS CALLABLE CONSTRUCTION
            Delta_geom = Delta_MeV / hc
            ms_geom = ms_MeV / hc
            B_geom = B / hc

            coeff_a = 3.0 / (4.0 * np.pi**2)
            eff_gap_term = Delta_geom**2 - (ms_geom**2 / 4.0)
            coeff_b = 3.0 * eff_gap_term / (np.pi**2)

            def eos_callable(P_query):
                P_geom = P_query / hc
                coeff_c = -(P_geom + B_geom)

                det = coeff_b**2 - 4 * coeff_a * coeff_c
                if det < 0:
                    return 0.0, 0.0

                mu2 = (-coeff_b + np.sqrt(det)) / (2 * coeff_a)
                eps_geom = 3.0 * coeff_a * (mu2**2) + coeff_b * mu2 + B_geom
                epsilon = eps_geom * hc

                term_shift = 2.0 * eff_gap_term
                numerator = 3.0 * mu2 + term_shift
                denominator = mu2 + term_shift

                if abs(denominator) < 1e-10:
                    cs2 = 1.0 / 3.0
                else:
                    dedp = numerator / denominator
                    cs2 = 1.0 / dedp if dedp > 0 else 0.0

                return float(epsilon), float(cs2)

            eps_surf, _ = eos_callable(0.0)

            # 3. SOLVE STRUCTURE
            try:
                curve, max_m = solve_sequence(
                    eos_callable, is_quark=True, eps_surf=eps_surf
                )
            except Exception:
                continue

            if max_m < 1.0 or max_m > CONSTANTS["Q_M_MAX_UPPER"]:
                continue

            c = np.array(curve)
            if len(c) < 2:
                continue

            if np.max(c[:, 1]) > CONSTANTS["Q_R_MAX"]:
                continue

            if c[0, 0] > 1.0 or c[-1, 0] < 1.0:
                continue

            # 4. GENERATE DENSE EOS GRID FOR PLOTTING
            try:
                p_plot_grid = np.logspace(-4.0, np.log10(3000.0), 200)
                eos_arr = []
                for p_val in p_plot_grid:
                    try:
                        eps_val, _ = eos_callable(p_val)
                        if eps_val > 0:
                            eos_arr.append([eps_val, p_val])
                    except Exception:
                        pass

                if len(eos_arr) == 0:
                    continue

                return (c, np.array(eos_arr))
            except Exception:
                continue
