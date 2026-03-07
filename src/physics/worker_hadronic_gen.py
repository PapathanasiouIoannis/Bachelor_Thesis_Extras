# src/physics/worker_hadronic_gen.py

"""
  Generates Hadronic Star EoS using an Anchored Speed-of-Sound generator.

Refactored:
  - REMOVED STRATIFIED BUCKETS: Removed the mathematically impossible constraint
    of forcing uniformly distributed M_max up to 3.6 M_sun while restricting
    R_1.4 to <= 14.5 km.
  - NATURAL SAMPLING: Reverted to natural forward sampling. The generator rolls
    random microphysics, enforces the viability cut (M_max >= 2.08) and the
    observational radius bounds (9.5 <= R_1.4 <= 14.5), and accepts the resulting
    valid stars. This runs extremely fast and reflects true physical probability.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence
from src.physics.feature_extraction import extract_features


def build_anchored_sos_spline(crusts, core_anchor, P_trans):
    """
    Constructs a C^1 continuous EoS by keeping the analytic crust, anchoring the
    low-density core to a dynamically chosen nuclear baseline, and generating a
    smooth random Speed-of-Sound (c_s^2) spline for the deep core.
    """
    fA_e, fA_de = core_anchor

    # ==============================================================
    # 1. THE DYNAMIC NUCLEAR ANCHOR (Low-Density Core)
    # ==============================================================
    # P_ceft randomly smears the transition point to prevent geometric EoS kinks
    P_ceft = np.random.uniform(2.0, 15.0)

    eps_ceft = fA_e(P_ceft)
    dedp_ceft = fA_de(P_ceft)
    cs2_ceft = np.clip(1.0 / dedp_ceft if dedp_ceft > 0 else 1e-5, 1e-5, 1.0)

    p_anchor = np.logspace(np.log10(P_trans), np.log10(P_ceft), 200)
    eps_anchor = np.zeros_like(p_anchor)
    cs2_anchor = np.zeros_like(p_anchor)

    for i, p in enumerate(p_anchor):
        e_val = fA_e(p)
        de_val = fA_de(p)
        eps_anchor[i] = e_val
        cs2_anchor[i] = np.clip(1.0 / de_val if de_val > 0 else 1e-5, 1e-5, 1.0)

    # ==============================================================
    # 2. RANDOMIZED DEEP CORE (High-Density Topology)
    # ==============================================================
    eps_peak = np.random.uniform(eps_ceft + 50.0, 1200.0)
    cs2_peak = np.random.uniform(0.4, 1.0)

    eps_asym = 4000.0
    cs2_asym = np.random.uniform(0.15, 0.333)

    eps_nodes = [eps_ceft, eps_peak, eps_asym]
    cs2_nodes = [cs2_ceft, cs2_peak, cs2_asym]

    cs2_eps_spline = PchipInterpolator(eps_nodes, cs2_nodes)

    eps_deep_full = np.logspace(np.log10(eps_ceft), np.log10(eps_asym), 600)
    cs2_deep_full = cs2_eps_spline(eps_deep_full)
    cs2_deep_full = np.clip(cs2_deep_full, 1e-5, 1.0)

    p_deep_full = P_ceft + cumulative_trapezoid(cs2_deep_full, eps_deep_full, initial=0)

    # Strip the overlap safely to ensure strictly monotonically increasing arrays
    p_deep = p_deep_full[1:]
    eps_deep = eps_deep_full[1:]
    cs2_deep = cs2_deep_full[1:]

    # ==============================================================
    # 3. THE CRUST
    # ==============================================================
    p_crust = np.logspace(
        np.log10(CONSTANTS["GRID_P_MIN_LOG"]),
        np.log10(P_trans),
        CONSTANTS["GRID_CRUST_POINTS"],
        endpoint=False,
    )

    eps_crust = np.zeros_like(p_crust)
    cs2_crust = np.zeros_like(p_crust)

    P_c1 = CONSTANTS["P_C1"]
    P_c2 = CONSTANTS["P_C2"]
    P_c3 = CONSTANTS["P_C3"]

    for i, pc in enumerate(p_crust):
        if pc > P_c1:
            e_init = crusts["c1"][0](pc)
            dedp = crusts["c1"][1](pc)
        elif pc > P_c2:
            e_init = crusts["c2"][0](pc)
            dedp = crusts["c2"][1](pc)
        elif pc > P_c3:
            e_init = crusts["c3"][0](pc)
            dedp = crusts["c3"][1](pc)
        else:
            e_init = crusts["c4"][0](pc)
            dedp = crusts["c4"][1](pc)

        eps_crust[i] = e_init
        cs2_crust[i] = np.clip(1.0 / dedp if dedp > 0 else 1e-5, 1e-5, 1.0)

    p_full = np.concatenate([p_crust, p_anchor, p_deep])
    eps_full = np.concatenate([eps_crust, eps_anchor, eps_deep])
    cs2_full = np.concatenate([cs2_crust, cs2_anchor, cs2_deep])

    return p_full, eps_full, cs2_full


def worker_hadronic_gen(n_curves_to_gen, baselines, seed_offset, batch_idx):
    """
    Worker process for generating unbiased, dynamically anchored Hadronic EoS curves.
    """
    np.random.seed(seed_offset)

    core_lib, crust_funcs = get_eos_library()
    model_names = list(core_lib.keys())

    valid_data = []
    curves_found = 0
    attempts = 0

    m_min_save = CONSTANTS["M_MIN_SAVE"]
    m_max_lower = CONSTANTS["M_MAX_LOWER_BOUND"]
    m_max_upper = CONSTANTS["H_M_MAX_UPPER"]

    max_attempts = n_curves_to_gen * 5000

    while curves_found < n_curves_to_gen and attempts < max_attempts:
        attempts += 1

        # 1. Select dynamic anchor
        anchor_name = np.random.choice(model_names)
        core_anchor = core_lib[anchor_name]

        P_trans = (
            CONSTANTS["P_TRANS_PS"]
            if anchor_name == "PS"
            else CONSTANTS["P_TRANS_DEFAULT"]
        )

        # 2. Build grids
        p_grid, eps_grid, cs2_grid = build_anchored_sos_spline(
            crust_funcs, core_anchor, P_trans
        )

        eps_spline = PchipInterpolator(p_grid, eps_grid)
        cs2_spline = PchipInterpolator(p_grid, cs2_grid)

        def eos_callable(p):
            p_safe = np.clip(p, p_grid[0], p_grid[-1])
            return float(eps_spline(p_safe)), float(cs2_spline(p_safe))

        eos_callable.eps_surf = 0.0

        # 3. Solve structure
        curve, max_m = solve_sequence(eos_callable, is_quark=False)

        # 4. Viability cut (Must support PSR J0740+6620, must obey causality bounds)
        if max_m < m_max_lower or max_m > m_max_upper:
            continue

        c_arr = np.array(curve)
        if len(c_arr) == 0 or c_arr[0, 0] > 0.5:
            continue

        # 5. Extract and validate
        features = extract_features(c_arr, max_m)
        if features is None:
            continue

        # Hard Observational Radius Limits
        if features["r_14"] > 14.5 or features["r_14"] < 9.5:
            continue

        # 6. Save Data
        curves_found += 1
        curve_id = f"H_{batch_idx}_{attempts}"

        for pt in curve:
            m_val = pt[0]
            if m_val >= m_min_save and m_val <= max_m:
                valid_data.append(
                    [
                        m_val,  # Mass
                        pt[1],  # Radius
                        pt[2],  # Lambda
                        0,  # Label (0 = Hadronic)
                        curve_id,  # Group ID
                        pt[3],  # P_Central
                        pt[4],  # Eps_Central
                        0.0,  # Eps_Surface
                        pt[5],  # CS2_Central
                        features["cs2_at_14"],
                        features["r_14"],
                        features["slopes"].get(1.4, np.nan),
                        features["slopes"].get(1.6, np.nan),
                        features["slopes"].get(1.8, np.nan),
                        features["slopes"].get(2.0, np.nan),
                        np.nan,
                        np.nan,
                        np.nan,  # Quark Params
                    ]
                )

    return valid_data
