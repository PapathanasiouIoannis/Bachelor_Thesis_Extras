# src/physics/calculate_baselines.py

"""
  Calculates the maximum mass for each unperturbed (pure) hadronic model
  in the library.

Refactored:
  - DRY PRINCIPLES: Now directly imports `build_hadronic_spline` from the
    hadronic generation worker. This guarantees that the parent max mass is
    computed using the EXACT same numerical discretization and causality clamps
    as the mixed child models, preventing scaling mismatches.
  - DECOUPLED: Passes the agnostic `eos_callable` to `solve_sequence`.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from src.const import CONSTANTS
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence
from src.physics.worker_hadronic_gen import build_hadronic_spline


def calculate_baselines():
    """
    Calculates the maximum mass for each pure hadronic model.

    Returns:
    - baselines: Dictionary mapping {ModelName: MaxMass_Msun}
    """
    print("--- Phase 0: Calculating Hadronic Baselines ---")

    # Retrieve the fast-cached library of analytic core models and crust functions
    core_lib, crust_funcs = get_eos_library()
    baselines = {}

    # Iterate through each 'parent' Hadronic EoS model
    for name in core_lib.keys():
        fA_e, fA_de = core_lib[name]

        # Determine the specific Crust-Core transition pressure.
        p_trans = (
            CONSTANTS["P_TRANS_PS"] if name == "PS" else CONSTANTS["P_TRANS_DEFAULT"]
        )

        # 1. Build the Compound C^1 Continuous Spline
        # We simulate a "pure" model by passing the same EoS twice with mixing weight 1.0
        # and scaling factor 1.0. This guarantees identical numerical treatment.
        p_grid, eps_grid, cs2_grid = build_hadronic_spline(
            fA_e, fA_de, fA_e, fA_de, 1.0, crust_funcs, 1.0, p_trans
        )

        if p_grid is None:
            print(f"  > {name:<10}: [FAILED] Unphysical Branch")
            continue

        # 2. Compile Splines
        eps_spline = PchipInterpolator(p_grid, eps_grid)
        cs2_spline = PchipInterpolator(p_grid, cs2_grid)

        # 3. Create Agnostic Callable for TOV Solver
        def eos_callable(p):
            p_safe = np.clip(p, p_grid[0], p_grid[-1])
            return float(eps_spline(p_safe)), float(cs2_spline(p_safe))

        # 4. Solve the TOV sequence to find the maximum stable mass
        _, max_m = solve_sequence(eos_callable, is_quark=False)

        # Store valid results
        if max_m > 1.0:
            baselines[name] = max_m
            print(f"  > {name:<10}: {max_m:.3f} M_sun")

    return baselines
