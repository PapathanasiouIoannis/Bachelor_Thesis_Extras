# src/const.py

"""
  Centralized configuration dictionary for Physics limits, Macroscopic constraints,
  Machine Learning features, and Plotting boundaries.

Refactored:
  - ASTROPHYSICAL VIABILITY CUTS: Introduced `M_MAX_LOWER_BOUND` = 2.08 to force
    all generated EoS models to support PSR J0740+6620. Models failing this are
    falsified by reality and discarded.
  - NATURAL UPPER BOUNDS: Separated the upper limits. Hadronic is capped at 3.6
    (Rhoades-Ruffini causality limit), Quark is capped at 4.0 (Natural QCD limit).
"""

CONSTANTS = {
    # ==========================================
    # 1. FUNDAMENTAL PHYSICS & UNITS
    # ==========================================
    "HC": 197.33,  # Planck constant times speed of light [MeV fm]
    "M_N": 939.0,  # Mass of a neutron [MeV]
    "G_CONV": 1.124e-5,  # Geometric conversion factor (G/c^4 equivalent)
    "A_CONV": 1.4766,  # Geometric conversion factor (G M_sun / c^2) [km/M_sun]
    # ==========================================
    # 2. CRUST PHYSICS (SLy Model Parameterization)
    # ==========================================
    "P_C1": 9.34375e-5,
    "P_C2": 4.1725e-8,
    "P_C3": 1.44875e-11,
    "P_TRANS_DEFAULT": 0.184,  # Core-crust transition pressure [MeV/fm^3]
    "P_TRANS_PS": 0.696,  # Specific transition pressure for the PS model
    # ==========================================
    # 3. NUMERICAL SOLVER SETTINGS (TOV & SPLINE GRIDS)
    # ==========================================
    "TOV_R_MIN": 1e-4,
    "TOV_R_MAX": 50.0,  # Safe padding for thick low-mass crusts
    "TOV_P_MIN_SAFE": 1e-12,
    # Compound Pressure Grid Settings
    "GRID_P_MIN_LOG": 1e-12,
    "GRID_P_TRANSITION": 1.0,
    "GRID_P_MAX_LIN": 4000.0,
    "GRID_CRUST_POINTS": 300,
    "GRID_CORE_POINTS": 1200,
    "SMALL_STEP_M": 1e-5,
    # ==========================================
    # 4. QUARK MODEL PARAMETERS (CFL)
    # ==========================================
    "Q_B_MIN": 57.0,
    "Q_B_ABS_MAX": 400.0,
    "Q_DELTA_RANGE": (57.0, 250.0),
    "Q_MS_RANGE": (80.0, 120.0),
    # ==========================================
    # 5. GENERATION CONSTRAINTS & PHYSICAL LIMITS
    # ==========================================
    "M_MIN_SAVE": 0.1,  # Include all theoretically valid continuous low-mass regions.
    # The Astrophysical Viability Cuts (The PSR J0740+6620 limit)
    "M_MAX_LOWER_BOUND": 2.08,  # ALL generated models must reach at least this mass
    # Natural maximum mass limits allowed by GR and Thermodynamics
    "H_M_MAX_UPPER": 3.6,  # Hadronic Causality ceiling
    "Q_M_MAX_UPPER": 4.0,  # Quark (CFL) ceiling
    "Q_R_MAX": 30.0,  # Rejection bound for pathological wide splines
    # Causality & GR Stability
    "CAUSALITY_EPS_LIMIT": 600.0,
    "BUCHDAHL_LIMIT": 4.0 / 9.0,
    "BH_LIMIT": 0.5,
    # ==========================================
    # 6. PLOTTING STANDARDS & VISUALIZATION LIMITS
    # ==========================================
    "PLOT_R_LIM": (5.0, 20.0),
    "PLOT_M_LIM": (0.0, 4.0),
    "PLOT_L_LIM": (0.0, 5.0),
    "PLOT_EPS_LIM": (0, 2500),
    "PLOT_EPS_LOG": (1e2, 5e3),
    "PLOT_CS2_LIM": (0, 1.05),
    "PLOT_SLOPE_LIM": (-8, 6),
    "BUCHDAHL_FACTOR": 2.25,
    # ==========================================
    # 7. DATA SCHEMA & ML ARCHITECTURE
    # ==========================================
    "COLUMN_SCHEMA": [
        "Mass",
        "Radius",
        "Lambda",
        "Label",
        "Curve_ID",
        "P_Central",
        "Eps_Central",
        "Eps_Surface",
        "CS2_Central",
        "CS2_at_14",
        "Radius_14",
        "Slope14",
        "Slope16",
        "Slope18",
        "Slope20",
        "Bag_B",
        "Gap_Delta",
        "Mass_Strange",
    ],
    "ML_FEATURES": {
        "Geo": ["Mass", "Radius"],
        "A": ["Mass", "Radius", "LogLambda"],
        "B": ["Mass", "Radius", "LogLambda", "Eps_Central"],
        "C": ["Mass", "Radius", "LogLambda", "Eps_Central", "CS2_Central"],
        "D": ["Mass", "Radius", "LogLambda", "Eps_Central", "CS2_Central", "Slope14"],
    },
}
