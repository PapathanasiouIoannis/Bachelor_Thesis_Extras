# main.py

"""
  Master Pipeline Orchestrator for Neutron Star EoS Inference.

Refactored:
  - PARQUET INFRASTRUCTURE: Upgraded data I/O from CSV to Apache Parquet using
    PyArrow. This reduces file size by ~80%, preserves strict data types, and
    loads massive datasets (10,000+ curves) almost instantly into RAM.
  - PRODUCTION SCALING: Increased default TOTAL_CURVES to 10,000 to fully map
    the high-resolution topological phase space for the XGBoost classifier.
"""

import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# --- CONFIGURATION & CONSTANTS ---
from src.const import CONSTANTS

# --- PHYSICS ENGINE ---
from src.physics.run_worker_wrapper import run_worker_wrapper

# --- MACHINE LEARNING PIPELINE ---
from src.ml_pipeline.train_model import train_model
from src.ml_pipeline.analyze_candidates import analyze_candidates
from src.ml_pipeline.advanced_analysis import run_advanced_analysis
from src.ml_pipeline.audit_performance import run_performance_audit

# --- VISUALIZATION SUITE ---
from src.visualize.plot_theoretical_eos import plot_theoretical_eos
from src.visualize.plot_grand_summary import plot_grand_summary
from src.visualize.plot_physics_manifold import (
    plot_physics_manifold,
    plot_manifold_curves,
)
from src.visualize.plot_diagnostics import plot_diagnostics
from src.visualize.plot_3d_separation import plot_3d_separation
from src.visualize.plot_3d_interactive_topology import plot_3d_topology
from src.visualize.plot_physical_insights import plot_physical_insights
from src.visualize.plot_corner import plot_corner
from src.visualize.plot_slope_diagnostics import (
    plot_slope_evolution,
    plot_slope_vs_radius,
)
from src.visualize.plot_stability_window import plot_stability_window
from src.visualize.plot_advanced_diagnostics import plot_misclassification_map
from src.visualize.plot_pdp import plot_partial_dependence
from src.visualize.plot_surface_density import plot_surface_density
from src.visualize.plot_statistical_bands import plot_statistical_bands
from src.visualize.plot_microphysics_3d import plot_microphysics_3d
from src.visualize.plot_correlations import plot_correlations
from src.visualize.plot_topology import (
    plot_geo_M_vs_R,
    plot_A_M_vs_R,
    plot_A_Lambda_vs_M,
    plot_A_Lambda_vs_R,
)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data"
# UPGRADE: Using Parquet for high-performance ML I/O
# DATA_FILE = os.path.join(DATA_DIR, "thesis_dataset.parquet")
DATA_FILE = os.path.join(DATA_DIR, "physics_test_dataset.parquet")
MODEL_FILE = os.path.join("models", "ml_models_geo_A.pkl")


def main():
    print("===============================================================")
    print("   THESIS PIPELINE: NEUTRON STAR EOS INFERENCE ORCHESTRATOR    ")
    print("===============================================================")

    # 1. Directory Setup
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    should_generate = True
    cols = CONSTANTS["COLUMN_SCHEMA"]

    # ==============================================================
    # PHASE 1: DATA MANAGEMENT & PHYSICS GENERATION
    # ==============================================================
    if os.path.exists(DATA_FILE):
        print(f"\n[INFO] Found existing Parquet dataset: {DATA_FILE}")
        try:
            # UPGRADE: Blazing fast load times with PyArrow
            df = pd.read_parquet(DATA_FILE, engine="pyarrow")
            if all(c in df.columns for c in cols):
                print(f"[INFO] Schema validation passed. Loaded {len(df)} samples.")
                should_generate = False
            else:
                print("[WARN] Dataset schema mismatch. Regenerating...")
                should_generate = True
        except Exception as e:
            print(f"[ERROR] Could not read Parquet dataset: {e}")
            should_generate = True

    if should_generate:
        print("\n--- Phase 1: Generating Physics Data (Parallel) ---")

        # PRODUCTION SCALING: Increased to 10,000 to maximize ML accuracy.
        # (Change to 2000 if you want a faster test run).
        TOTAL_CURVES = 1000
        CURVES_PER_BATCH = 20
        N_JOBS = -1

        tasks = []
        num_batches = max(1, TOTAL_CURVES // CURVES_PER_BATCH)

        for i in range(num_batches):
            t_type = "hadronic" if i % 2 == 0 else "quark"
            tasks.append((t_type, CURVES_PER_BATCH, i, i))

        print(f"Spawning {len(tasks)} tasks to generate {TOTAL_CURVES} EoS curves...")

        res = Parallel(n_jobs=N_JOBS)(
            delayed(run_worker_wrapper)(t, None) for t in tqdm(tasks)
        )

        valid_rows = [
            item for sublist in res if sublist is not None for item in sublist
        ]
        df = pd.DataFrame(valid_rows, columns=cols)

        print("\n[Data Integrity] Checking Class Distribution...")
        counts = df["Label"].value_counts()
        print(counts)

        # Shuffle to thoroughly mix classes
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # UPGRADE: Save to Parquet format
        df.to_parquet(DATA_FILE, engine="pyarrow", index=False)
        print(f"[SUCCESS] Saved massive dataset ({len(df)} samples) to {DATA_FILE}.")

    # Reload df to ensure data types are perfectly aligned
    df = pd.read_parquet(DATA_FILE, engine="pyarrow")

    # ==============================================================
    # PHASE 2: MACHINE LEARNING (XGBoost + Optuna)
    # ==============================================================
    models_dict, X_test, y_test = train_model(df)

    # ==============================================================
    # PHASE 3: VISUALIZATION & INFERENCE SUITE
    # ==============================================================
    print("\n--- Phase 3: Running Visualization Suite ---")

    # ML Diagnostics
    plot_diagnostics(models_dict, X_test, y_test)
    run_advanced_analysis(df, models_dict, X_test, y_test)
    run_performance_audit(models_dict, X_test, y_test)

    # Physics Manifold (Data Driven)
    plot_grand_summary(df)
    plot_statistical_bands(df)
    plot_physics_manifold(df)
    plot_manifold_curves(df)
    plot_theoretical_eos(df)

    # Feature Interpretation
    plot_3d_separation(df)
    plot_microphysics_3d(df)
    plot_corner(df)
    plot_3d_topology()

    # Microphysics Verification
    plot_physical_insights(models_dict, df)
    plot_correlations(df)

    # Thesis-Specific Diagnostics
    plot_slope_evolution(df)
    plot_slope_vs_radius(df)
    plot_stability_window(df)

    # Final Physics Checks
    plot_surface_density(df)

    # Inference Phase (Real Astrophysical Stars)
    analyze_candidates(models_dict)

    # Model Interpretability
    plot_misclassification_map(models_dict, X_test, y_test)
    plot_partial_dependence(models_dict, X_test)

    # The Safe Classification Topology Suite
    plot_geo_M_vs_R(models_dict)
    plot_A_M_vs_R(models_dict, df)
    plot_A_Lambda_vs_M(models_dict, df)
    plot_A_Lambda_vs_R(models_dict, df)
    print("\n===============================================================")
    print("             PIPELINE COMPLETED SUCCESSFULLY                   ")
    print("===============================================================")


if __name__ == "__main__":
    main()
