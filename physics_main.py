# physics_main.py

"""
  Isolated test-bed orchestrator for the Physics Generation and Visualization
  pipeline. This script is used to strictly validate the Equation of State (EoS)
  generation, Tolman-Oppenheimer-Volkoff (TOV) solver stability, and
  thermodynamic constraints.

Refactored:
  - PARQUET UPGRADE: Synced with the main pipeline to use Apache Parquet
    via PyArrow for lightning-fast file I/O and strict type preservation.
"""

import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.const import CONSTANTS
from src.physics.run_worker_wrapper import run_worker_wrapper

from src.visualize.plot_theoretical_eos import plot_theoretical_eos
from src.visualize.plot_grand_summary import plot_grand_summary
from src.visualize.plot_physics_manifold import (
    plot_physics_manifold,
    plot_manifold_curves,
)
from src.visualize.plot_stability_window import plot_stability_window
from src.visualize.plot_surface_density import plot_surface_density

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data"
# UPGRADE: Synced to Parquet format
DATA_FILE = os.path.join(DATA_DIR, "physics_test_dataset.parquet")


def main():
    print("===============================================================")
    print("      PHYSICS ISOLATION ENVIRONMENT: EOS & TOV PIPELINE        ")
    print("===============================================================")

    # 1. Directory Setup
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 2. Physics Initialization
    print("\n[Step 1] Initializing Physics Environment...")
    baselines = None

    # 3. Parallel Generation
    print("\n[Step 2] Generating Physics Test Data (Parallel) ...")

    # Keep totals moderate for rapid physics debugging (e.g., 200 total curves)
    TOTAL_CURVES = 2000
    CURVES_PER_BATCH = 20
    N_JOBS = -1  # Use all available CPU cores

    tasks = []
    num_batches = max(1, TOTAL_CURVES // CURVES_PER_BATCH)

    for i in range(num_batches):
        # Interleave Hadronic and Quark tasks for load balancing across cores
        t_type = "hadronic" if i % 2 == 0 else "quark"
        tasks.append((t_type, CURVES_PER_BATCH, i, i))

    print(f"Spawning {len(tasks)} tasks to generate {TOTAL_CURVES} EoS curves...")

    # Execute Parallel Workers
    res = Parallel(n_jobs=N_JOBS)(
        delayed(run_worker_wrapper)(t, baselines) for t in tqdm(tasks)
    )

    # Flatten results
    valid_rows = [item for sublist in res if sublist is not None for item in sublist]
    cols = CONSTANTS["COLUMN_SCHEMA"]

    # Create DataFrame
    df = pd.DataFrame(valid_rows, columns=cols)

    # 4. Balancing & Saving
    print("\n[Data Integrity] Checking Class Distribution...")
    counts = df["Label"].value_counts()
    print(counts)

    # Shuffle to thoroughly mix classes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # UPGRADE: Save to Parquet
    df.to_parquet(DATA_FILE, engine="pyarrow", index=False)
    print(f"[SUCCESS] Saved physics dataset ({len(df)} samples) to {DATA_FILE}.")

    # 5. Physics Visualization Suite
    print("\n[Step 3] Running Physics Visualization Suite...")

    # Panel 1: The Equation of State (Microphysics)
    plot_theoretical_eos(df)

    # Panel 2: The Stability Window (QCD Vacuum Bounds)
    plot_stability_window(df)

    # Panel 3: Boundary Condition Validation (The Forbidden Gap)
    plot_surface_density(df)

    # Panel 4: The Macroscopic Phase Space (KDE & Raw Curves)
    plot_physics_manifold(df)
    plot_manifold_curves(df)

    # Panel 5: The Grand Summary Triptych
    plot_grand_summary(df)

    print("\n===============================================================")
    print("             PHYSICS PIPELINE COMPLETED SUCCESSFULLY           ")
    print("===============================================================")


if __name__ == "__main__":
    main()
