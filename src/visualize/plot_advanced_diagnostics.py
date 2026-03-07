# src/visualize/plot_advanced_diagnostics.py

"""
  Generates the Universal Relations check and the Misclassification Geography map.

Refactored:
  - 1x2 GRID: Removed the obsolete physics models (B, C, D) to focus purely on
    the observational baseline (Geo) vs. the tidal-informed model (A).
  - AESTHETICS: Upgraded to use the centralized color palette and vector/raster
    hybrid rendering for high-resolution PDF output without massive file sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_universal_relations(df):
    """
    Plots Tidal Deformability (Lambda) vs Compactness (C = GM/Rc^2).
    Checks adherence to the 'I-Love-Q' universal relations.
    """
    set_paper_style()
    print("\n--- Generating Universal Relations Check (I-Love-Q) ---")

    plot_df = df.copy()
    plot_df["Compactness"] = CONSTANTS["A_CONV"] * (plot_df["Mass"] / plot_df["Radius"])

    if "LogLambda" not in plot_df.columns:
        plot_df["LogLambda"] = np.log10(plot_df["Lambda"])

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(plot_df) > 10000:
        plot_df = plot_df.sample(10000, random_state=42)

    h_data = plot_df[plot_df["Label"] == 0]
    ax.scatter(
        h_data["Compactness"],
        h_data["LogLambda"],
        c=COLORS["H_main"],
        s=10,
        alpha=0.3,
        label="Hadronic",
        edgecolors="none",
        rasterized=True,
    )

    q_data = plot_df[plot_df["Label"] == 1]
    ax.scatter(
        q_data["Compactness"],
        q_data["LogLambda"],
        c=COLORS["Q_main"],
        s=10,
        alpha=0.3,
        label="Quark (CFL)",
        edgecolors="none",
        rasterized=True,
    )

    ax.set_xlabel(r"Compactness $C = GM/Rc^2$")
    ax.set_ylabel(r"Log Tidal Deformability $\log_{10}\Lambda$")
    ax.set_title(r"Universal Relations: Compactness vs Deformability")

    # Physical Limits
    ax.axvline(
        CONSTANTS["BH_LIMIT"],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="Black Hole Limit",
    )
    ax.axvline(
        CONSTANTS["BUCHDAHL_LIMIT"],
        color=COLORS["Constraint"],
        linestyle="--",
        linewidth=1.5,
        label="Buchdahl Limit",
    )

    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0, 5)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/fig_universal_relations.pdf")
    plt.close()
    print("[SUCCESS] Saved Universal Relations Plot.")


def plot_misclassification_map(models_dict, X_test_all, y_test):
    """
    Visualizes the 'Geography of Failure' comparing Model Geo and Model A.
    Shows exactly where in the Mass-Radius plane the model gets confused.
    """
    set_paper_style()
    print("\n--- Generating Misclassification Geography Map (Geo vs A) ---")

    model_keys = ["Geo", "A"]
    model_names = {
        "Geo": "Model Geo: Mass & Radius Only",
        "A": "Model A: Mass, Radius & Tidal Deformability",
    }

    # Validate models exist
    for key in model_keys:
        if key not in models_dict:
            print(f"[Warn] {key} missing from models_dict. Skipping misclass map.")
            return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05)

    # Common Physics Data (Buchdahl Limit)
    m_buch = np.linspace(0, 4.5, 100)
    r_buch = CONSTANTS["BUCHDAHL_FACTOR"] * CONSTANTS["A_CONV"] * m_buch

    for i, key in enumerate(model_keys):
        ax = axes[i]
        model = models_dict[key]["classifier"]
        features = CONSTANTS["ML_FEATURES"][key]

        # Predict
        X_slice = X_test_all[features]
        y_pred = model.predict(X_slice)

        # Define Masks
        mask_correct_h = (y_test == 0) & (y_pred == 0)
        mask_correct_q = (y_test == 1) & (y_pred == 1)
        mask_fp = (y_test == 0) & (y_pred == 1)  # False Positive (H predicted as Q)
        mask_fn = (y_test == 1) & (y_pred == 0)  # False Negative (Q predicted as H)

        # 1. Plot Background (Correct predictions, faint)
        ax.scatter(
            X_test_all.loc[mask_correct_h, "Radius"],
            X_test_all.loc[mask_correct_h, "Mass"],
            c=COLORS["H_main"],
            s=10,
            alpha=0.1,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            X_test_all.loc[mask_correct_q, "Radius"],
            X_test_all.loc[mask_correct_q, "Mass"],
            c=COLORS["Q_main"],
            s=10,
            alpha=0.1,
            edgecolors="none",
            rasterized=True,
        )

        # 2. Plot Errors (Foreground, bold)
        ax.scatter(
            X_test_all.loc[mask_fp, "Radius"],
            X_test_all.loc[mask_fp, "Mass"],
            marker="x",
            c=COLORS["FalsePos"],
            s=40,
            lw=1.5,
            label="False Pos (H→Q)",
        )
        ax.scatter(
            X_test_all.loc[mask_fn, "Radius"],
            X_test_all.loc[mask_fn, "Mass"],
            marker="x",
            c=COLORS["FalseNeg"],
            s=40,
            lw=1.5,
            label="False Neg (Q→H)",
        )

        # 3. Physics Overlays
        ax.fill_betweenx(
            m_buch, 0, r_buch, color=COLORS["Guide"], alpha=0.15, zorder=-5
        )
        ax.axhline(2.08, color=COLORS["Constraint"], linestyle="-.", lw=1.5, alpha=0.8)

        # 4. Stats Box
        n_errors = mask_fp.sum() + mask_fn.sum()
        acc = 100 * (1 - n_errors / len(y_test))
        stats = f"Test Accuracy: {acc:.2f}%\nTotal Errors: {n_errors}"
        ax.text(
            0.05,
            0.95,
            stats,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        ax.set_title(model_names[key], y=1.02)
        ax.set_xlim(CONSTANTS["PLOT_R_LIM"])
        ax.set_ylim(max(CONSTANTS["M_MIN_SAVE"], 1.0), CONSTANTS["PLOT_M_LIM"][1])
        ax.set_xlabel(r"Radius $R$ [km]")

    axes[0].set_ylabel(r"Mass $M$ [$M_{\odot}$]")

    # Comprehensive Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["H_main"],
            label="Correct Hadronic",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["Q_main"],
            label="Correct Quark",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["FalsePos"],
            linestyle="None",
            lw=2,
            label="False Positive (H misclassified as Q)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["FalseNeg"],
            linestyle="None",
            lw=2,
            label="False Negative (Q misclassified as H)",
        ),
        Patch(facecolor="gray", alpha=0.2, label="GR Forbidden (Buchdahl)"),
    ]

    # Place legend safely in the top left of the second panel
    axes[1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.85),
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig("plots/fig_ml_misclassification_map.pdf")
    plt.close()
    print("[SUCCESS] Saved 1x2 Misclassification Map.")
