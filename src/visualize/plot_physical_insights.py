import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS


def plot_physical_insights(models_dict, df):
    """
    Generates Feature Importance and Speed of Sound diagnostic plots.

    Figures:
    - Fig 13: Feature Importance (Comparing Observables vs Topology).
    - Fig 15: Speed of Sound Trajectories vs Stellar Cores.

    Parameters:
    - models_dict: Dictionary of trained models (Requires Model 'D').
    - df: The master dataframe containing 'Eps_Central' and 'CS2_Central'.
    """
    set_paper_style()
    print("\n--- Generating Physical Insight Plots ---")

    # ==============================================================
    # FIGURE 13: FEATURE IMPORTANCE (Model D)
    # ==============================================================
    if "D" in models_dict:
        model_D = models_dict["D"]
        features = [
            "Mass",
            "Radius",
            "LogLambda",
            "Eps_Central",
            "CS2_Central",
            "Slope14",
        ]

        # Extract importances (patched in train_model.py)
        if hasattr(model_D, "feature_importances_"):
            importances = model_D.feature_importances_
            # Use standard deviation across folds if available
            std = getattr(model_D, "feature_importances_std_", np.zeros(len(features)))
        else:
            importances = np.zeros(len(features))
            std = np.zeros(len(features))
            print("[Warn] Feature importances not found on Model D.")

        # Split into Macro (Obs) vs Micro (Physics) groups for coloring
        idx_obs = [0, 1, 2]  # Mass, Radius, Lambda
        idx_mic = [3, 4, 5]  # Eps, CS2, Slope

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Observables (Blue-ish)
        y_pos_obs = np.arange(3)
        ax.barh(
            y_pos_obs,
            importances[idx_obs],
            xerr=std[idx_obs],
            color="#1f77b4",
            align="center",
            label="Macroscopic (Observables)",
        )

        # Plot Microphysics (Red-ish)
        y_pos_mic = np.arange(3) + 4  # Offset to separate groups
        ax.barh(
            y_pos_mic,
            importances[idx_mic],
            xerr=std[idx_mic],
            color="#d62728",
            align="center",
            label="Microscopic (Topology)",
        )

        # Formatting
        ax.set_yticks(np.concatenate((y_pos_obs, y_pos_mic)))
        ax.set_yticklabels([features[i] for i in idx_obs + idx_mic])
        ax.set_xlabel("Gini Importance Score")
        ax.set_title(r"Feature Importance: Observables vs. Topology (Model D)")
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig("plots/fig_13_feature_importance.pdf")
        plt.close()
    else:
        print("[Info] Model D not found. Skipping Feature Importance plot.")

    # ==============================================================
    # FIGURE 15: SPEED OF SOUND (Lines + Scatter)
    # ==============================================================
    # This plot shows the "Roads" (EoS trajectories) and the "Cities" (Stellar Cores)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Plot the "Roads" (EoS Trajectories)
    grouped = df.groupby("Curve_ID")
    curve_ids = list(grouped.groups.keys())

    # Subsample lines to avoid visual clutter
    np.random.seed(42)
    limit_lines = 500
    if len(curve_ids) > limit_lines:
        selected_ids = set(np.random.choice(curve_ids, limit_lines, replace=False))
    else:
        selected_ids = set(curve_ids)

    print(f"  > Drawing {len(selected_ids)} background EoS lines...")

    for name, group in grouped:
        if name not in selected_ids:
            continue
        label = group["Label"].iloc[0]
        color = COLORS["Q_main"] if label == 1 else COLORS["H_main"]

        # Sort by density to draw the line correctly
        g = group.sort_values(by="Eps_Central")

        # Filter purely for plotting range
        mask = g["Eps_Central"] < CONSTANTS["PLOT_EPS_LIM"][1]
        if np.any(mask):
            ax.plot(
                g.loc[mask, "Eps_Central"],
                g.loc[mask, "CS2_Central"],
                color=color,
                alpha=0.1,
                lw=0.5,
                zorder=1,
            )

    # 2. Plot the "Cities" (Actual Stellar Cores)
    # We sample actual cores to show where the population lives along the roads
    sample_dots = df.sample(min(2000, len(df)), random_state=42)

    # Hadronic Cores (Green)
    h_dots = sample_dots[sample_dots["Label"] == 0]
    ax.scatter(
        h_dots["Eps_Central"],
        h_dots["CS2_Central"],
        color=COLORS["H_main"],
        s=10,
        alpha=0.6,
        edgecolors="none",
        label="Hadronic Cores",
        zorder=2,
    )

    # Quark Cores (Magenta)
    q_dots = sample_dots[sample_dots["Label"] == 1]
    ax.scatter(
        q_dots["Eps_Central"],
        q_dots["CS2_Central"],
        color=COLORS["Q_main"],
        s=10,
        alpha=0.6,
        edgecolors="none",
        label="Quark Cores",
        zorder=2,
    )

    # 3. Physics Constraints
    ax.set_xlim(CONSTANTS["PLOT_EPS_LIM"])
    ax.set_ylim(CONSTANTS["PLOT_CS2_LIM"])

    # Causal Limit
    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="Causal Limit ($c_s=1$)")
    # Conformal Limit
    ax.axhline(
        1.0 / 3.0,
        color="gray",
        ls=":",
        alpha=0.5,
        label="Conformal Limit ($c_s^2=1/3$)",
    )

    ax.set_xlabel(r"Central Energy Density $\varepsilon_c$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Central Speed of Sound Squared $c_s^2$")
    ax.set_title(r"Microphysics: EoS Trajectories vs. Stellar Cores")

    # Custom Legend
    lines = [
        Line2D([0], [0], color=COLORS["H_main"], lw=2, label="Hadronic Models"),
        Line2D([0], [0], color=COLORS["Q_main"], lw=2, label="Quark Models"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["H_main"],
            label="Hadronic Cores",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["Q_main"],
            label="Quark Cores",
        ),
    ]
    ax.legend(handles=lines, loc="center right", frameon=True)

    plt.tight_layout()
    plt.savefig("plots/fig_15_speed_of_sound.pdf")
    plt.close()

    print("[Success] Figures 13 & 15 Saved.")
