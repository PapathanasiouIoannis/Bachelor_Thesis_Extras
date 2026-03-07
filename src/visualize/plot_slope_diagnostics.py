# src/visualize/plot_slope_diagnostics.py
import matplotlib.pyplot as plt
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_slope_evolution(df):
    """
    Generates a 2x2 grid of Slope vs Sound Speed diagnostics.

    This visualizes how the topological stability (Slope dR/dM) correlates
    with the stiffness of the EoS (Speed of Sound) at the canonical mass.
    """
    set_paper_style()
    print("\n--- Generating Slope Evolution Diagnostics (Paper Style) ---")

    # Define Targets: We analyze the slope at 4 distinct mass steps
    targets = [
        {"col": "Slope14", "mass": "1.4"},
        {"col": "Slope16", "mass": "1.6"},
        {"col": "Slope18", "mass": "1.8"},
        {"col": "Slope20", "mass": "2.0"},
    ]

    # Paper-style: 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Optimization: Drop duplicates to plot one point per EoS Curve
    # We are visualizing global EoS properties, not individual samples.
    unique_stars = df.drop_duplicates(subset=["Curve_ID"])

    for i, target in enumerate(targets):
        ax = axes[i]
        col_name = target["col"]

        # Filter NaNs (Stars that collapsed before reaching this mass step)
        plot_data = unique_stars.dropna(subset=[col_name, "CS2_at_14"])

        # 1. Plot Hadronic (H_main)
        h_data = plot_data[plot_data["Label"] == 0]
        ax.scatter(
            h_data["CS2_at_14"],
            h_data[col_name],
            color=COLORS["H_main"],
            s=15,
            alpha=0.5,
            label="Hadronic",
            edgecolors="none",
            rasterized=True,
        )

        # 2. Plot Quark (Q_main)
        q_data = plot_data[plot_data["Label"] == 1]
        ax.scatter(
            q_data["CS2_at_14"],
            q_data[col_name],
            color=COLORS["Q_main"],
            s=15,
            alpha=0.5,
            label="Quark (CFL)",
            edgecolors="none",
            rasterized=True,
        )

        # --- AXIS & GUIDES ---
        # Use centralized limits to ensure consistency with other slope plots
        ax.set_xlim(CONSTANTS["PLOT_CS2_LIM"])
        ax.set_ylim(CONSTANTS["PLOT_SLOPE_LIM"])

        # Zero line: Separates expanding stars (dR/dM > 0) from compressing stars (dR/dM < 0)
        ax.axhline(
            0, color="black", linestyle=":", alpha=0.6, lw=1.5, label="Zero Slope"
        )

        # Physical Limits for Sound Speed
        ax.axvline(
            1.0 / 3.0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            lw=1,
            label=r"$c_s^2=1/3$",
        )
        ax.axvline(
            1.0, color="gray", linestyle="-", alpha=0.3, lw=1, label=r"$c_s^2=1$"
        )

        # Labels and Style
        ax.set_xlabel(r"$c_s^2(r=0)$ at $1.4 M_{\odot}$")
        ax.set_ylabel(r"Slope $dR/dM$")

        # Inner Text Tag (e.g., "M = 1.4 M_sun")
        ax.text(
            0.05,
            0.05,
            r"$M = {mass_label} M_{{\odot}}$",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
                "boxstyle": "round,pad=0.3",
            },
        )

        # Legend only on first plot
        if i == 0:
            ax.legend(loc="upper right", frameon=True, markerscale=2.0)

    plt.tight_layout()
    plt.savefig("plots/fig_slope_evolution_paper_style.pdf")
    plt.close()
    print("[Success] Saved Slope Evolution Plot (Paper Style).")


def plot_slope_vs_radius(df):
    """
    Generates a 2x2 grid of Slope (dR/dM) vs Radius diagnostics at distinct mass steps.

    This visualizes the 'Topological Phase Transition':
    - Hadronic stars typically have negative slopes (Standard Branch).
    - Quark stars (CFL) often exhibit positive slopes (Stable Branch).
    """
    set_paper_style()
    print("\n--- Generating Slope vs Radius Diagnostics ---")

    # Check if the specific diagnostic column exists
    if "Radius_14" not in df.columns:
        print("[Warn] 'Radius_14' column missing. Skipping plot.")
        return

    targets = [
        {"col": "Slope14", "mass": "1.4"},
        {"col": "Slope16", "mass": "1.6"},
        {"col": "Slope18", "mass": "1.8"},
        {"col": "Slope20", "mass": "2.0"},
    ]

    # 2x2 Grid Layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Filter to unique EoS curves to avoid plotting duplicate scalar values
    # (The dataset contains multiple rows per curve, but Slope is a global property)
    unique_stars = df.drop_duplicates(subset=["Curve_ID"])

    for i, target in enumerate(targets):
        ax = axes[i]
        col = target["col"]

        # Filter NaNs (Stars that collapsed before reaching this mass step)
        data = unique_stars.dropna(subset=[col, "Radius_14"])

        # 1. Plot Hadronic Population (Green)
        h_data = data[data["Label"] == 0]
        ax.scatter(
            h_data["Radius_14"],
            h_data[col],
            color=COLORS["H_main"],
            s=15,
            alpha=0.5,
            label="Hadronic",
            edgecolors="none",
            rasterized=True,
        )

        # 2. Plot Quark Population (Magenta)
        q_data = data[data["Label"] == 1]
        ax.scatter(
            q_data["Radius_14"],
            q_data[col],
            color=COLORS["Q_main"],
            s=15,
            alpha=0.5,
            label="Quark (CFL)",
            edgecolors="none",
            rasterized=True,
        )

        # --- Formatting ---
        # Match Axes to global plotting standards
        ax.set_xlim(CONSTANTS["PLOT_R_LIM"])
        ax.set_ylim(CONSTANTS["PLOT_SLOPE_LIM"])

        # Stability Reference Line (Zero Slope)
        # Positive Slope -> Stable Branch (dr/dM > 0)
        # Negative Slope -> Standard Branch (dr/dM < 0)
        ax.axhline(0, color="black", linestyle=":", alpha=0.6, lw=1)

        # Text Tag
        ax.text(
            0.05,
            0.05,
            r"$M = {mass_val} M_{{\odot}}$",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray",
                "boxstyle": "round,pad=0.3",
            },
        )

        ax.set_xlabel(r"Radius $R_{1.4}$ [km]")
        ax.set_ylabel(r"Slope $dR/dM$")

        # Legend (Only on first subplot to reduce clutter)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, markerscale=2.0)

    plt.tight_layout()
    plt.savefig("plots/fig_slope_vs_radius_paper_style.pdf")
    plt.close()
    print("[Success] Saved Slope vs Radius Plot.")
