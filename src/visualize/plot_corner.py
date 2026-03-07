import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_corner(df):
    """
    Generates feature corner plots (PairPlots) for Macroscopic and Microscopic variables.

    Figures:
    - fig_corner_macro.pdf: Observables (Mass, Radius, Tidal).
    - fig_corner_micro.pdf: Physics (Density, Sound Speed, Slope).
    """
    set_paper_style()
    print("\n--- Generating Feature Corner Plots (Macro & Micro) ---")

    # ==========================================
    # 1. SHARED HELPER FUNCTION
    # ==========================================
    def generate_corner(data, vars_list, labels_map, limits_map, title, filename):
        """
        Generic function to create a "Cloud & Core" Corner Plot.
        """
        # Subsample for performance and file size (Vector graphics can get huge)
        if len(data) > 5000:
            df_sample = data.sample(5000, random_state=42).copy()
        else:
            df_sample = data.copy()

        # Rename columns for LaTeX labels
        rename_dict = {k: v for k, v in labels_map.items() if k in df_sample.columns}
        df_sample = df_sample.rename(columns=rename_dict)
        plot_vars = [labels_map[v] for v in vars_list]

        # Setup Grid
        g = sns.PairGrid(
            df_sample,
            vars=plot_vars,
            hue="Label",
            corner=True,
            height=3.5,
            diag_sharey=False,
            palette={0: COLORS["H_main"], 1: COLORS["Q_main"]},
        )

        # --- Custom Plotters ---
        def plot_cloud_core(x, y, **kwargs):
            color = kwargs.get("color")
            # 1. Scatter Cloud (Rasterized)
            plt.scatter(
                x, y, color=color, s=5, alpha=0.1, edgecolors="none", rasterized=True
            )

            # 2. Density Core (Vector Contours)
            try:
                # Draw 50% and 90% confidence contours
                sns.kdeplot(
                    x=x,
                    y=y,
                    color=color,
                    levels=3,
                    thresh=0.2,
                    linewidths=1.5,
                    ax=plt.gca(),
                )
            except Exception:
                pass

        def plot_diag(x, **kwargs):
            color = kwargs.get("color")
            # Heuristic based on color

            # Hadronic (0): Solid Fill
            # Note: We can't easily access 'Label' directly inside map_diag grouping
            # without passing it explicitly, but hue mapping handles the color.
            # We assume standard color mapping holds.

            is_hadronic = color == COLORS["H_main"]

            if is_hadronic:
                sns.kdeplot(
                    x=x, color=color, fill=True, alpha=0.3, linewidth=2, ax=plt.gca()
                )
            else:
                # Quark (1): Hatched (No Fill)
                ax = plt.gca()
                sns.kdeplot(
                    x=x, color=color, fill=False, linewidth=2, linestyle="--", ax=ax
                )
                try:
                    # Manually add hatch pattern under the curve
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(x)
                    x_grid = np.linspace(x.min(), x.max(), 100)
                    y_grid = kde(x_grid)
                    ax.fill_between(
                        x_grid,
                        0,
                        y_grid,
                        facecolor="none",
                        edgecolor=color,
                        hatch="////",
                        alpha=0.5,
                    )
                except Exception:
                    pass

        # Map Plots
        g.map_lower(plot_cloud_core)
        g.map_diag(plot_diag)

        # Apply Limits
        for i in range(len(plot_vars)):
            for j in range(len(plot_vars)):
                ax = g.axes[i, j]
                if ax is None:
                    continue

                # X-Axis (Column variable)
                x_var_name = plot_vars[j]
                if x_var_name in limits_map:
                    ax.set_xlim(limits_map[x_var_name])

                # Y-Axis (Row variable) - Only for off-diagonals
                if i != j:
                    y_var_name = plot_vars[i]
                    if y_var_name in limits_map:
                        ax.set_ylim(limits_map[y_var_name])

                # Clean Diagonal Labels (Self vs Self is density, no Y label needed)
                if i == j:
                    ax.set_ylabel("")

        # Legend
        legend_elements = [
            Line2D([0], [0], color=COLORS["H_main"], lw=2, label="Hadronic (Solid)"),
            Line2D(
                [0],
                [0],
                color=COLORS["Q_main"],
                lw=2,
                linestyle="--",
                label="Quark (Hatched)",
            ),
            Patch(facecolor="gray", alpha=0.1, label="Scatter Cloud"),
            Line2D([0], [0], color="gray", lw=1.5, label="Density Core"),
        ]
        g.fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.95, 0.95),
            fontsize=12,
            frameon=True,
            title=r"Stellar Family",
        )

        g.fig.suptitle(title, y=1.02, fontsize=18)
        plt.savefig(f"plots/{filename}")
        plt.close()
        print(f"[Success] Saved {filename}")

    # ==========================================
    # 2. MACRO PLOT (Observables)
    # ==========================================
    df_macro = df.copy()
    if "LogLambda" not in df_macro.columns:
        df_macro["LogLambda"] = np.log10(df_macro["Lambda"])

    vars_macro = ["Radius", "Mass", "LogLambda"]

    labels_macro = {
        "Radius": r"Radius $R$ [km]",
        "Mass": r"Mass $M$ [$M_{\odot}$]",
        "LogLambda": r"Log Tidal $\log_{10}\Lambda$",
    }

    limits_macro = {
        r"Radius $R$ [km]": CONSTANTS["PLOT_R_LIM"],
        r"Mass $M$ [$M_{\odot}$]": CONSTANTS["PLOT_M_LIM"],
        r"Log Tidal $\log_{10}\Lambda$": CONSTANTS["PLOT_L_LIM"],
    }

    generate_corner(
        df_macro,
        vars_macro,
        labels_macro,
        limits_macro,
        r"Observational Phase Space (Macro)",
        "fig_corner_macro.pdf",
    )

    # ==========================================
    # 3. MICRO PLOT (Internal Physics)
    # ==========================================
    # Filter for valid physics data
    df_micro = df.dropna(subset=["Slope14", "Eps_Central", "CS2_Central"]).copy()

    vars_micro = ["Eps_Central", "CS2_Central", "Slope14"]

    labels_micro = {
        "Eps_Central": r"Central Density $\varepsilon_c$ [MeV/fm$^3$]",
        "CS2_Central": r"Sound Speed $c_s^2$",
        "Slope14": r"Slope $dR/dM|_{1.4}$",
    }

    limits_micro = {
        r"Central Density $\varepsilon_c$ [MeV/fm$^3$]": CONSTANTS["PLOT_EPS_LIM"],
        r"Sound Speed $c_s^2$": CONSTANTS["PLOT_CS2_LIM"],
        r"Slope $dR/dM|_{1.4}$": CONSTANTS["PLOT_SLOPE_LIM"],
    }

    generate_corner(
        df_micro,
        vars_micro,
        labels_micro,
        limits_micro,
        r"Internal Physics Phase Space (Micro)",
        "fig_corner_micro.pdf",
    )
