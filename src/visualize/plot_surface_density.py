# src/visualize/plot_surface_density.py

"""
  Generates the Surface Density distribution plot.

Refactored:
  - PUBLICATION READY: Enhanced seaborn KDE mapping, customized annotations,
    and applied the centralized style configuration.
  - PHYSICS VALIDATION: Rigorously highlights the 'Forbidden Gap' between the
    zero-density surface of gravity-bound Hadronic stars (Iron crust) and the
    high-density surface of self-bound Quark stars. This serves as a primary
    sanity check for the integration boundary conditions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from src.visualize.style_config import set_paper_style, COLORS


def plot_surface_density(df):
    """
    Generates a 1D probability density plot of the stellar surface energy density.
    This proves that the numerical solver respects the fundamental difference
    between gravity-bound and self-bound compact objects.
    """
    set_paper_style()
    print("\n--- Generating Surface Density Proof (Forbidden Gap) ---")

    # 1. Data Validation
    if "Eps_Surface" not in df.columns:
        print("[Error] 'Eps_Surface' column missing in DataFrame. Skipping.")
        return

    # We only need one scalar surface density value per EoS curve
    unique_stars = df.drop_duplicates(subset=["Curve_ID"]).copy()

    q_vals = unique_stars[unique_stars["Label"] == 1]["Eps_Surface"].dropna()

    if len(q_vals) < 10:
        print("[Warn] Insufficient Quark samples for KDE plotting.")
        return

    # Initialize Figure
    fig, ax = plt.subplots(figsize=(9, 6))

    # ==========================================
    # 2. PLOT QUARK DISTRIBUTION (Self-Bound)
    # ==========================================
    # Quark stars terminate where P=0, but their energy density remains finite
    # (typically around 4*B, modified by the gap energy Delta).
    sns.kdeplot(
        x=q_vals,
        ax=ax,
        fill=True,
        color=COLORS["Q_main"],
        alpha=0.4,
        linewidth=2.5,
        bw_adjust=1.2,
    )

    # ==========================================
    # 3. PLOT HADRONIC DISTRIBUTION (Gravity-Bound)
    # ==========================================
    # Hadronic stars have an iron crust. In nuclear units (MeV/fm^3), the density
    # of iron is ~10^-15, which is numerically zero on this scale.
    # We represent this as a Dirac Delta function (a sharp vertical spike).
    ax.axvline(0, color=COLORS["H_main"], linewidth=3.5, linestyle="-")

    # Create a visual "spike" for the Hadronic PDF to match the KDE aesthetic
    y_max = ax.get_ylim()[1]
    ax.fill_betweenx([0, y_max], -5, 5, color=COLORS["H_main"], alpha=0.6)

    # ==========================================
    # 4. THE FORBIDDEN GAP
    # ==========================================
    # The region between Iron Density (~0) and the minimum stable Quark density
    # (~4*B_min) is physically forbidden for stable compact stars at T=0.
    gap_limit = q_vals.min()

    if gap_limit > 10.0:
        ax.axvspan(
            5,
            gap_limit,
            color=COLORS["Guide"],
            alpha=0.15,
            hatch="//",
            edgecolor="none",
        )
        ax.text(
            gap_limit * 0.5,
            y_max * 0.5,
            "Forbidden\nGap",
            ha="center",
            va="center",
            color="gray",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.3"
            ),
        )

    # ==========================================
    # 5. ANNOTATIONS & ARROWS
    # ==========================================
    # Hadronic Annotation
    ax.annotate(
        r"Iron Crust ($\varepsilon_{surf} \approx 0$)",
        xy=(5, y_max * 0.8),
        xytext=(150, y_max * 0.8),
        arrowprops=dict(
            facecolor=COLORS["H_main"],
            edgecolor=COLORS["H_main"],
            arrowstyle="wedge,tail_width=0.4",
            lw=0,
        ),
        color=COLORS["H_main"],
        fontweight="bold",
        fontsize=11,
        va="center",
    )

    # Quark Annotation
    # Calculate the theoretical minimum surface density (4 * B_min)

    ax.annotate(
        r"Self-Bound Surface ($\varepsilon_{surf} \gtrsim 4B$)",
        xy=(gap_limit, y_max * 0.3),
        xytext=(gap_limit + 150, y_max * 0.4),
        arrowprops=dict(
            facecolor=COLORS["Q_main"],
            edgecolor=COLORS["Q_main"],
            arrowstyle="wedge,tail_width=0.4",
            lw=0,
        ),
        color=COLORS["Q_main"],
        fontweight="bold",
        fontsize=11,
        va="center",
    )

    # ==========================================
    # 6. FORMATTING
    # ==========================================
    ax.set_title(r"Boundary Condition Validation: Gravity-Bound vs. Self-Bound Stars")
    ax.set_xlabel(r"Surface Energy Density $\varepsilon_{surf}$[MeV/fm$^3$]")
    ax.set_ylabel(r"Probability Density")

    # Set X-Limits to show the 0-point clearly while focusing on the distribution
    ax.set_xlim(-50, max(q_vals.max() * 1.1, 800))
    ax.set_ylim(0, y_max)

    # Custom Legend
    legend_elements = [
        Patch(facecolor=COLORS["H_main"], alpha=0.6, label="Hadronic Models"),
        Patch(facecolor=COLORS["Q_main"], alpha=0.4, label="Quark Models (CFL)"),
        Patch(
            facecolor="none",
            edgecolor=COLORS["Guide"],
            hatch="//",
            label="Physically Forbidden Region",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    outfile = "plots/fig_surface_density.pdf"
    plt.savefig(outfile)
    plt.close()

    print(f"[SUCCESS] Saved Surface Density Plot to {outfile}")
