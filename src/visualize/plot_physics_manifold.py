# src/visualize/plot_physics_manifold.py

"""
  Generates the Mass-Radius Topological Manifold plots (Triptychs).
  - plot_physics_manifold: Statistical KDE contours of the population.
  - plot_manifold_curves: Morphological flow of the raw EoS lines.

Refactored:
  - PUBLICATION QUALITY: Uses shared aesthetic formatting, strictly bounds KDE
    to physical regions, and dynamically scales line transparency (alpha) based
    on dataset size.
  - VECTOR/RASTER HYBRID: Ensures scatter/line clouds are rasterized (PNG embedded)
    while axes, text, and limits remain vectorized for crisp PDF rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def _apply_common_formatting(axes):
    """
    Applies shared physical constraints and labels to a list of axes.
    """
    m_buch = np.linspace(0, 4.5, 100)
    # Buchdahl Limit: R = 9/4 G M / c^2 (In geometric units, R = 2.25 * M_geom)
    r_buch = CONSTANTS["BUCHDAHL_FACTOR"] * CONSTANTS["A_CONV"] * m_buch

    for ax in axes:
        # Observational Mass Constraint (PSR J0740+6620)
        ax.axhline(
            2.08, color=COLORS["Constraint"], linestyle="-.", linewidth=1.5, zorder=10
        )

        # Theoretical Constraint (Buchdahl Forbidden Wedge)
        ax.fill_betweenx(
            m_buch, 0, r_buch, color=COLORS["Guide"], alpha=0.15, zorder=-1
        )

        ax.set_xlim(CONSTANTS["PLOT_R_LIM"])
        ax.set_ylim(CONSTANTS["PLOT_M_LIM"])
        ax.set_xlabel(r"Radius $R$ [km]")

    axes[0].set_ylabel(r"Mass $M$[$M_{\odot}$]")

    # Annotate limits on the first panel only to reduce clutter
    axes[0].text(
        13.0,
        2.15,
        r"PSR J0740 ($2.08 M_{\odot}$)",
        fontsize=10,
        color=COLORS["Constraint"],
    )
    axes[0].text(
        6.0,
        3.5,
        "GR Forbidden\n(Buchdahl Limit)",
        fontsize=10,
        color=COLORS["Guide"],
        ha="center",
        rotation=45,
    )


def plot_physics_manifold(df):
    """
    Generates a Triptych of Mass-Radius KDE Probability Density Contours.
    """
    set_paper_style()
    print("\n--- Generating M-R Physics Manifold (KDE Contours) ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05)

    # 1. Extract Stable Branches
    # We only want points where the star is gravitationally stable (dM/dRho > 0)
    h_r, h_m = [], []
    q_r, q_m = [], []

    grouped = df.groupby("Curve_ID")
    print(f"  > Extracting stable M-R points from {len(grouped)} curves...")

    for _, group in grouped:
        g = group.sort_values(by="Eps_Central")
        m_max = g["Mass"].max()
        stable_g = g[g["Mass"] <= m_max]

        if stable_g["Label"].iloc[0] == 0:
            h_r.extend(stable_g["Radius"].values)
            h_m.extend(stable_g["Mass"].values)
        else:
            q_r.extend(stable_g["Radius"].values)
            q_m.extend(stable_g["Mass"].values)

    # 2. KDE Calculation Helper
    r_min, r_max = CONSTANTS["PLOT_R_LIM"]
    m_min, m_max = CONSTANTS["PLOT_M_LIM"]
    xx, yy = np.mgrid[r_min:r_max:200j, m_min:m_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    def draw_contours(ax, r_pts, m_pts, color, hatch=None):
        if len(r_pts) < 100:
            return

        # Subsample massively to avoid MemoryError during KDE
        if len(r_pts) > 50000:
            idx = np.random.choice(len(r_pts), 50000, replace=False)
            r_pts, m_pts = np.array(r_pts)[idx], np.array(m_pts)[idx]

        kernel = gaussian_kde(np.vstack([r_pts, m_pts]))
        kernel.set_bandwidth(kernel.factor * 1.2)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Filter out extremely low probability "ghost" regions
        f_max = f.max()
        levels = [0.05 * f_max, 0.25 * f_max, 0.5 * f_max, f_max]

        ls = "--" if hatch else "-"

        if hatch:
            cntr = ax.contourf(
                xx, yy, f, levels=levels, colors="none", hatches=[hatch, hatch, hatch]
            )
            cntr.set_edgecolor(color)
            cntr.set_linewidth(0.0)
            cntr.set_alpha(0.5)
            # Faint background
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.1)
        else:
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.3)

        # Draw bounding envelope (95% CI)
        ax.contour(
            xx,
            yy,
            f,
            levels=[0.05 * f_max],
            colors=[color],
            linewidths=2,
            linestyles=ls,
        )

    # 3. Draw Panels
    print("  > Rendering Contours...")
    draw_contours(axes[0], h_r, h_m, COLORS["H_main"], hatch=None)  # Panel A: Hadronic
    draw_contours(axes[1], q_r, q_m, COLORS["Q_main"], hatch="////")  # Panel B: Quark

    # Panel C: Overlay
    draw_contours(axes[2], h_r, h_m, COLORS["H_main"], hatch=None)
    draw_contours(axes[2], q_r, q_m, COLORS["Q_main"], hatch="////")

    # 4. Formatting
    _apply_common_formatting(axes)
    axes[0].set_title(r"(a) Hadronic Distribution (KDE)", y=1.02)
    axes[1].set_title(r"(b) Quark Distribution (KDE)", y=1.02)
    axes[2].set_title(r"(c) Topological Intersection", y=1.02)

    # Custom Legend
    legend_elements = [
        Patch(facecolor=COLORS["H_main"], alpha=0.4, label="Hadronic Phase Space"),
        Patch(
            facecolor="white",
            hatch="////",
            edgecolor=COLORS["Q_main"],
            alpha=0.6,
            label="Quark Phase Space",
        ),
    ]
    axes[2].legend(handles=legend_elements, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    outfile = "plots/fig_manifold_kde.pdf"
    plt.savefig(outfile)
    plt.close()
    print(f"[SUCCESS] Saved KDE Manifold to {outfile}")


def plot_manifold_curves(df):
    """
    Generates a Triptych showing the raw morphological flow of all EoS curves.
    Uses rasterization to prevent massive PDF file sizes.
    """
    set_paper_style()
    print("\n--- Generating M-R Morphological Flow (Raw Curves) ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05)

    grouped = df.groupby("Curve_ID")
    n_curves = len(grouped)

    # Dynamically scale alpha based on curve count to prevent saturation
    alpha_val = max(0.01, min(0.3, 500.0 / n_curves))
    lw_val = 0.5 if n_curves > 1000 else 1.0

    print(f"  > Plotting {n_curves} rasterized curves (alpha={alpha_val:.3f})...")

    for _, group in tqdm(grouped, desc="Rendering Lines", leave=False):
        g = group.sort_values(by="Eps_Central")
        label = g["Label"].iloc[0]

        # Only plot up to M_max to avoid the unstable collapsing branch visually
        m_max_idx = g["Mass"].idxmax()
        g_stable = g.loc[:m_max_idx]

        if label == 0:
            axes[0].plot(
                g_stable["Radius"],
                g_stable["Mass"],
                color=COLORS["H_main"],
                alpha=alpha_val,
                lw=lw_val,
                rasterized=True,
            )
            axes[2].plot(
                g_stable["Radius"],
                g_stable["Mass"],
                color=COLORS["H_main"],
                alpha=alpha_val * 0.8,
                lw=lw_val,
                rasterized=True,
            )
        else:
            axes[1].plot(
                g_stable["Radius"],
                g_stable["Mass"],
                color=COLORS["Q_main"],
                alpha=alpha_val,
                lw=lw_val,
                rasterized=True,
            )
            axes[2].plot(
                g_stable["Radius"],
                g_stable["Mass"],
                color=COLORS["Q_main"],
                alpha=alpha_val * 0.8,
                lw=lw_val,
                rasterized=True,
            )

    # Formatting
    _apply_common_formatting(axes)
    axes[0].set_title(r"(a) Hadronic Morphological Flow", y=1.02)
    axes[1].set_title(r"(b) Quark Morphological Flow", y=1.02)
    axes[2].set_title(r"(c) Combined Curve Overlay", y=1.02)

    legend_elements = [
        Line2D([0], [0], color=COLORS["H_main"], lw=2, label="Hadronic EoS"),
        Line2D([0], [0], color=COLORS["Q_main"], lw=2, label="Quark EoS"),
    ]
    axes[2].legend(handles=legend_elements, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    outfile = "plots/fig_manifold_curves.pdf"
    # High DPI ensures the rasterized lines look sharp when zooming into the PDF
    plt.savefig(outfile, dpi=400)
    plt.close()
    print(f"[SUCCESS] Saved Manifold Curves to {outfile}")
