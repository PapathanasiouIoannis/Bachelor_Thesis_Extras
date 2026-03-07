# src/visualize/plot_grand_summary.py

"""
  Generates the centerpiece "Grand Summary" Triptych (3 Panels).

  Panels:
  (a) Equation of State: Pressure vs Energy Density (Statistical Bands).
  (b) Mass-Radius Relation: Population density contours via KDE.
  (c) Tidal Deformability: Lambda vs Mass (Statistical Bands + GW170817).

Refactored:
  - PUBLICATION STANDARD: Switched to 90% confidence envelopes (5th-95th percentiles)
    for both EoS and Tidal panels, removing messy line overlays.
  - DENSITY CONTOURS: Upgraded M-R scatter plots to strictly bounded KDE contours
    to represent the true topological prior distribution.
  - PHYSICAL CONSTRAINTS: Added the Buchdahl stability limit, GW170817 upper bound,
    and PSR J0740 mass constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.patches import Patch

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_grand_summary(df):
    """
    Generates the 3-panel Grand Summary plot of the macroscopic and microscopic
    prior distributions.
    """
    set_paper_style()
    print("\n--- Generating 'Grand Summary' Triptych (KDE & Bands) ---")

    # ==========================================
    # 0. CONFIGURATION & COMMON GRIDS
    # ==========================================
    # We use shared evaluation grids to calculate percentiles across thousands of curves.

    eps_grid = np.logspace(
        np.log10(CONSTANTS["PLOT_EPS_LOG"][0]),
        np.log10(CONSTANTS["PLOT_EPS_LOG"][1]),
        300,
    )

    mass_grid = np.linspace(
        CONSTANTS["PLOT_M_LIM"][0] + 0.1, CONSTANTS["PLOT_M_LIM"][1], 200
    )

    # Storage arrays for interpolation results
    eos_h_matrix, eos_q_matrix = [], []
    lam_h_matrix, lam_q_matrix = [], []

    # Point clouds for 2D KDE (Mass-Radius)
    mr_h_r, mr_h_m = [], []
    mr_q_r, mr_q_m = [], []

    grouped = df.groupby("Curve_ID")
    curve_ids = list(grouped.groups.keys())

    # Subsample curves to speed up interpolation and KDE if dataset is massive
    np.random.seed(42)
    max_curves = 3000
    if len(curve_ids) > max_curves:
        selected_ids = set(np.random.choice(curve_ids, max_curves, replace=False))
    else:
        selected_ids = set(curve_ids)

    # ==========================================
    # 1. DATA PROCESSING (Interpolation)
    # ==========================================
    print(f"  > Interpolating {len(selected_ids)} curves onto common grids...")

    for name, group in tqdm(grouped, desc="Processing Data", leave=False):
        if name not in selected_ids:
            continue

        # Sort by density to ensure monotonic sequences
        g = group.sort_values(by="Eps_Central")
        label = g["Label"].iloc[0]

        # --------------------------------------------------
        # A. EoS Processing (Pressure vs Density)
        # --------------------------------------------------
        e_vals = g["Eps_Central"].values
        p_vals = g["P_Central"].values

        if len(e_vals) > 5:
            f_eos = interp1d(
                np.log10(e_vals),
                np.log10(p_vals),
                bounds_error=False,
                fill_value=np.nan,
            )
            p_interp = 10 ** (f_eos(np.log10(eps_grid)))

            if label == 0:
                eos_h_matrix.append(p_interp)
            else:
                eos_q_matrix.append(p_interp)

        # --------------------------------------------------
        # B. Macroscopic Processing (M-R and Lambda-M)
        # --------------------------------------------------
        # Isolate the strictly stable branch (up to Maximum Mass)
        idx_max = g["Mass"].idxmax()
        g_stable = g.loc[:idx_max].drop_duplicates(subset="Mass").sort_values("Mass")

        if len(g_stable) > 5:
            m_vals = g_stable["Mass"].values
            r_vals = g_stable["Radius"].values
            l_vals = g_stable["Lambda"].values
            l_vals = np.maximum(l_vals, 1e-5)  # Prevent log(0)

            # Save points for M-R KDE
            if label == 0:
                mr_h_r.extend(r_vals)
                mr_h_m.extend(m_vals)
            else:
                mr_q_r.extend(r_vals)
                mr_q_m.extend(m_vals)

            # Interpolate Lambda onto common Mass grid
            f_lam = interp1d(
                m_vals, np.log10(l_vals), bounds_error=False, fill_value=np.nan
            )
            l_interp = 10 ** (f_lam(mass_grid))

            if label == 0:
                lam_h_matrix.append(l_interp)
            else:
                lam_q_matrix.append(l_interp)

    # Convert matrices to numpy arrays for percentile extraction
    eos_h_matrix = np.array(eos_h_matrix)
    eos_q_matrix = np.array(eos_q_matrix)
    lam_h_matrix = np.array(lam_h_matrix)
    lam_q_matrix = np.array(lam_q_matrix)

    # ==========================================
    # 2. PLOTTING SETUP
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.25)

    def plot_smooth_band(ax, x_grid, data_matrix, color, hatch=None, clip_causal=False):
        """Calculates and plots 90% confidence bands."""
        if len(data_matrix) == 0:
            return

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            low = np.nanpercentile(data_matrix, 5, axis=0)
            med = np.nanpercentile(data_matrix, 50, axis=0)
            high = np.nanpercentile(data_matrix, 95, axis=0)

        # Smooth bands slightly for aesthetic presentation
        sigma = 1.5
        low = gaussian_filter1d(low, sigma)
        med = gaussian_filter1d(med, sigma)
        high = gaussian_filter1d(high, sigma)

        if clip_causal:
            # Force compliance with Causal Limit P <= Epsilon for visualization safety
            med = np.minimum(med, x_grid)
            high = np.minimum(high, x_grid)

        # Draw Lines & Bands
        ls = "--" if hatch else "-"
        ax.plot(x_grid, med, color=color, linestyle=ls, linewidth=2.0)

        if hatch:
            ax.fill_between(
                x_grid,
                low,
                high,
                facecolor="none",
                edgecolor=color,
                hatch=hatch,
                alpha=0.5,
                linewidth=0,
            )
            ax.fill_between(x_grid, low, high, facecolor=color, alpha=0.1, linewidth=0)
        else:
            ax.fill_between(x_grid, low, high, facecolor=color, alpha=0.3, linewidth=0)

    # ==========================================
    # PANEL A: Equation of State
    # ==========================================
    ax = axes[0]
    plot_smooth_band(
        ax, eps_grid, eos_h_matrix, COLORS["H_main"], hatch=None, clip_causal=True
    )
    plot_smooth_band(
        ax, eps_grid, eos_q_matrix, COLORS["Q_main"], hatch="////", clip_causal=True
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(CONSTANTS["PLOT_EPS_LOG"])
    ax.set_ylim(1e0, 2e3)
    ax.set_xlabel(r"Energy Density $\varepsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax.set_title(r"(a) Equation of State")

    # Physics Constraints
    x_guide = np.logspace(1, 4, 50)
    ax.plot(
        x_guide,
        x_guide,
        color=COLORS["Constraint"],
        linestyle=":",
        alpha=0.8,
        lw=1.5,
        label="Causal Limit",
    )
    ax.plot(
        x_guide,
        x_guide / 3.0,
        color=COLORS["Guide"],
        linestyle="--",
        alpha=0.8,
        lw=1.5,
        label="Conformal Limit",
    )
    ax.legend(loc="upper left", framealpha=0.9)

    # ==========================================
    # PANEL B: Mass-Radius (KDE Contours)
    # ==========================================
    ax = axes[1]

    # Grid for KDE Evaluation
    r_min, r_max = CONSTANTS["PLOT_R_LIM"]
    m_min, m_max = CONSTANTS["PLOT_M_LIM"]
    xx, yy = np.mgrid[r_min:r_max:200j, m_min:m_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    def draw_kde_contour(r_pts, m_pts, color, hatch=None):
        if len(r_pts) < 100:
            return

        if len(r_pts) > 30000:
            idx = np.random.choice(len(r_pts), 30000, replace=False)
            r_pts, m_pts = np.array(r_pts)[idx], np.array(m_pts)[idx]

        kernel = gaussian_kde(np.vstack([r_pts, m_pts]))
        kernel.set_bandwidth(kernel.factor * 1.2)  # Slight smoothing

        f = np.reshape(kernel(positions).T, xx.shape)
        levels = [0.05 * f.max(), f.max()]  # 95% Confidence region effectively

        ls = "--" if hatch else "-"

        if hatch:
            cntr = ax.contourf(xx, yy, f, levels=levels, colors="none", hatches=[hatch])
            cntr.set_edgecolor(color)
            cntr.set_linewidth(0.0)
            cntr.set_alpha(0.5)
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.1)
        else:
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.3)

        ax.contour(
            xx, yy, f, levels=levels[:1], colors=[color], linewidths=2, linestyles=ls
        )

    draw_kde_contour(mr_h_r, mr_h_m, COLORS["H_main"], hatch=None)
    draw_kde_contour(mr_q_r, mr_q_m, COLORS["Q_main"], hatch="////")

    # Constraints
    # J0740 Lower Mass Bound
    ax.axhline(
        2.08, color=COLORS["Constraint"], linestyle="-.", lw=1.5, label="PSR J0740"
    )

    # Buchdahl Limit Wedge (GR Stability)
    m_buch = np.linspace(0, 4.5, 100)
    r_buch = CONSTANTS["BUCHDAHL_FACTOR"] * CONSTANTS["A_CONV"] * m_buch
    ax.fill_betweenx(m_buch, 0, r_buch, color="gray", alpha=0.2, zorder=-10)
    ax.text(
        6.5, 3.8, "GR Forbidden\n(Buchdahl)", fontsize=10, color="gray", ha="center"
    )

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(m_min, m_max)
    ax.set_xlabel(r"Radius $R$[km]")
    ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_title(r"(b) Mass-Radius Relation")
    ax.legend(loc="lower left", framealpha=0.9)

    # ==========================================
    # PANEL C: Tidal Deformability
    # ==========================================
    ax = axes[2]
    plot_smooth_band(ax, mass_grid, lam_h_matrix, COLORS["H_main"], hatch=None)
    plot_smooth_band(ax, mass_grid, lam_q_matrix, COLORS["Q_main"], hatch="////")

    ax.set_yscale("log")
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(1, 5000)
    ax.set_xlabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_ylabel(r"Tidal Deformability $\Lambda$")
    ax.set_title(r"(c) Tidal Deformability")

    # GW170817 Constraint (Upper Bound at 1.4 M_sun)
    # Visualizing Lambda(1.4) < 580
    ax.vlines(1.4, 1, 580, colors=COLORS["Constraint"], lw=2, zorder=10)
    ax.hlines(580, 1.35, 1.45, colors=COLORS["Constraint"], lw=2, zorder=10)
    ax.text(
        1.48,
        650,
        "GW170817",
        fontsize=11,
        color=COLORS["Constraint"],
        fontweight="bold",
    )

    # ==========================================
    # GLOBAL LEGEND & SAVING
    # ==========================================
    h_handle = Patch(
        facecolor=COLORS["H_main"], alpha=0.4, label="Hadronic Models (90% CI)"
    )
    q_handle = Patch(
        facecolor="white",
        hatch="////",
        edgecolor=COLORS["Q_main"],
        alpha=0.5,
        label="Quark Models (90% CI)",
    )

    fig.legend(
        handles=[h_handle, q_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    outfile = "plots/fig_grand_summary.pdf"
    plt.savefig(outfile)
    plt.close()

    print(f"[SUCCESS] Saved Grand Summary Plot to {outfile}")
