import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib.patches import Patch
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS


def plot_statistical_bands(df):
    """
    Generates the Statistical Summary plot (Figure 3 in Thesis).

    Visualizes the 90% Confidence Intervals for:
    (a) Equation of State (Pressure vs Density)
    (b) Mass-Radius Relation (2D Probability Density)
    (c) Tidal Deformability (Lambda vs Mass)
    """
    set_paper_style()
    print("\n--- Generating 'Statistical Summary' (Theoretical Bands) ---")

    # ==========================================
    # 1. SETUP GRIDS FOR INTERPOLATION
    # ==========================================
    # Common grids are defined to aggregate distinct curves into statistical bands.

    # EoS Grid (Log-space for density)
    eps_common = np.logspace(
        np.log10(CONSTANTS["PLOT_EPS_LOG"][0]),
        np.log10(CONSTANTS["PLOT_EPS_LOG"][1]),
        500,
    )

    # Mass Grid (Linear-space for macroscopic properties)
    mass_common = np.linspace(
        CONSTANTS["PLOT_M_LIM"][0] + 0.1, CONSTANTS["PLOT_M_LIM"][1], 400
    )

    # Storage Lists
    eos_h_matrix, eos_q_matrix = [], []
    lam_h_matrix, lam_q_matrix = [], []

    # M-R Point Clouds (For 2D KDE Contours)
    mr_h_r, mr_h_m = [], []
    mr_q_r, mr_q_m = [], []

    # ==========================================
    # 2. DATA AGGREGATION
    # ==========================================
    grouped = df.groupby("Curve_ID")
    curve_ids = list(grouped.groups.keys())

    print(f"Aggregating statistics from {len(curve_ids)} curves...")

    for name, group in tqdm(grouped, desc="Processing Curves"):
        # Sort by density to ensure monotonic interpolation
        g = group.sort_values(by="Eps_Central")
        label = g["Label"].iloc[0]

        # --- A. EoS Interpolation (P vs Eps) ---
        e_vals = g["Eps_Central"].values
        p_vals = g["P_Central"].values

        if len(e_vals) > 5:
            # Interpolate in Log-Log space for stability
            f_eos = interp1d(
                np.log10(e_vals),
                np.log10(p_vals),
                bounds_error=False,
                fill_value=np.nan,
            )
            p_interp = 10 ** f_eos(np.log10(eps_common))

            if label == 0:
                eos_h_matrix.append(p_interp)
            else:
                eos_q_matrix.append(p_interp)

        # --- B. M-R Points (Stable Branch Only) ---
        m_max = g["Mass"].max()
        g_stable = g[g["Mass"] <= m_max]

        if label == 0:
            mr_h_r.extend(g_stable["Radius"].values)
            mr_h_m.extend(g_stable["Mass"].values)
        else:
            mr_q_r.extend(g_stable["Radius"].values)
            mr_q_m.extend(g_stable["Mass"].values)

        # --- C. Tidal Interpolation (Lambda vs Mass) ---
        # Lambda is interpolated against Mass to form bands
        g_tidal = g_stable.drop_duplicates("Mass").sort_values("Mass")

        if len(g_tidal) > 5:
            m_vals = g_tidal["Mass"].values
            l_vals = g_tidal["Lambda"].values
            # Clamp Lambda to avoid log(0) errors
            l_vals = np.maximum(l_vals, 1e-5)

            f_lam = interp1d(
                m_vals, np.log10(l_vals), bounds_error=False, fill_value=np.nan
            )
            l_interp = 10 ** f_lam(mass_common)

            if label == 0:
                lam_h_matrix.append(l_interp)
            else:
                lam_q_matrix.append(l_interp)

    # Convert lists to numpy arrays for vectorized percentile calculation
    eos_h_matrix = np.array(eos_h_matrix)
    eos_q_matrix = np.array(eos_q_matrix)
    lam_h_matrix = np.array(lam_h_matrix)
    lam_q_matrix = np.array(lam_q_matrix)

    # ==========================================
    # 3. PLOTTING
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.25)

    # --- Helper: Draw Statistical Band ---
    def draw_band(ax, x, matrix, color, label, hatch=None, linestyle="-"):
        """Plots the 5th-95th percentile bands and the median line."""
        # Calculate Percentiles (ignoring NaNs)
        low = np.nanpercentile(matrix, 5, axis=0)
        med = np.nanpercentile(matrix, 50, axis=0)
        high = np.nanpercentile(matrix, 95, axis=0)

        # Gaussian Smoothing is applied for aesthetic bands
        sigma = 2.0
        low = gaussian_filter1d(low, sigma)
        med = gaussian_filter1d(med, sigma)
        high = gaussian_filter1d(high, sigma)

        # Plot Median
        ax.plot(x, med, color=color, linestyle=linestyle, linewidth=2, label=label)

        # Plot Bounds
        ax.plot(x, low, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.5)
        ax.plot(x, high, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.5)

        # Fill Band
        if hatch:
            # Hatched fill (Quark style)
            ax.fill_between(
                x,
                low,
                high,
                facecolor="none",
                edgecolor=color,
                hatch=hatch,
                alpha=0.5,
                linewidth=0,
            )
            ax.fill_between(x, low, high, facecolor=color, alpha=0.1, linewidth=0)
        else:
            # Solid fill (Hadronic style)
            ax.fill_between(x, low, high, facecolor=color, alpha=0.25, linewidth=0)

    # --- PANEL A: Equation of State ---
    ax = axes[0]
    draw_band(ax, eps_common, eos_h_matrix, COLORS["H_main"], "Hadronic", linestyle="-")
    draw_band(
        ax,
        eps_common,
        eos_q_matrix,
        COLORS["Q_main"],
        "Quark",
        hatch="////",
        linestyle="--",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(CONSTANTS["PLOT_EPS_LOG"])
    ax.set_ylim(1e0, 2e3)
    ax.set_xlabel(r"Energy Density $\varepsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax.set_title(r"(a) Equation of State")

    # Causal Limit Guide
    x_guide = np.logspace(2, 4, 50)
    ax.plot(x_guide, x_guide, "k:", alpha=0.5, label="Causal Limit")

    # --- PANEL B: Mass-Radius (Contours) ---
    ax = axes[1]

    # Define grid for KDE
    xx, yy = np.mgrid[4:22:100j, 0:4.5:100j]
    pos = np.vstack([xx.ravel(), yy.ravel()])

    def draw_contour(r_pts, m_pts, color, hatch=None, ls="-"):
        """Computes and plots 90% confidence contours via KDE."""
        # Cap point count if extreme memory usage is detected
        if len(r_pts) > 50000:
            idx = np.random.choice(len(r_pts), 50000, replace=False)
            r_pts, m_pts = np.array(r_pts)[idx], np.array(m_pts)[idx]

        # Gaussian Kernel Density Estimation
        kde = gaussian_kde(np.vstack([r_pts, m_pts]))
        kde.set_bandwidth(kde.factor * 1.2)
        f = np.reshape(kde(pos).T, xx.shape)

        # 90% Confidence Interval Contour (0.1 * Max Density)
        levels = [0.1 * f.max(), f.max()]

        if hatch:
            cntr = ax.contourf(xx, yy, f, levels=levels, colors="none", hatches=[hatch])
            cntr.set_edgecolor(color)
            cntr.set_linewidth(0.0)
            cntr.set_alpha(0.5)
            ax.contour(
                xx,
                yy,
                f,
                levels=levels[:1],
                colors=[color],
                linewidths=2,
                linestyles=ls,
            )
        else:
            ax.contourf(xx, yy, f, levels=levels, colors=[color], alpha=0.25)
            ax.contour(
                xx,
                yy,
                f,
                levels=levels[:1],
                colors=[color],
                linewidths=2,
                linestyles=ls,
            )

    draw_contour(mr_h_r, mr_h_m, COLORS["H_main"], hatch=None, ls="-")
    draw_contour(mr_q_r, mr_q_m, COLORS["Q_main"], hatch="////", ls="--")

    # Physics Constraint: Buchdahl Limit
    m_buch = np.linspace(0, 4.5, 100)
    r_buch = CONSTANTS["BUCHDAHL_FACTOR"] * CONSTANTS["A_CONV"] * m_buch
    ax.fill_betweenx(m_buch, 0, r_buch, color="gray", alpha=0.1, zorder=-10)
    ax.text(6.0, 3.8, "GR Forbidden", fontsize=9, color="gray", ha="center")

    ax.set_xlim(CONSTANTS["PLOT_R_LIM"])
    ax.set_ylim(CONSTANTS["PLOT_M_LIM"])
    ax.set_xlabel(r"Radius $R$ [km]")
    ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_title(r"(b) Mass-Radius Relation")

    # --- PANEL C: Tidal Deformability ---
    ax = axes[2]
    draw_band(
        ax, mass_common, lam_h_matrix, COLORS["H_main"], "Hadronic", linestyle="-"
    )
    draw_band(
        ax,
        mass_common,
        lam_q_matrix,
        COLORS["Q_main"],
        "Quark",
        hatch="////",
        linestyle="--",
    )

    ax.set_yscale("log")
    ax.set_xlim(CONSTANTS["PLOT_M_LIM"])
    ax.set_ylim(0.1, 5000)
    ax.set_xlabel(r"Mass $M$ [$M_{\odot}$]")
    ax.set_ylabel(r"Tidal Deformability $\Lambda$")
    ax.set_title(r"(c) Tidal Deformability")

    # Global Legend
    h_patch = Patch(
        facecolor=COLORS["H_main"],
        alpha=0.3,
        edgecolor=COLORS["H_main"],
        label="Hadronic (90% CI)",
    )
    q_patch = Patch(
        facecolor="white",
        hatch="////",
        edgecolor=COLORS["Q_main"],
        label="Quark (90% CI)",
    )

    fig.legend(
        handles=[h_patch, q_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("plots/fig_statistical_bands.pdf")
    plt.close()
    print("[Success] Saved Statistical Bands Plot.")
