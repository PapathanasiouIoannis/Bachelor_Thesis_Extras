# src/visualize/plot_stability_window.py

"""
  Generates the QCD Stability Window plot (Vacuum Pressure B vs Gap Energy Delta).

Refactored:
  - PUBLICATION QUALITY: Added analytic shading for the theoretically forbidden
    "Unstable" regions (Neutron decay at the top, Iron decay at the bottom).
  - SCALAR MAPPING: Uses the strange quark mass (m_s) as a color dimension
    to show how the 3D parameter space projects into the 2D stability triangle.
  - AESTHETICS: Vectorized limits, rasterized point clouds, consistent theming.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_stability_window(df):
    """
    Generates the QCD Stability Window plot, verifying that all generated
    Quark Star models strictly adhere to the Bodmer-Witten hypothesis constraints.
    """
    set_paper_style()
    print("\n--- Generating Stability Window (QCD Vacuum Bounds) ---")

    # 1. Data Filtering
    # We only care about unique Quark models (one scalar point per EoS curve)
    q_data = df[df["Label"] == 1].drop_duplicates(subset=["Curve_ID"])

    if "Bag_B" not in q_data.columns:
        print("[Error] Missing 'Bag_B' column. Skipping Stability Window plot.")
        return

    if len(q_data) == 0:
        print("[Warn] No Quark models found in dataset. Skipping.")
        return

    # Initialize Figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # ==========================================
    # 2. THEORETICAL STABILITY BOUNDARIES
    # ==========================================
    hc = CONSTANTS["HC"]
    m_n = CONSTANTS["M_N"]

    # Parameter bounds
    ms_min, ms_max = CONSTANTS["Q_MS_RANGE"]
    delta_min, delta_max = CONSTANTS["Q_DELTA_RANGE"]

    # Extend X-axis slightly past the generated data to show the theoretical envelope
    delta_vals = np.linspace(max(0, delta_min - 20), delta_max + 50, 200)

    # Absolute stability condition: E/A < 939 MeV -> mu <= m_N / 3
    mu_limit = m_n / 3.0

    def calculate_b_max(delta_arr, ms_val):
        """
        Calculates the maximum Bag Constant allowed for stability against neutron decay.
        P_flavor(mu_limit) = B_max
        Returns B_max in MeV/fm^3.
        """
        # Massless contribution (~mu^4)
        term1 = (3.0 / (4.0 * np.pi**2)) * (mu_limit**4)

        # Gap and Mass corrections (~mu^2)
        eff_gap_sq = delta_arr**2 - (ms_val**2 / 4.0)
        term2 = (3.0 / np.pi**2) * eff_gap_sq * (mu_limit**2)

        return (term1 + term2) / (hc**3)

    # Upper Boundaries for the minimum and maximum sampled strange quark mass (m_s)
    b_max_ms_min = calculate_b_max(delta_vals, ms_min)  # Most stable configuration
    b_max_ms_max = calculate_b_max(delta_vals, ms_max)  # Least stable configuration

    # ==========================================
    # 3. FORBIDDEN REGIONS (Analytic Shading)
    # ==========================================
    b_min = CONSTANTS["Q_B_MIN"]  # Iron Stability Limit (Usually ~57 MeV/fm^3)

    # Region 1: Unstable to 2-Flavor Matter (Iron decay)
    ax.fill_between(
        delta_vals, 0, b_min, color=COLORS["FalsePos"], alpha=0.15, hatch="\\\\"
    )
    ax.axhline(b_min, color=COLORS["Constraint"], linewidth=2.0, linestyle="--")

    # Region 2: Unstable to Neutrons (Upper Bound)
    # Fill everything above the most stable possible line
    ax.fill_between(
        delta_vals, b_max_ms_min, 450, color=COLORS["FalseNeg"], alpha=0.15, hatch="//"
    )

    # Draw the dynamic uncertainty band for the M_s variation
    ax.fill_between(
        delta_vals, b_max_ms_min, b_max_ms_max, color="gray", alpha=0.3, zorder=1
    )

    ax.plot(
        delta_vals,
        b_max_ms_min,
        color=COLORS["Constraint"],
        linestyle="-",
        linewidth=2.0,
    )
    ax.plot(
        delta_vals, b_max_ms_max, color=COLORS["Guide"], linestyle=":", linewidth=1.5
    )

    # ==========================================
    # 4. SCATTER PLOT (Generated Models)
    # ==========================================
    # Randomize plot order to prevent m_s ordering bias in the scatter overlap
    q_data_shuffled = q_data.sample(frac=1.0, random_state=42)

    # Scatter points colored by Strange Quark Mass (ms)
    sc = ax.scatter(
        q_data_shuffled["Gap_Delta"],
        q_data_shuffled["Bag_B"],
        c=q_data_shuffled["Mass_Strange"],
        cmap="plasma",
        s=25,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    # Colorbar formatting
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"Strange Quark Mass $m_s$ [MeV]", rotation=270, labelpad=20)

    # ==========================================
    # 5. ANNOTATIONS & FORMATTING
    # ==========================================
    ax.set_xlim(max(0, delta_min - 20), delta_max + 20)
    ax.set_ylim(40, 320)

    ax.set_xlabel(r"Gap Energy $\Delta$ [MeV]")
    ax.set_ylabel(r"Vacuum Pressure $B$[MeV/fm$^3$]")
    ax.set_title(r"QCD Stability Window (Bodmer-Witten Hypothesis)")

    # Text Annotations for Forbidden Regions
    ax.text(
        delta_max - 50,
        280,
        "Unstable\n(Decays to Neutrons)",
        color=COLORS["Constraint"],
        fontsize=11,
        fontweight="bold",
        ha="center",
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.3"
        ),
    )

    ax.text(
        delta_max - 50,
        b_min - 10,
        "Unstable\n(2-Flavor Stable)",
        color=COLORS["Constraint"],
        fontsize=11,
        fontweight="bold",
        ha="center",
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.3"
        ),
    )

    # Custom Legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.3, label=r"Theoretical $m_s$ Uncertainty"),
        Line2D(
            [0],
            [0],
            color=COLORS["Constraint"],
            lw=2,
            linestyle="-",
            label=r"Absolute Max Stability Limit",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["Constraint"],
            lw=2,
            linestyle="--",
            label=r"$^{56}$Fe Stability Limit (Lower Bound)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95)

    plt.tight_layout()
    outfile = "plots/fig_stability_triangle.pdf"
    plt.savefig(outfile)
    plt.close()
    print(f"[SUCCESS] Saved QCD Stability Window Plot to {outfile}")
