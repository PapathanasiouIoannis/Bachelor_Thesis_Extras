# src/visualize/plot_theoretical_eos.py

"""
  Generates the Theoretical Equation of State Prior plot (Pressure vs Density).

Refactored:
  - DATA-DRIVEN: Instead of re-simulating curves, it extracts the exact (Pc, Eps_c)
    points from the generated DataFrame. This guarantees the plot perfectly
    represents the ML training data.
  - PUBLICATION READY: Replaced messy "spaghetti" lines with statistical
    90% confidence envelopes (5th to 95th percentiles) and median tracking.
  - PHYSICS LIMITS: Added dynamic theoretical constraint boundaries (Causality
    and Conformality).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_theoretical_eos(df):
    """
    Generates a publication-quality Equation of State band plot.

    Parameters:
    - df: The master DataFrame containing 'Eps_Central', 'P_Central', and 'Curve_ID'.
    """
    set_paper_style()
    print("\n--- Generating Theoretical EoS Prior (Statistical Bands) ---")

    # 1. Setup Interpolation Grids
    # We use Log-Log interpolation to compute smooth statistical percentiles.
    eps_log_min, eps_log_max = CONSTANTS["PLOT_EPS_LOG"]
    eps_common = np.logspace(np.log10(eps_log_min), np.log10(eps_log_max), 300)

    # Storage for interpolated pressure arrays
    p_hadronic_matrix = []
    p_quark_matrix = []

    # 2. Extract and Interpolate Curves
    grouped = df.groupby("Curve_ID")
    print(f"  > Interpolating {len(grouped)} EoS curves onto common grid...")

    for curve_id, group in tqdm(grouped, desc="Processing EoS", leave=False):
        # Sort strictly by density to ensure monotonic sequences for interpolation
        g = group.sort_values(by="Eps_Central")
        label = g["Label"].iloc[0]

        eps_vals = g["Eps_Central"].values
        p_vals = g["P_Central"].values

        # Require at least a few points to form a valid line
        if len(eps_vals) > 5:
            # Interpolate in Log-Log space
            f_eos = interp1d(
                np.log10(eps_vals),
                np.log10(p_vals),
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

            p_interp = 10 ** (f_eos(np.log10(eps_common)))

            if label == 0:
                p_hadronic_matrix.append(p_interp)
            else:
                p_quark_matrix.append(p_interp)

    # Convert to Numpy arrays for vectorized column-wise percentiles
    p_hadronic_matrix = np.array(p_hadronic_matrix)
    p_quark_matrix = np.array(p_quark_matrix)

    # 3. Initialize Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # ==========================================================================
    # Helper Function to Draw Confidence Envelopes
    # ==========================================================================
    def draw_eos_band(matrix, color, fade_color, hatch=None):
        if len(matrix) == 0:
            return

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate 5%, 50% (Median), and 95% percentiles ignoring NaNs
            p_low = np.nanpercentile(matrix, 5, axis=0)
            p_med = np.nanpercentile(matrix, 50, axis=0)
            p_high = np.nanpercentile(matrix, 95, axis=0)

        # Ensure the upper bound does not violate causality (P <= epsilon) visually
        p_high = np.minimum(p_high, eps_common)
        p_med = np.minimum(p_med, eps_common)

        # Draw Median Line
        ls = "--" if hatch else "-"
        ax.plot(eps_common, p_med, color=color, linestyle=ls, linewidth=2.0)

        # Draw 90% Confidence Envelope
        if hatch:
            # Hatched style for Quark models to distinguish overlap regions
            ax.fill_between(
                eps_common,
                p_low,
                p_high,
                facecolor="none",
                edgecolor=color,
                hatch=hatch,
                alpha=0.5,
                linewidth=0.0,
            )
            # Add a faint solid background so it pops
            ax.fill_between(
                eps_common,
                p_low,
                p_high,
                facecolor=fade_color,
                alpha=0.1,
                linewidth=0.0,
            )
        else:
            # Solid style for Hadronic models
            ax.fill_between(
                eps_common,
                p_low,
                p_high,
                facecolor=fade_color,
                alpha=0.3,
                linewidth=0.0,
            )

    # ==========================================================================

    # 4. Draw Bands
    draw_eos_band(p_hadronic_matrix, COLORS["H_main"], COLORS["H_fade"], hatch=None)
    draw_eos_band(p_quark_matrix, COLORS["Q_main"], COLORS["Q_fade"], hatch="////")

    # 5. Draw Physics Constraints
    x_guide = np.logspace(np.log10(eps_log_min), np.log10(eps_log_max), 100)

    # Causal Limit (P = epsilon, c_s = 1) -> Maximum Stiffness
    ax.plot(
        x_guide,
        x_guide,
        color=COLORS["Constraint"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=r"Causal Limit ($c_s=1$)",
    )

    # Conformal Limit (P = epsilon/3, c_s^2 = 1/3) -> QCD Asymptote
    ax.plot(
        x_guide,
        x_guide / 3.0,
        color=COLORS["Guide"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=r"Conformal Limit ($c_s^2=1/3$)",
    )

    # 6. Formatting & Aesthetics
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(eps_log_min, eps_log_max)
    ax.set_ylim(1e0, 2e3)

    ax.set_xlabel(r"Energy Density $\varepsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$[MeV/fm$^3$]")
    ax.set_title(r"Prior EoS Phase Space (90% Confidence Intervals)")

    # Custom Legend
    legend_elements = [
        Patch(facecolor=COLORS["H_fade"], alpha=0.6, label="Hadronic Models"),
        Patch(
            facecolor="white",
            edgecolor=COLORS["Q_main"],
            hatch="////",
            alpha=0.6,
            label="Quark Models (CFL)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["Constraint"],
            linestyle=":",
            label=r"Causal ($c_s=1$)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["Guide"],
            linestyle="--",
            label=r"Conformal ($c_s^2=1/3$)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Save Figure
    plt.tight_layout()
    outfile = "plots/fig_theoretical_eos.pdf"
    plt.savefig(outfile)
    plt.close()

    print(f"[SUCCESS] Saved EoS Phase Space Plot to {outfile}")
