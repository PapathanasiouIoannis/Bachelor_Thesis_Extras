# src/visualize/style_config.py

"""
Centralized Matplotlib style configuration for publication-ready figures.
Ensures uniform aesthetics, PDF vector embedding (Type 42), colorblind-safe
palettes, and correct LaTeX math formatting across all thesis plots.
"""

import matplotlib.pyplot as plt

# ==========================================
# 1. COLOR PALETTE (Publication & Colorblind Safe)
# ==========================================
# Optimized for high contrast, distinct monochrome printing, and visual clarity.
COLORS = {
    "H_main": "#0072B2",  # Blue (Colorblind-safe standard for Hadronic)
    "Q_main": "#D55E00",  # Vermilion/Orange (Colorblind-safe standard for Quark)
    "H_fade": "#56B4E9",  # Light Blue for fills/confidence bands
    "Q_fade": "#E69F00",  # Orange/Yellow for fills/confidence bands
    "Constraint": "#000000",  # Black for observational limits (J0740, GW170817)
    "Guide": "#999999",  # Gray for grids and reference lines
    "FalsePos": "#CC79A7",  # Purplish Pink (for ML false positives)
    "FalseNeg": "#009E73",  # Bluish Green (for ML false negatives)
}


def set_paper_style():
    """
    Configures Matplotlib global rcParams for ApJ/PRD/MNRAS quality plots.

    Configuration Details:
    - Font: Serif body (Times-like), Computer Modern for math ($...$).
    - Output: TrueType fonts (Type 42) embedded for journal PDF compliance.
    - DPI: 300 for high-resolution rasterized backdrops.
    - Ticks: Inward facing, present on all four sides.
    - Grid: Subtle dashed lines to guide the eye without cluttering.
    """
    import logging
    logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

    plt.rcParams.update(
        {
            # --- LAYOUT & RESOLUTION ---
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # --- FONTS & TEXT ---
            "font.family": "serif",  # Journal standard
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",  # Computer Modern for LaTeX math
            "pdf.fonttype": 42,  # Embed TrueType fonts (editable/compliant)
            "ps.fonttype": 42,
            # --- SIZES ---
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # --- TICKS ---
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,  # Ticks on all framing edges
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # --- LINES & GEOMETRY ---
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 1.0,  # Frame thickness
            # --- GRID ---
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": COLORS["Guide"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # --- LEGEND ---
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": COLORS["Guide"],
            "legend.fancybox": False,  # Square corners (academic style)
            "legend.loc": "best",
        }
    )
