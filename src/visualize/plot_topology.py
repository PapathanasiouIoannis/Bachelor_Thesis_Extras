# src/visualize/plot_topology.py

"""
Consolidated module for generating all Topological Classification Maps.
These maps show the topological decision boundaries and confidence intervals
learned by the models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS
from src.visualize.eval_utils import evaluate_grid, bifurcated_dual_knn_evaluation, evaluate_base_models_grid


def plot_geo_M_vs_R(models_dict):
    """
    Plots the Safe Classification Topology Map for Model Geo (M vs R).
    Demonstrates the severe degeneracy (Ambiguity) when Lambda is omitted.
    """
    set_paper_style()
    print("\n--- Generating Topological Map: Model Geo (M vs R) ---")

    if "Geo" not in models_dict:
        print("[Error] Model Geo missing. Skipping plot.")
        return

    res = 300
    m_min, m_max = 0.5, 3.0
    r_min, r_max = 8.0, 17.0

    xx_R, yy_M = np.meshgrid(
        np.linspace(r_min, r_max, res), np.linspace(m_min, m_max, res)
    )
    grid_geo = pd.DataFrame({"Mass": yy_M.ravel(), "Radius": xx_R.ravel()})[
        CONSTANTS["ML_FEATURES"]["Geo"]
    ]

    Z_geo = evaluate_grid(models_dict["Geo"], grid_geo, res).reshape((res, res))

    # Base Model Extraction
    print("  > Evaluating Base Models for Individual Topology Maps...")
    base_probs = evaluate_base_models_grid(models_dict["Geo"], grid_geo)
    
    # Bundle topologies into a dictionary
    topologies_to_plot = {"Ensemble": Z_geo}
    
    # Identify OOD space to preserve Void mapping across individual models
    is_ood = Z_geo == 0
    tau = models_dict["Geo"]["conformal_tau"]
    
    for algo_name, probs in base_probs.items():
        Z_base = np.zeros_like(Z_geo)
        probs_2d = probs.reshape(Z_geo.shape)
        
        probs_H = 1.0 - probs_2d
        probs_Q = probs_2d
        
        set_Q = probs_Q >= tau
        set_H = probs_H >= tau
        
        Z_base[set_H & ~set_Q] = 1
        Z_base[set_Q & ~set_H] = 2
        Z_base[set_H & set_Q] = 3
        Z_base[~set_H & ~set_Q] = 3
        
        Z_base[is_ood] = 0 # Carry over Void
        topologies_to_plot[algo_name] = Z_base
    
    # Plotting Loop
    for model_name, Z_map in topologies_to_plot.items():
        fig, ax = plt.subplots(figsize=(9, 8))

        cmap_zones = ListedColormap(
            ["#E5E5E5", COLORS["H_fade"], COLORS["Q_fade"], "#CC79A7"]
        )
        bbox_props = {
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.3",
        }

        ax.imshow(
            Z_map,
            origin="lower",
            extent=(r_min, r_max, m_min, m_max),
            cmap=cmap_zones,
            vmin=0,
            vmax=3,
            alpha=0.7,
            aspect="auto",
        )

        ax.set_title(f"Model Geo: The Degeneracy of Mass & Radius ({model_name})", y=1.02, fontsize=16)
        ax.set_xlabel(r"Radius $R$ [km]", fontsize=14)
        ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]", fontsize=14)

        # Direct Annotations
        ax.text(
            9.5,
            1.2,
            "Safe Quark",
            color=COLORS["Q_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            15.5,
            2.5,
            "Safe Hadronic",
            color=COLORS["H_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            12.5,
            2.8,
            "Zone of Ambiguity",
            color="#8A2BE2",
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            8.5,
            2.8,
            "The Void (OOD)",
            color="gray",
            fontweight="bold",
            fontsize=10,
            ha="center",
            rotation=45,
            bbox=bbox_props,
        )

        # Observational Constraints
        stars = [
            {"Name": "HESS J1731", "M": 0.77, "sM": 0.17, "X": 10.40, "sX": 0.78},
            {"Name": "GW170817", "M": 1.40, "sM": 0.10, "X": 11.90, "sX": 1.40},
            {"Name": "PSR J0030", "M": 1.44, "sM": 0.15, "X": 13.02, "sX": 1.06},
            {"Name": "PSR J0740", "M": 2.08, "sM": 0.07, "X": 12.35, "sX": 0.75},
        ]
        for s in stars:
            m_val, sm_val = float(s["M"]), float(s["sM"])  # type: ignore
            x_val, sx_val = float(s["X"]), float(s["sX"])  # type: ignore
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=2 * sx_val,
                    height=2 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=2,
                )
            )
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=4 * sx_val,
                    height=4 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=1,
                    ls=":",
                )
            )
            ax.plot(x_val, m_val, marker="x", color="black")
            ax.text(
                x_val,
                m_val + sm_val + 0.05,
                str(s["Name"]),
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=bbox_props,
            )

        # GW190814 (secondary) horizontal band for mass constraint
        ax.axhspan(2.59 - 0.09, 2.59 + 0.09, color=COLORS["Constraint"], alpha=0.3)
        ax.axhline(2.59, color=COLORS["Constraint"], linestyle="-.", lw=1.5)
        ax.text(
            14.5,
            2.65,
            "GW190814",
            fontsize=11,
            color=COLORS["Constraint"],
            fontweight="bold",
        )

        legend_elements = [
            Patch(
                facecolor=COLORS["H_fade"],
                edgecolor="black",
                alpha=0.7,
                label="Safe Hadronic",
            ),
            Patch(
                facecolor=COLORS["Q_fade"], edgecolor="black", alpha=0.7, label="Safe Quark"
            ),
            Patch(facecolor="#E5E5E5", edgecolor="black", alpha=0.7, label="The Void"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="-",
                label=r"1-$\sigma$ Error Ellipse",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle=":",
                label=r"2-$\sigma$ Error Ellipse",
            ),
        ]
        legend_elements.insert(2, Patch(facecolor="#CC79A7", edgecolor="black", alpha=0.7, label="Zone of Ambiguity"))

        ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=11)

        plt.tight_layout()
        save_name = f"plots/topology_geo_M_vs_R_{model_name.replace(' ', '_')}.pdf"
        plt.savefig(save_name, metadata={"CreationDate": None})
        plt.close()
        print(f"[SUCCESS] Saved Topological Map: {save_name}.")
    print("[SUCCESS] Saved Topological Map: Model Geo (M vs R).")


def plot_A_M_vs_R(models_dict, df_valid):
    """
    Plots the Safe Classification Topology Map for Model A (M vs R).
    Uses the Bifurcated Dual-KNN Subspace Projection to safely estimate the
    bimodal Lambda coordinate for the background grid without artifacting.
    """
    set_paper_style()
    print("\n--- Generating Topological Map: Model A (M vs R) ---")

    if "A" not in models_dict:
        print("[Error] Model A missing. Skipping plot.")
        return

    res = 300
    m_min, m_max = 0.5, 3.0
    r_min, r_max = 8.0, 17.0

    xx_R, yy_M = np.meshgrid(
        np.linspace(r_min, r_max, res), np.linspace(m_min, m_max, res)
    )

    grid_df = pd.DataFrame({"Mass": yy_M.ravel(), "Radius": xx_R.ravel()})

    # Apply Bifurcated Dual-KNN to predict missing 'LogLambda'
    print("  > Running Bifurcated Dual-KNN Projection for Lambda...")
    topologies_1d = bifurcated_dual_knn_evaluation(models_dict, df_valid, grid_df, res)
    
    print("  > Evaluating Base Models for Individual Topology Maps...")
    topologies_to_plot = {k: v.reshape((res, res)) for k, v in topologies_1d.items()}
    
    # Identify OOD space
    is_ood = topologies_to_plot["Ensemble"] == 0
    
    for algo_name in topologies_to_plot:
        Z_base = topologies_to_plot[algo_name]
        Z_base[is_ood] = 0 # Carry over Void
        topologies_to_plot[algo_name] = Z_base
    
    # Plotting Loop
    for model_name, Z_map in topologies_to_plot.items():
        fig, ax = plt.subplots(figsize=(9, 8))

        cmap_zones = ListedColormap(
            ["#E5E5E5", COLORS["H_fade"], COLORS["Q_fade"], "#CC79A7"]
        )
        bbox_props = {
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.3",
        }

        ax.imshow(
            Z_map,
            origin="lower",
            extent=(r_min, r_max, m_min, m_max),
            cmap=cmap_zones,
            vmin=0,
            vmax=3,
            alpha=0.7,
            aspect="auto",
        )

        ax.set_title(
            f"Model A: Resolving Degeneracy with $\\Lambda$ (Projected) ({model_name})", y=1.02, fontsize=16
        )
        ax.set_xlabel(r"Radius $R$ [km]", fontsize=14)
        ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]", fontsize=14)

        # Direct Annotations
        ax.text(
            9.5,
            1.2,
            "Safe Quark",
            color=COLORS["Q_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            15.5,
            2.5,
            "Safe Hadronic",
            color=COLORS["H_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            12.5,
            2.8,
            "Ambiguity Resolved",
            color="#8A2BE2",
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )

        # Observational Constraints
        stars = [
            {"Name": "HESS J1731", "M": 0.77, "sM": 0.17, "X": 10.40, "sX": 0.78},
            {"Name": "GW170817", "M": 1.40, "sM": 0.10, "X": 11.90, "sX": 1.40},
            {"Name": "PSR J0030", "M": 1.44, "sM": 0.15, "X": 13.02, "sX": 1.06},
            {"Name": "PSR J0740", "M": 2.08, "sM": 0.07, "X": 12.35, "sX": 0.75},
        ]
        for s in stars:
            m_val, sm_val = float(s["M"]), float(s["sM"])  # type: ignore
            x_val, sx_val = float(s["X"]), float(s["sX"])  # type: ignore
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=2 * sx_val,
                    height=2 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=2,
                )
            )
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=4 * sx_val,
                    height=4 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=1,
                    ls=":",
                )
            )
            ax.plot(x_val, m_val, marker="x", color="black")
            ax.text(
                x_val,
                m_val + sm_val + 0.05,
                str(s["Name"]),
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=bbox_props,
            )

        # GW190814 (secondary) horizontal band for mass constraint
        ax.axhspan(2.59 - 0.09, 2.59 + 0.09, color=COLORS["Constraint"], alpha=0.3)
        ax.axhline(2.59, color=COLORS["Constraint"], linestyle="-.", lw=1.5)
        ax.text(
            14.5,
            2.65,
            "GW190814",
            fontsize=11,
            color=COLORS["Constraint"],
            fontweight="bold",
        )

        legend_elements = [
            Patch(
                facecolor=COLORS["H_fade"],
                edgecolor="black",
                alpha=0.7,
                label="Safe Hadronic",
            ),
            Patch(
                facecolor=COLORS["Q_fade"], edgecolor="black", alpha=0.7, label="Safe Quark"
            ),
            Patch(facecolor="#E5E5E5", edgecolor="black", alpha=0.7, label="The Void"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="-",
                label=r"1-$\sigma$ Error Ellipse",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle=":",
                label=r"2-$\sigma$ Error Ellipse",
            ),
        ]
        legend_elements.insert(2, Patch(facecolor="#CC79A7", edgecolor="black", alpha=0.7, label="Zone of Ambiguity"))

        ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=11)

        plt.tight_layout()
        save_name = f"plots/topology_A_M_vs_R_{model_name.replace(' ', '_')}.pdf"
        plt.savefig(save_name, metadata={"CreationDate": None})
        plt.close()
        print(f"[SUCCESS] Saved Topological Map: {save_name}.")
    print("[SUCCESS] Saved Topological Map: Model A (M vs R).")


def plot_A_Lambda_vs_M(models_dict, df_valid):
    """
    Plots the Safe Classification Topology Map for Model A (Lambda vs M).
    Uses the Bifurcated Dual-KNN Subspace Projection to safely estimate the
    missing R coordinate for the background grid.
    """
    set_paper_style()
    print("\n--- Generating Topological Map: Model A (Lambda vs M) ---")

    if "A" not in models_dict:
        print("[Error] Model A missing. Skipping plot.")
        return

    res = 300
    m_min, m_max = 0.5, 3.0
    lam_min, lam_max = 1.0, 3.5  # log10 scale (10 to ~3000)

    xx_L, yy_M = np.meshgrid(
        np.linspace(lam_min, lam_max, res), np.linspace(m_min, m_max, res)
    )

    grid_df = pd.DataFrame({"Mass": yy_M.ravel(), "LogLambda": xx_L.ravel()})

    # Apply Bifurcated Dual-KNN to predict missing 'Radius'
    print("  > Running Bifurcated Dual-KNN Projection for Radius...")
    topologies_1d = bifurcated_dual_knn_evaluation(models_dict, df_valid, grid_df, res)
    
    print("  > Evaluating Base Models for Individual Topology Maps...")
    topologies_to_plot = {k: v.reshape((res, res)) for k, v in topologies_1d.items()}
    
    # Identify OOD space
    is_ood = topologies_to_plot["Ensemble"] == 0
    
    for algo_name in topologies_to_plot:
        Z_base = topologies_to_plot[algo_name]
        Z_base[is_ood] = 0 # Carry over Void
        topologies_to_plot[algo_name] = Z_base
    
    # Plotting Loop
    for model_name, Z_map in topologies_to_plot.items():
        fig, ax = plt.subplots(figsize=(9, 8))

        cmap_zones = ListedColormap(
            ["#E5E5E5", COLORS["H_fade"], COLORS["Q_fade"], "#CC79A7"]
        )
        bbox_props = {
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.3",
        }

        ax.imshow(
            Z_map,
            origin="lower",
            extent=(lam_min, lam_max, m_min, m_max),
            cmap=cmap_zones,
            vmin=0,
            vmax=3,
            alpha=0.7,
            aspect="auto",
        )

        ax.set_title(rf"Model A: Resolving Degeneracy with $\Lambda$ ({model_name})", y=1.02, fontsize=16)
        ax.set_xlabel(r"Tidal Deformability $\log_{10}\Lambda$", fontsize=14)
        ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]", fontsize=14)

        # Direct Annotations
        ax.text(
            2.0,
            2.5,
            "Safe Hadronic",
            color=COLORS["H_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            2.8,
            1.2,
            "Safe Quark",
            color=COLORS["Q_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )

        # Observational Constraints
        gw_stars = [
            {"Name": "GW170817", "M": 1.40, "sM": 0.10, "X": np.log10(190), "sX": 0.25}
        ]

        for s in gw_stars:
            m_val, sm_val = float(s["M"]), float(s["sM"])  # type: ignore
            x_val, sx_val = float(s["X"]), float(s["sX"])  # type: ignore
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=2 * sx_val,
                    height=2 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=2,
                )
            )
            ax.add_patch(
                Ellipse(
                    xy=(x_val, m_val),
                    width=4 * sx_val,
                    height=4 * sm_val,
                    edgecolor="black",
                    fc="none",
                    lw=1,
                    ls=":",
                )
            )
            ax.plot(x_val, m_val, marker="x", color="black")
            ax.text(
                x_val,
                m_val + sm_val + 0.05,
                str(s["Name"]),
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=bbox_props,
            )

        legend_elements = [
            Patch(
                facecolor=COLORS["H_fade"],
                edgecolor="black",
                alpha=0.7,
                label="Safe Hadronic",
            ),
            Patch(
                facecolor=COLORS["Q_fade"], edgecolor="black", alpha=0.7, label="Safe Quark"
            ),
            Patch(facecolor="#E5E5E5", edgecolor="black", alpha=0.7, label="The Void"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="-",
                label=r"1-$\sigma$ Error Ellipse",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle=":",
                label=r"2-$\sigma$ Error Ellipse",
            ),
        ]
        legend_elements.insert(2, Patch(facecolor="#CC79A7", edgecolor="black", alpha=0.7, label="Zone of Ambiguity"))

        ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=11)

        plt.tight_layout()
        save_name = f"plots/topology_A_Lambda_vs_M_{model_name.replace(' ', '_')}.pdf"
        plt.savefig(save_name, metadata={"CreationDate": None})
        plt.close()
        print(f"[SUCCESS] Saved Topological Map: {save_name}.")
    print("[SUCCESS] Saved Topological Map: Model A (Lambda vs M).")


def plot_A_Lambda_vs_R(models_dict, df_valid):
    """
    Plots the Safe Classification Topology Map for Model A (Lambda vs R).
    Uses the Bifurcated Dual-KNN Subspace Projection to safely estimate the
    missing M coordinate for the background grid.
    """
    set_paper_style()
    print("\n--- Generating Topological Map: Model A (Lambda vs R) ---")

    if "A" not in models_dict:
        print("[Error] Model A missing. Skipping plot.")
        return

    res = 300
    r_min, r_max = 8.0, 17.0
    lam_min, lam_max = 1.0, 3.5  # log10 scale (10 to ~3000)

    xx_L, yy_R = np.meshgrid(
        np.linspace(lam_min, lam_max, res), np.linspace(r_min, r_max, res)
    )

    grid_df = pd.DataFrame({"Radius": yy_R.ravel(), "LogLambda": xx_L.ravel()})

    # Apply Bifurcated Dual-KNN to predict missing 'Mass'
    print("  > Running Bifurcated Dual-KNN Projection for Mass...")
    topologies_1d = bifurcated_dual_knn_evaluation(models_dict, df_valid, grid_df, res)
    
    print("  > Evaluating Base Models for Individual Topology Maps...")
    topologies_to_plot = {k: v.reshape((res, res)) for k, v in topologies_1d.items()}
    
    # Identify OOD space
    is_ood = topologies_to_plot["Ensemble"] == 0
    
    for algo_name in topologies_to_plot:
        Z_base = topologies_to_plot[algo_name]
        Z_base[is_ood] = 0 # Carry over Void
        topologies_to_plot[algo_name] = Z_base
    
    # Plotting Loop
    for model_name, Z_map in topologies_to_plot.items():
        fig, ax = plt.subplots(figsize=(9, 8))

        cmap_zones = ListedColormap(
            ["#E5E5E5", COLORS["H_fade"], COLORS["Q_fade"], "#CC79A7"]
        )
        bbox_props = {
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.3",
        }

        ax.imshow(
            Z_map,
            origin="lower",
            extent=(lam_min, lam_max, r_min, r_max),
            cmap=cmap_zones,
            vmin=0,
            vmax=3,
            alpha=0.7,
            aspect="auto",
        )

        ax.set_title(rf"Model A: Tidal Deformability vs Radius ({model_name})", y=1.02, fontsize=16)
        ax.set_xlabel(r"Tidal Deformability $\log_{10}\Lambda$", fontsize=14)
        ax.set_ylabel(r"Radius $R$ [km]", fontsize=14)

        # Direct Annotations
        ax.text(
            1.5,
            12.5,
            "Safe Quark",
            color=COLORS["Q_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )
        ax.text(
            3.0,
            14.5,
            "Safe Hadronic",
            color=COLORS["H_main"],
            fontweight="bold",
            fontsize=12,
            ha="center",
            bbox=bbox_props,
        )

        # Observational Constraints
        gw_stars = [
            {"Name": "GW170817", "R": 11.90, "sR": 1.40, "X": np.log10(190), "sX": 0.25}
        ]

        for s in gw_stars:
            r_val, sr_val = float(s["R"]), float(s["sR"])  # type: ignore
            x_val, sx_val = float(s["X"]), float(s["sX"])  # type: ignore
            ax.add_patch(
                Ellipse(
                    xy=(x_val, r_val),
                    width=2 * sx_val,
                    height=2 * sr_val,
                    edgecolor="black",
                    fc="none",
                    lw=2,
                )
            )
            ax.add_patch(
                Ellipse(
                    xy=(x_val, r_val),
                    width=4 * sx_val,
                    height=4 * sr_val,
                    edgecolor="black",
                    fc="none",
                    lw=1,
                    ls=":",
                )
            )
            ax.plot(x_val, r_val, marker="x", color="black")
            ax.text(
                x_val,
                r_val + sr_val + 0.15,
                str(s["Name"]),
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=bbox_props,
            )

        legend_elements = [
            Patch(
                facecolor=COLORS["H_fade"],
                edgecolor="black",
                alpha=0.7,
                label="Safe Hadronic",
            ),
            Patch(
                facecolor=COLORS["Q_fade"], edgecolor="black", alpha=0.7, label="Safe Quark"
            ),
            Patch(facecolor="#E5E5E5", edgecolor="black", alpha=0.7, label="The Void"),
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle="-",
                label=r"1-$\sigma$ Error Ellipse",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1,
                linestyle=":",
                label=r"2-$\sigma$ Error Ellipse",
            ),
        ]
        legend_elements.insert(2, Patch(facecolor="#CC79A7", edgecolor="black", alpha=0.7, label="Zone of Ambiguity"))

        ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=11)

        plt.tight_layout()
        save_name = f"plots/topology_A_Lambda_vs_R_{model_name.replace(' ', '_')}.pdf"
        plt.savefig(save_name, metadata={"CreationDate": None})
        plt.close()
        print(f"[SUCCESS] Saved Topological Map: {save_name}.")
    print("[SUCCESS] Saved Topological Map: Model A (Lambda vs R).")
