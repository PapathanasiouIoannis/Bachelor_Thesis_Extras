# src/visualize/plot_diagnostics.py

"""
  Generates ML diagnostic plots: ROC Curves, Reliability Diagrams, and Data
  Distribution checks.

Refactored:
  - SCOPE REDUCTION: Strictly scoped to Models Geo and A to match our decision
    to focus entirely on observable macroscopic features.
  - MASS ALIGNMENT: Adjusted the Split Violin Plot to correctly reflect the new
    M >= M_MIN_SAVE M_sun training filter, splitting the remaining valid masses into
    "Canonical", "Heavy", and "Extreme" regimes.
  - COLOR AESTHETICS: Updated to use the centralized publication color palette.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from matplotlib.lines import Line2D

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def plot_diagnostics(models_dict, X_test_all, y_test):
    """
    Generates core ML diagnostic plots for the observable models (Geo & A).
    """
    set_paper_style()
    print("\n--- Generating ML Diagnostic Plots (ROC, Calibration, Violins) ---")

    # Feature definitions
    feature_sets = {
        "Geo": CONSTANTS["ML_FEATURES"]["Geo"],
        "A": CONSTANTS["ML_FEATURES"]["A"],
    }

    # Visual map for models
    model_styles = {
        "Geo": {
            "color": COLORS["Guide"],
            "ls": "--",
            "label": "Model Geo (Mass, Radius)",
        },
        "A": {
            "color": COLORS["Q_main"],
            "ls": "-",
            "label": "Model A (+ Deformability)",
        },
    }

    # ==========================================
    # 1. ROC CURVE COMPARISON
    # ==========================================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(
        [0, 1], [0, 1], color="black", linestyle=":", alpha=0.5, label="Random Chance"
    )

    for name in ["Geo", "A"]:
        if name not in models_dict:
            continue

        model = models_dict[name]["classifier"]
        features = feature_sets[name]
        style = model_styles[name]

        # Ensemble probabilities (Calibrated)
        y_probs = model.predict_proba(X_test_all[features])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        ax_roc.plot(
            fpr,
            tpr,
            lw=3.0,
            color=style["color"],
            linestyle=style["ls"],
            label=f"{style['label']} Ensemble (AUC={roc_auc:.4f})",
            zorder=10
        )
        
        # Extract base estimators to show they are utilized
        base_probs = {"XGBoost": [], "LightGBM": [], "MLP": []}
        for cal_clf in model.calibrated_classifiers_:
            voting_clf = cal_clf.estimator
            base_probs["XGBoost"].append(voting_clf.estimators_[0].predict_proba(X_test_all[features])[:, 1])
            base_probs["LightGBM"].append(voting_clf.estimators_[1].predict_proba(X_test_all[features])[:, 1])
            base_probs["MLP"].append(voting_clf.estimators_[2].predict_proba(X_test_all[features])[:, 1])

        algos = {
            "XGBoost": (np.mean(base_probs["XGBoost"], axis=0), "--", 0.7),
            "LightGBM": (np.mean(base_probs["LightGBM"], axis=0), ":", 0.6),
            "MLP": (np.mean(base_probs["MLP"], axis=0), "-.", 0.6)
        }
        
        for algo_name, (algo_prob, ls, alpha) in algos.items():
            fpr_b, tpr_b, _ = roc_curve(y_test, algo_prob)
            auc_b = auc(fpr_b, tpr_b)
            ax_roc.plot(
                fpr_b,
                tpr_b,
                lw=1.5,
                color=style["color"],
                linestyle=ls,
                alpha=alpha,
                label=f"  - {name} Base: {algo_name}",
            )

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("plots/fig_ml_roc_combined.pdf")
    plt.close()

    # ==========================================
    # 2. CALIBRATION CURVE (RELIABILITY)
    # ==========================================
    # Verifies if Isotonic Calibration successfully aligned the Ensemble probabilities
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
    ax_cal.plot([0, 1], [0, 1], linestyle=":", color="black", label="Ideal Calibration")

    for name in ["Geo", "A"]:
        if name not in models_dict:
            continue

        model = models_dict[name]["classifier"]
        features = feature_sets[name]
        style = model_styles[name]

        # Ensemble probabilities (Calibrated)
        y_probs = model.predict_proba(X_test_all[features])[:, 1]

        # Quantile binning ensures equal samples per bin, handling skewed distributions
        prob_true, prob_pred = calibration_curve(
            y_test, y_probs, n_bins=10, strategy="quantile"
        )

        ax_cal.plot(
            prob_pred,
            prob_true,
            marker="s",
            markersize=8,
            lw=2.5,
            color=style["color"],
            linestyle=style["ls"],
            label=f"{style['label']} Ensemble",
            zorder=10
        )
        
        # Overlay uncalibrated base models to show calibration impact
        base_probs = {"XGBoost": [], "LightGBM": [], "MLP": []}
        for cal_clf in model.calibrated_classifiers_:
            voting_clf = cal_clf.estimator
            base_probs["XGBoost"].append(voting_clf.estimators_[0].predict_proba(X_test_all[features])[:, 1])
            base_probs["LightGBM"].append(voting_clf.estimators_[1].predict_proba(X_test_all[features])[:, 1])
            base_probs["MLP"].append(voting_clf.estimators_[2].predict_proba(X_test_all[features])[:, 1])

        algos = {
            "XGBoost": (np.mean(base_probs["XGBoost"], axis=0), "o", "--", 0.5),
            "LightGBM": (np.mean(base_probs["LightGBM"], axis=0), "^", ":", 0.5),
            "MLP": (np.mean(base_probs["MLP"], axis=0), "d", "-.", 0.5)
        }
        
        for algo_name, (algo_prob, marker, ls, alpha) in algos.items():
            prob_true_b, prob_pred_b = calibration_curve(
                y_test, algo_prob, n_bins=10, strategy="quantile"
            )
            ax_cal.plot(
                prob_pred_b,
                prob_true_b,
                marker=marker,
                markersize=5,
                lw=1.0,
                color=style["color"],
                linestyle=ls,
                alpha=alpha,
                label=f"  - {name} Base: {algo_name}",
            )

    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("True Positive Fraction")
    ax_cal.set_title("Reliability Diagram (Quantile Binned)")
    ax_cal.legend(loc="best")

    plt.tight_layout()
    plt.savefig("plots/fig_ml_calibration.pdf")
    plt.close()

    # ==========================================
    # 3. SPLIT VIOLIN PLOT (Radius Distribution)
    # ==========================================
    # Visualizes how the Radius distributions of Hadronic vs Quark stars
    # diverge across the filtered training regimes (M >= M_MIN_SAVE).

    plot_df = X_test_all.copy()
    plot_df["Label"] = y_test

    # Define Mass Bins specifically for the filtered > M_MIN_SAVE M_sun regime
    bins = [max(1.39, CONSTANTS["M_MIN_SAVE"]), 1.7, 2.1, 4.0]
    labels = [r"Canonical", r"Heavy ($1.7-2.1$)", r"Extreme ($>2.1$)"]
    plot_df["Mass_Bin"] = pd.cut(plot_df["Mass"], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=plot_df,
        x="Mass_Bin",
        y="Radius",
        hue="Label",
        split=True,
        inner="quartile",
        palette={0: COLORS["H_main"], 1: COLORS["Q_main"]},
        linewidth=1.2,
        alpha=0.7,
        ax=ax,
    )

    ax.set_title("Radius Distribution by Mass Regime (Test Set)")
    ax.set_xlabel(r"Mass Regime [$M_{\odot}$]")
    ax.set_ylabel(r"Radius $R$ [km]")

    # Custom Legend
    handles = [
        Line2D([0], [0], color=COLORS["H_main"], lw=4, label="Hadronic"),
        Line2D([0], [0], color=COLORS["Q_main"], lw=4, label="Quark"),
    ]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/fig_ml_violin_radius.pdf")
    plt.close()

    print("[SUCCESS] Saved Diagnostics (ROC, Calibration, Violins).")
