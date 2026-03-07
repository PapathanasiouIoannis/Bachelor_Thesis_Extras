# src/visualize/plot_pdp.py

"""
  Generates Partial Dependence Plots (PDP) for the Machine Learning pipeline.

Refactored:
  - FIXED UNPACKING BUG: Safely extracts `['classifier']` from the new ML
    Safety dictionary structure before passing to Scikit-Learn.
  - SCOPE REDUCTION: Removed the obsolete Model D. Focuses entirely on how
    Model A interprets macroscopic observables (Mass, Radius, Tidal Deformability)
    to predict the core phase state.
"""

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS


def plot_partial_dependence(models_dict, X_test_all):
    """
    Generates Partial Dependence Plots (PDP) for Model A.

    These plots visualize the marginal effect of specific features on the
    predicted probability of a star being a Quark Star (Class 1).
    """
    set_paper_style()
    print("\n--- Generating Partial Dependence Plots (PDP) ---")

    if "A" not in models_dict:
        print("[Warn] Model A not found in models_dict. Skipping PDP.")
        return

    # FIX: Safely unpack the calibrated Ensemble classifier
    model_A = models_dict["A"]["classifier"]
    features_A = CONSTANTS["ML_FEATURES"]["A"]

    # Common settings for the PDP lines
    common_params = {
        "kind": "average",
        "grid_resolution": 100,
        "percentiles": (0.00, 1.00),
        "n_jobs": -1,
    }

    # Style: Thick line colored by the Quark family palette
    line_style = {"color": COLORS["Q_main"], "linewidth": 3}

    # =======================================================
    # PLOT: OBSERVABLES (Model A)
    # =======================================================
    # Ensure we slice only the columns needed by the model
    X_A = X_test_all[features_A]

    # Map the models we want to plot
    models_to_plot = {"Ensemble": model_A}
    voting_clf = model_A.calibrated_classifiers_[0].estimator
    base_names = ["XGBoost", "LightGBM", "MLP"]
    for i, base_model in enumerate(voting_clf.estimators_):
        models_to_plot[base_names[i]] = base_model

    for model_name, estimator in models_to_plot.items():
        fig, ax = plt.subplots(figsize=(14, 4))

        display = PartialDependenceDisplay.from_estimator(
            estimator, X_A, features_A, ax=ax, line_kw=line_style, **common_params
        )
        
        # Access the underlying axes to customize labels and limits
        axes = display.axes_[0]

        fig.suptitle(
            rf"Partial Dependence: Macroscopic Observables (Model A - {model_name})", y=1.05, fontsize=16
        )

        # 1. Mass Axis
        axes[0].set_xlabel(r"Mass $M$ [$M_{\odot}$]")
        axes[0].set_xlim(max(CONSTANTS["M_MIN_SAVE"], 0.7), CONSTANTS["PLOT_M_LIM"][1])

        # 2. Radius Axis
        axes[1].set_xlabel(r"Radius $R$ [km]")
        axes[1].set_xlim(CONSTANTS["PLOT_R_LIM"])

        # 3. Tidal Axis
        axes[2].set_xlabel(r"Log Tidal $\log_{10}\Lambda$")
        axes[2].set_xlim(CONSTANTS["PLOT_L_LIM"])

        # Global Style Updates
        for sub_ax in axes:
            # Force Y-axis to 0-1 to show absolute probability impact
            sub_ax.set_ylim(0, 1.05)
            # Add Decision Boundary
            sub_ax.axhline(0.5, color="gray", linestyle=":", alpha=0.8, linewidth=1.5)
            # Grid
            sub_ax.grid(True, alpha=0.2)
            # Clean up the partial dependence ylabel default text 
            sub_ax.set_ylabel("")

        axes[0].set_ylabel(r"Probability $P(\text{Quark})$")

        plt.tight_layout()
        save_name = f"plots/fig_ml_pdp_observables_{model_name.replace(' ', '_')}.pdf"
        plt.savefig(save_name, bbox_inches="tight", metadata={"CreationDate": None})
        plt.close()

    print("[SUCCESS] Saved Partial Dependence Plots.")
