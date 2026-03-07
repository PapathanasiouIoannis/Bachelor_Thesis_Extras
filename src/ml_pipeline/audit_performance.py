# src/ml_pipeline/audit_performance.py

"""
  Stress-tests the Machine Learning models across specific mass regimes.
  Identifies exactly where adding Tidal Deformability (Model A) provides
  the most value over pure Geometric observations (Model Geo).

Refactored:
  - FIXED BINNING DIMENSIONS: Added the missing 1.1 M_sun edge to the bins array
    so it correctly evaluates the 4 distinct astrophysical regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS


def run_performance_audit(models_dict, X_test, y_test):
    """
    Runs a detailed mass-dependent performance audit on the trained ML pipeline.
    """
    set_paper_style()
    print("\n=========================================================")
    print("   PERFORMANCE AUDIT: MODEL STRESS TESTING")
    print("=========================================================")

    target_models = ["Geo", "A"]
    valid_models = [m for m in target_models if m in models_dict]

    if len(valid_models) == 0:
        print("[Error] No valid models found for audit.")
        return

    # Define strict, physically meaningful Mass Bins
    # FIXED: Restored 1.1 to properly create 4 bins for the 4 labels.
    bins = [CONSTANTS["M_MIN_SAVE"], 1.1, 1.6, 2.1, 4.0]
    bin_labels = [
        f"Crossover Zone\\n({CONSTANTS['M_MIN_SAVE']} - 1.1 $M_{{\\odot}}$)",
        r"Canonical\n(1.1 - 1.6 $M_{\odot}$)",
        r"Heavy\n(1.6 - 2.1 $M_{\odot}$)",
        r"Extreme\n(> 2.1 $M_{\odot}$)",
    ]

    results = {name: [] for name in valid_models}
    counts = []

    # =======================================================
    # TEST 1: MASS-DEPENDENT ACCURACY
    # =======================================================
    print("\n  > [Audit 1] Calculating Accuracy per Mass Regime...")

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]

        # Filter Test Set for this specific Mass Bin
        mask = (X_test["Mass"] >= low) & (X_test["Mass"] < high)
        X_sub = X_test[mask]
        y_sub = y_test[mask]

        counts.append(len(y_sub))

        if len(y_sub) == 0:
            for name in valid_models:
                results[name].append(np.nan)
            continue

        for name in valid_models:
            # Unpack the classifier from the safety dictionary
            model = models_dict[name]["classifier"]
            cols = CONSTANTS["ML_FEATURES"][name]

            # Score this specific slice
            acc = model.score(X_sub[cols], y_sub)
            results[name].append(acc)

    # --- PLOT 1: ACCURACY VS MASS ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    plt.subplots_adjust(hspace=0.05)

    x_pos = np.arange(len(bin_labels))

    # Accuracy Lines
    model_styles = {
        "Geo": {"color": COLORS["Guide"], "marker": "X", "label": "Model Geo (M, R)"},
        "A": {
            "color": COLORS["Q_main"],
            "marker": "o",
            "label": "Model A (+ Deformability)",
        },
    }

    for name in valid_models:
        style = model_styles[name]
        ax1.plot(
            x_pos,
            results[name],
            marker=style["marker"],
            color=style["color"],
            linewidth=3.0,
            markersize=10,
            label=style["label"],
            alpha=0.9,
        )

    ax1.set_ylabel("Classification Accuracy")
    ax1.set_title("The 'Zone of Ambiguity': Accuracy vs. Mass Regime")
    ax1.set_ylim(0.45, 1.05)

    # Thresholds
    ax1.axhline(
        0.90,
        color=COLORS["Constraint"],
        linestyle=":",
        lw=1.5,
        label="90% Reliability Target",
    )
    ax1.axhline(
        0.50,
        color="black",
        linestyle="--",
        lw=1.0,
        alpha=0.5,
        label="Random Guessing (50%)",
    )

    ax1.legend(loc="lower right", framealpha=0.95)

    # Sample Count Bar Chart (Provides statistical context)
    ax2.bar(
        x_pos, counts, color=COLORS["H_fade"], alpha=0.5, edgecolor=COLORS["H_main"]
    )
    ax2.set_ylabel("Test Samples")
    ax2.set_xlabel("Astrophysical Mass Regime")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels)

    plt.tight_layout()
    plt.savefig("plots/fig_ml_accuracy_vs_mass.pdf")
    plt.close()
    print("  > Saved Accuracy vs Mass chart.")

    # =======================================================
    # TEST 2: CONFIDENCE OF ERRORS (Calibration Check)
    # =======================================================
    print("  > [Audit 2] Analyzing Error Confidence (Model A)...")

    if "A" in models_dict:
        # Unpack the classifier from the safety dictionary
        model = models_dict["A"]["classifier"]
        cols = CONSTANTS["ML_FEATURES"]["A"]

        # Get raw probabilities
        probs = model.predict_proba(X_test[cols])[:, 1]
        preds = (probs > 0.5).astype(int)

        # Isolate instances where the model guessed wrong
        error_mask = preds != y_test
        error_probs = probs[error_mask]

        if len(error_probs) > 0:
            # Calculate "Confidence" (Distance from 0.5 uncertainty)
            # 0.0 = Totally Unsure. 1.0 = Confidently Wrong.
            error_conf = 2 * np.abs(error_probs - 0.5)

            print(f"    Total Errors in Model A: {len(error_probs)} / {len(y_test)}")
            print(
                f"    Mean Confidence of Errors: {np.mean(error_conf):.2f} (Ideal is ~0.0)"
            )

            # Plot Histogram of Predicted Probabilities for WRONG answers
            fig, ax = plt.subplots(figsize=(8, 5))

            sns.histplot(
                error_probs,
                bins=20,
                color=COLORS["FalseNeg"],
                alpha=0.6,
                kde=True,
                ax=ax,
                edgecolor="black",
            )

            ax.axvline(
                0.5,
                color="black",
                linestyle="--",
                lw=2,
                label="Maximum Uncertainty (Ideal for Errors)",
            )

            ax.set_title(
                "Calibration Check: Probability Assigned to Incorrect Predictions"
            )
            ax.set_xlabel(r"Predicted Probability $P(\text{Quark})$")
            ax.set_ylabel("Number of Misclassifications")
            ax.set_xlim(0, 1)
            ax.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig("plots/fig_ml_error_confidence.pdf")
            plt.close()
            print("  > Saved Error Confidence chart.")

            if np.mean(error_conf) < 0.4:
                print(
                    "    [VERDICT] ROBUST. The model naturally expresses uncertainty when it is wrong."
                )
            else:
                print(
                    "    [VERDICT] WARNING. The model is making highly confident incorrect predictions."
                )
        else:
            print("    [VERDICT] FLAWLESS. Zero errors found on the test set.")

    print("=========================================================\n")
