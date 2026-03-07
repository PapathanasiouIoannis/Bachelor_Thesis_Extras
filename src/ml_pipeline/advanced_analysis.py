# src/ml_pipeline/advanced_analysis.py

"""
  Orchestrates advanced ML diagnostics and physics correlation checks.

Refactored:
  - SAFETY PIPELINE UNPACKING: Properly extracts the ['classifier'] from the
    new models_dict safety package before passing to sklearn functions.
  - XGBOOST ALIGNMENT: Updated SHAP extraction to handle XGBoost log-odds matrices.
    - MASS CUTOFF: Applied M >= M_MIN_SAVE constraint to Learning Curves to align with training.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import learning_curve, GroupKFold

from src.const import CONSTANTS
from src.visualize.style_config import set_paper_style, COLORS

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(
        "[Info] 'shap' library not found. Skipping SHAP plots. (Run: pip install shap)"
    )


def run_advanced_analysis(df, models_dict, X_test, y_test):
    """
    Orchestrates advanced ML diagnostics and physics correlation checks.
    """
    set_paper_style()
    print("\n=======================================================")
    print("   RUNNING ADVANCED ML DIAGNOSTICS (XGBoost)")
    print("=======================================================")

    # 0. Safely unpack the classifiers from the new ML Safety dictionary
    clf_A = models_dict["A"]["classifier"] if "A" in models_dict else None

    # 1. Learning Curves (Evaluates Data Sufficiency)
    if clf_A is not None:
        plot_learning_curve(clf_A, df, "A", CONSTANTS["ML_FEATURES"]["A"])

    # 2. Noise Robustness (Evaluates Observational Uncertainty)
    if clf_A is not None:
        plot_noise_robustness(clf_A, X_test, y_test)

    # 3. Physics Correlation Maps (Hidden Variable Analysis)
    print("\n  > Generating Physics Correlation Plots (KDE Contours)...")

    # Recover the true microphysics for the Test Set points
    df_test_slice = df.loc[X_test.index].copy()

    physics_params = [
        (
            "Eps_Central",
            r"Central Energy Density $\varepsilon_c$ [MeV/fm$^3$]",
            "epsilon",
        ),
        ("CS2_Central", r"Squared Sound Speed $c_s^2(r=0)$", "cs2"),
        ("Slope14", r"Slope $dR/dM$ at $1.4 M_\odot$", "slope"),
    ]

    for model_name in ["Geo", "A"]:
        if model_name not in models_dict:
            continue

        # Unpack the classifier again for this loop
        model = models_dict[model_name]["classifier"]
        cols_needed = CONSTANTS["ML_FEATURES"][model_name]

        # Get Probabilities (Confidence of Quark classification)
        probs = model.predict_proba(X_test[cols_needed])[:, 1]

        for col, label, tag in physics_params:
            if col not in df_test_slice.columns or df_test_slice[col].isna().all():
                continue

            plot_probability_kde(
                x_data=df_test_slice[col],
                y_probs=probs,
                x_label=label,
                model_name=model_name,
                tag=tag,
            )

    # 4. SHAP Analysis (Feature Importance / Interpretability)
    if SHAP_AVAILABLE and clf_A is not None:
        plot_shap_analysis(clf_A, X_test, "A")


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================


def plot_probability_kde(x_data, y_probs, x_label, model_name, tag):
    """
    Generates a clean KDE Contour plot linking hidden Microphysics (X)
    to Macroscopic ML Confidence (Y).
    """
    # Filter NaNs
    valid_idx = ~x_data.isna() & ~np.isnan(y_probs)
    x_s = x_data[valid_idx].values
    y_s = y_probs[valid_idx]

    if len(x_s) < 100:
        return

    # Subsample for KDE performance
    if len(x_s) > 10000:
        idx = np.random.choice(len(x_s), 10000, replace=False)
        x_s, y_s = x_s[idx], y_s[idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create evaluation grid
    xmin, xmax = x_s.min(), x_s.max()
    ymin, ymax = -0.05, 1.05

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kernel = gaussian_kde(np.vstack([x_s, y_s]))
        kernel.set_bandwidth(kernel.factor * 1.2)
        Z = np.reshape(kernel(positions).T, xx.shape)

        # Ghost-Busting: Mask extremely low-density areas
        threshold = 0.05 * Z.max()
        Z[Z < threshold] = np.nan

        ax.contourf(xx, yy, Z, levels=10, cmap="Purples", alpha=0.8)

    except Exception:
        print(f"   [Warning] KDE failed for {tag}. Fallback to scatter.")
        ax.scatter(x_s, y_s, c=COLORS["Q_main"], s=5, alpha=0.3)

    # Aesthetics
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Predicted Probability $P(\text{Quark})$")
    ax.set_title(f"Model {model_name}: Hidden Physics Correlation ({tag})")

    # Decision Boundary
    ax.axhline(
        0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary"
    )

    # Physics Constraints
    if tag == "cs2":
        ax.axvline(
            1.0 / 3.0,
            color="gray",
            linestyle=":",
            linewidth=2,
            label=r"Conformal Limit",
        )
        ax.set_xlim(CONSTANTS["PLOT_CS2_LIM"])
    elif tag == "slope":
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=2, label="Zero Slope")
        ax.set_xlim(CONSTANTS["PLOT_SLOPE_LIM"])
    elif tag == "epsilon":
        ax.set_xlim(CONSTANTS["PLOT_EPS_LIM"])

    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"plots/fig_ml_corr_{model_name}_{tag}.pdf")
    plt.close()


def plot_shap_analysis(model, X_test, model_name):
    """
    Generates SHAP Beeswarm plot for XGBoost.
    """
    print(f"  > Generating SHAP Beeswarm for Model {model_name}...")

    # Unwrap CalibratedClassifierCV to get the raw XGBoost estimator
    if hasattr(model, "base_model_for_shap"):
        explainer_model = model.base_model_for_shap
    else:
        print("[Warn] Base model for SHAP not found. Skipping.")
        return

    cols = CONSTANTS["ML_FEATURES"][model_name]

    if len(X_test) > 2000:
        X_shap = X_test[cols].sample(2000, random_state=42)
    else:
        X_shap = X_test[cols]

    name_map = {"Mass": r"$M$", "Radius": r"$R$", "LogLambda": r"$\log_{10}\Lambda$"}
    X_display = X_shap.rename(columns=name_map)

    # Compute SHAP
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_shap)

    # XGBoost binary classification returns a single (N, Features) matrix of log-odds.
    # RandomForest returns a list[Class0, Class1]. We handle both just in case.
    if isinstance(shap_values, list):
        vals_to_plot = shap_values[1]
    else:
        vals_to_plot = shap_values

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        vals_to_plot, X_display, show=False, plot_type="dot", cmap="coolwarm"
    )

    plt.title(
        f"SHAP Feature Importance: Model {model_name} (XGBoost)", fontsize=14, pad=15
    )
    plt.tight_layout()
    plt.savefig(f"plots/fig_ml_shap_beeswarm_{model_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_learning_curve(model, df, model_name, features):
    """
    Generates a Learning Curve to prove we have generated enough physics data.
    """
    print(f"  > Generating Learning Curve for Model {model_name}...")

    # STRICT FILTER: Align with training constraints (M >= M_MIN_SAVE)
    df_ml = df[df["Mass"] >= CONSTANTS["M_MIN_SAVE"]].dropna(subset=features)

    X = df_ml[features]
    y = df_ml["Label"]
    groups = df_ml["Curve_ID"]

    cv_splitter = GroupKFold(n_splits=3)
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv_splitter,
        groups=groups,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(train_sizes, train_mean, "o-", color="gray", label="Training Score")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="gray",
    )

    ax.plot(
        train_sizes,
        test_mean,
        "o-",
        color=COLORS["Q_main"],
        linewidth=2,
        label="Cross-Validation Score",
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color=COLORS["Q_main"],
    )

    ax.set_title(f"Learning Curve: Model {model_name} (Canonical Masses Only)")
    ax.set_xlabel("Training Set Size (Number of Stars)")
    ax.set_ylabel("Classification Accuracy")

    ax.set_ylim(0.80, 1.01)
    ax.legend(loc="lower right")

    gap = train_mean[-1] - test_mean[-1]
    msg = "Converged" if gap < 0.05 else "Needs More Data"
    ax.text(
        0.05,
        0.5,
        msg,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(f"plots/fig_ml_learning_curve_{model_name}.pdf")
    plt.close()


def plot_noise_robustness(model, X_test, y_test):
    """
    Simulates observational measurement error (NICER/LIGO) to test model decay.
    """
    print("  > Generating Noise Robustness Stress Test (Model A)...")
    required_cols = CONSTANTS["ML_FEATURES"]["A"]
    X_base = X_test[required_cols].copy()

    noise_levels = np.linspace(0.0, 2.0, 20)  # Up to 2km of radius uncertainty
    accuracies = []

    for sigma in noise_levels:
        X_noisy = X_base.copy()
        # Inject Gaussian noise strictly into the Radius feature
        noise = np.random.normal(0, sigma, size=len(X_noisy))
        X_noisy["Radius"] += noise

        acc = model.score(X_noisy[required_cols], y_test)
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        noise_levels,
        accuracies,
        "D-",
        color=COLORS["FalseNeg"],
        linewidth=2,
        markersize=6,
    )

    ax.set_title(r"Robustness to Observational Noise")
    ax.set_xlabel(r"Injected Noise $\sigma_{Radius}$ [km]")
    ax.set_ylabel(r"Model Accuracy")

    # Reference constraints
    ax.axvline(0.5, color="gray", linestyle="--", label="Typical NICER Error (~0.5km)")
    ax.axhline(0.90, color=COLORS["Constraint"], linestyle=":", label="90% Reliability")

    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("plots/fig_ml_noise_robustness.pdf")
    plt.close()
