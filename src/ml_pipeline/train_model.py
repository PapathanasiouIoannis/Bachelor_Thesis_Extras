"""
Machine Learning Pipeline: Training, Tuning, and Safety Calibration.

This module replaces the sole XGBoost classifier with an algorithm-agnostic
Soft-Voting Ensemble consisting of XGBoost, LightGBM, and a Multi-Layer Perceptron (MLP).
It applies rigorous Bayesian hyperparameter optimization (Optuna) avoiding
data leakage via GroupShuffleSplit on `Curve_ID`.

The safety protocols include:
    1. Isolation Forest: Out-of-Distribution (OOD) anomaly detection.
    2. Split Conformal Prediction: Calibrated thresholding for finite-sample
       coverage guarantees to rigorously map ambiguous EOS states.
"""

import os
import joblib
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from src.const import CONSTANTS


def obj_xgb(trial, X_fit, y_fit, groups_fit):
    """Optuna objective function for XGBoost."""
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    scores = []

    for tr_idx, val_idx in gss.split(X_fit, y_fit, groups=groups_fit):
        X_tr, X_val = X_fit.iloc[tr_idx], X_fit.iloc[val_idx]
        y_tr, y_val = y_fit.iloc[tr_idx], y_fit.iloc[val_idx]

        ratio = (
            float(np.sum(y_tr == 0)) / np.sum(y_tr == 1)
            if np.sum(y_tr == 1) > 0
            else 1.0
        )
        param["scale_pos_weight"] = ratio

        clf = XGBClassifier(**param)
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_val, y_val))

    return np.mean(scores)


def obj_lgb(trial, X_fit, y_fit, groups_fit):
    """Optuna objective function for LightGBM."""
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    scores = []

    for tr_idx, val_idx in gss.split(X_fit, y_fit, groups=groups_fit):
        X_tr, X_val = X_fit.iloc[tr_idx], X_fit.iloc[val_idx]
        y_tr, y_val = y_fit.iloc[tr_idx], y_fit.iloc[val_idx]

        ratio = (
            float(np.sum(y_tr == 0)) / np.sum(y_tr == 1)
            if np.sum(y_tr == 1) > 0
            else 1.0
        )
        param["scale_pos_weight"] = ratio

        clf = LGBMClassifier(**param)
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_val, y_val))

    return np.mean(scores)


def obj_mlp(trial, X_fit, y_fit, groups_fit):
    """Optuna objective function for MLP Classifier."""
    hidden_layers_choice = trial.suggest_categorical(
        "hidden_layer_sizes", ["64_32", "128_64", "64_64_32"]
    )

    # Parse the string back to a tuple
    layer_sizes = tuple(map(int, hidden_layers_choice.split("_")))

    alpha = trial.suggest_float("alpha", 1e-4, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

    gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    scores = []

    for tr_idx, val_idx in gss.split(X_fit, y_fit, groups=groups_fit):
        X_tr, X_val = X_fit.iloc[tr_idx], X_fit.iloc[val_idx]
        y_tr, y_val = y_fit.iloc[tr_idx], y_fit.iloc[val_idx]

        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=layer_sizes,
                        alpha=alpha,
                        learning_rate_init=learning_rate_init,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_val, y_val))

    return np.mean(scores)


def train_model(df):
    """
    Executes the training and calibration pipeline.
    Identifies features to model, optimizes hyperparameters, constructs an ensemble,
    calibrates probabilities, and computes the required Split Conformal Predictive thresholds.

    Args:
        df: Input macro-physics dataframe containing EOS samples.

    Returns:
        models (dict): The packaged output mapping feature groups to trained components.
        X_test_all (pd.DataFrame): The holdout features for independent evaluation.
        y_test (pd.Series): The holdout labels.
    """
    print("\n" + "=" * 55)
    print(f"{'MACHINE LEARNING PIPELINE: SAFETY & RIGOR':^55}")
    print("=" * 55)

    if "LogLambda" not in df.columns:
        df["LogLambda"] = np.log10(df["Lambda"])

    # Strictly define analytical region and filter invalid geometries.
    df_ml = df[df["Mass"] >= CONSTANTS["M_MIN_SAVE"]].copy()
    df_clean = df_ml.dropna(subset=["Radius", "Mass", "LogLambda"]).copy()

    y = df_clean["Label"]
    groups = df_clean["Curve_ID"]

    print(
        f"[INFO] Applied Mass Filter (M >= {CONSTANTS['M_MIN_SAVE']}). Valid ML Samples: {len(df_clean)}"
    )

    # Stratify Data: Fit (64%), Calibration (16%), Test (20%) strictly by Curve_ID
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    temp_idx, test_idx = next(gss_test.split(df_clean, y, groups=groups))

    df_temp = df_clean.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]

    gss_calib = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    fit_idx, calib_idx = next(gss_calib.split(df_temp, y_temp, groups=groups_temp))

    df_fit = df_temp.iloc[fit_idx]
    y_fit = y_temp.iloc[fit_idx]
    groups_fit = groups_temp.iloc[fit_idx]

    df_calib = df_temp.iloc[calib_idx]
    y_calib = y_temp.iloc[calib_idx]

    df_test = df_clean.iloc[test_idx]
    y_test = y.iloc[test_idx]

    feature_sets = {"Geo": ["Mass", "Radius"], "A": ["Mass", "Radius", "LogLambda"]}

    models = {}
    os.makedirs("models", exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for model_name, cols in feature_sets.items():
        print(
            f"\n--- Constructing Safety Pipeline for Structure Group {model_name} ---"
        )

        X_fit = df_fit[cols]
        X_calib = df_calib[cols]
        X_test_feat = df_test[cols]

        # 1. Anomaly Detection Boundary Definition
        print("  > Fitting Isolation Forest (OOD Prior Boundary)...")
        iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        iso_forest.fit(X_fit)

        global_ratio = (
            float(np.sum(y_fit == 0)) / np.sum(y_fit == 1)
            if np.sum(y_fit == 1) > 0
            else 1.0
        )

        # 2a. XGBoost
        print("    [Tuning] XGBoost...")
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(
            lambda trial: obj_xgb(trial, X_fit, y_fit, groups_fit),
            n_trials=10,
            n_jobs=1,
        )
        best_xgb = study_xgb.best_params
        best_xgb.update(
            {
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                "scale_pos_weight": global_ratio,
            }
        )
        clf_xgb = XGBClassifier(**best_xgb)

        # 2b. LightGBM
        print("    [Tuning] LightGBM...")
        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(
            lambda trial: obj_lgb(trial, X_fit, y_fit, groups_fit),
            n_trials=10,
            n_jobs=1,
        )
        best_lgb = study_lgb.best_params
        best_lgb.update(
            {
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "scale_pos_weight": global_ratio,
            }
        )
        clf_lgb = LGBMClassifier(**best_lgb)

        # 2c. Deep MLP
        print("    [Tuning] Multi-Layer Perceptron...")
        study_mlp = optuna.create_study(direction="maximize")
        study_mlp.optimize(
            lambda trial: obj_mlp(trial, X_fit, y_fit, groups_fit), n_trials=8, n_jobs=1
        )
        best_mlp = study_mlp.best_params

        # Parse the best string choice back into a tuple
        best_layer_sizes = tuple(map(int, best_mlp["hidden_layer_sizes"].split("_")))

        clf_mlp = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=best_layer_sizes,
                        alpha=best_mlp["alpha"],
                        learning_rate_init=best_mlp["learning_rate_init"],
                        max_iter=500,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )

        # 3. Algorithm-Agnostic Soft Voting Ensembling
        print("  > Constructing Algorithm-Agnostic Soft Voting Ensemble...")
        ensemble_clf = VotingClassifier(
            estimators=[("xgb", clf_xgb), ("lgbm", clf_lgb), ("mlp", clf_mlp)],
            voting="soft",
        )
        ensemble_clf.fit(X_fit, y_fit)

        # 4. Calibration & Non-Conformity Quantification
        print("  > Executing Isotonic Calibration & Split Conformal Prediction...")
        cal_clf = CalibratedClassifierCV(ensemble_clf, method="isotonic", cv=3)
        cal_clf.fit(X_fit, y_fit)

        # Determine strict threshold inclusion boundaries on held-out calibration block
        calib_probs = cal_clf.predict_proba(X_calib)
        true_class_probs = calib_probs[np.arange(len(y_calib)), y_calib.values]
        non_conformity_scores = 1.0 - true_class_probs

        alpha = 0.05
        n_calib = len(y_calib)
        q_level = min(np.ceil((n_calib + 1) * (1 - alpha)) / n_calib, 1.0)
        q_hat = np.quantile(non_conformity_scores, q_level, method="higher")
        tau = 1.0 - q_hat

        print(f"    [Conformal Threshold] tau (95% Confidence) = {tau:.4f}")
        test_acc = cal_clf.score(X_test_feat, y_test)
        print(
            f"  >[Model {model_name}] Zero-Knowledge Test Accuracy: {test_acc * 100:.2f}%"
        )

        models[model_name] = {
            "classifier": cal_clf,
            "ood_detector": iso_forest,
            "conformal_tau": tau,
        }

    model_path = os.path.join("models", "ml_models_geo_A.pkl")
    joblib.dump(models, model_path)
    print(f"\n[SUCCESS] Serialized Algorithm-Agnostic Pipeline to {model_path}")

    all_needed_cols = list(set(feature_sets["Geo"] + feature_sets["A"]))
    X_test_all = df_test[all_needed_cols]

    return models, X_test_all, y_test
