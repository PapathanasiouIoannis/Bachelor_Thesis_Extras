"""
Utility functions for evaluating ML model topology zones.

This module provides methods to evaluate macroscopic and multi-dimensional
features through the loaded models. It applies out-of-distribution (OOD)
detection and conformal prediction probability thresholding to map features
into distinct physical topologies.
"""

import numpy as np


def evaluate_grid(model_pkg, features_df, resolution):
    """
    Evaluates a feature grid using the provided model package to assign topological zones.

    Applies the Isolation Forest for Out-Of-Distribution (OOD) detection and Split
    Conformal Prediction to threshold probabilities into four distinct domains.

    Topologies:
        0: OOD / Unphysical
        1: Strictly Hadronic
        2: Strictly Quark
        3: Ambiguous (Conformal Overlap)

    Args:
        model_pkg (dict): A dictionary containing 'classifier', 'ood_detector', and 'conformal_tau'.
        features_df (pd.DataFrame): The feature set to evaluate.
        resolution (int or None): The resolution parameter (currently unused, maintained for API consistency).

    Returns:
        np.ndarray: A 1D numpy array of topological zone identifiers corresponding to the rows in features_df.
    """
    clf = model_pkg["classifier"]
    iso = model_pkg["ood_detector"]
    tau = model_pkg["conformal_tau"]

    ood_preds = iso.predict(features_df)
    is_ood = ood_preds == -1

    probs_Q = clf.predict_proba(features_df)[:, 1]
    probs_H = 1.0 - probs_Q

    set_Q = probs_Q >= tau
    set_H = probs_H >= tau

    Z = np.zeros(len(features_df))
    Z[set_H & ~set_Q] = 1
    Z[set_Q & ~set_H] = 2
    Z[set_H & set_Q] = 3
    Z[~set_H & ~set_Q] = 3

    # OOD supersedes probabilistic topology mappings
    Z[is_ood] = 0

    return Z


def evaluate_base_models_grid(model_pkg, features_df):
    """
    Extracts the uncalibrated probability predictions for XGBoost, LightGBM, and MLP
    separately across the given coordinate grid.

    Args:
        model_pkg (dict): A dictionary containing 'classifier' (a CalibratedClassifierCV
                          wrapping a VotingClassifier of the 3 base models).
        features_df (pd.DataFrame): The feature set to evaluate.

    Returns:
        dict: A dictionary mapping model names ("XGBoost", "LightGBM", "MLP") to their
              1D numpy arrays of P(Quark) predictions for the grid.
    """
    model = model_pkg["classifier"]
    
    base_probs = {"XGBoost": [], "LightGBM": [], "MLP": []}
    for cal_clf in model.calibrated_classifiers_:
        voting_clf = cal_clf.estimator
        base_probs["XGBoost"].append(voting_clf.estimators_[0].predict_proba(features_df)[:, 1])
        base_probs["LightGBM"].append(voting_clf.estimators_[1].predict_proba(features_df)[:, 1])
        base_probs["MLP"].append(voting_clf.estimators_[2].predict_proba(features_df)[:, 1])

    return {
        "XGBoost": np.mean(base_probs["XGBoost"], axis=0),
        "LightGBM": np.mean(base_probs["LightGBM"], axis=0),
        "MLP": np.mean(base_probs["MLP"], axis=0),
    }


def bifurcated_dual_knn_evaluation(models_dict, df_valid, grid_df, resolution):
    """
    Evaluates topological probability subspaces via the Dual-KNN strategy.

    Targets the missing physical column among ['Mass', 'Radius', 'LogLambda'] by
    training independent K-Nearest Neighbors regressors for Hadronic and Quark classes.

    Args:
        models_dict (dict): The dictionary of trained model packages.
        df_valid (pd.DataFrame): Training dataframe containing valid EOS samples.
        grid_df (pd.DataFrame): Grid coordinates for the known dimensions.
        resolution (int): Resolution of the 2D grid parameter.

    Returns:
        np.ndarray: A 1D array of the combined topological map.

    Raises:
        ValueError: If all required columns are present and no target exists.
    """
    from sklearn.neighbors import KNeighborsRegressor

    cols = ["Mass", "Radius", "LogLambda"]
    known_cols = list(grid_df.columns)

    target_cols = [c for c in cols if c not in known_cols]
    if not target_cols:
        raise ValueError(
            f"Grid contains all required columns {cols}. No missing target to predict."
        )
    target_col = target_cols[0]

    df_H = df_valid[df_valid["Label"] == 0]
    df_Q = df_valid[df_valid["Label"] == 1]

    knn_H = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    knn_H.fit(df_H[known_cols], df_H[target_col])

    knn_Q = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    knn_Q.fit(df_Q[known_cols], df_Q[target_col])

    pred_target_H = knn_H.predict(grid_df)
    pred_target_Q = knn_Q.predict(grid_df)

    grid_full_H = grid_df.copy()
    grid_full_H[target_col] = pred_target_H
    grid_full_H = grid_full_H[cols]

    grid_full_Q = grid_df.copy()
    grid_full_Q[target_col] = pred_target_Q
    grid_full_Q = grid_full_Q[cols]

    Z_H = evaluate_grid(models_dict["A"], grid_full_H, resolution)
    Z_Q = evaluate_grid(models_dict["A"], grid_full_Q, resolution)

    Z_combined = np.zeros_like(Z_H)

    H_possible = (Z_H == 1) | (Z_H == 3) | (Z_Q == 1) | (Z_Q == 3)
    Q_possible = (Z_Q == 2) | (Z_Q == 3) | (Z_H == 2) | (Z_H == 3)

    Z_combined[(H_possible) & (~Q_possible)] = 1
    Z_combined[(Q_possible) & (~H_possible)] = 2
    Z_combined[(H_possible) & (Q_possible)] = 3

    topologies_to_plot = {"Ensemble": Z_combined.copy()}
    
    base_probs_H = evaluate_base_models_grid(models_dict["A"], grid_full_H)
    base_probs_Q = evaluate_base_models_grid(models_dict["A"], grid_full_Q)
    tau = models_dict["A"]["conformal_tau"]
    
    for algo_name in base_probs_H:
        probs_H_Q = base_probs_H[algo_name] 
        probs_H_H = 1.0 - probs_H_Q
        set_Q_on_H = probs_H_Q >= tau
        set_H_on_H = probs_H_H >= tau
        Z_base_H = np.zeros_like(Z_H)
        Z_base_H[set_H_on_H & ~set_Q_on_H] = 1
        Z_base_H[set_Q_on_H & ~set_H_on_H] = 2
        Z_base_H[set_H_on_H & set_Q_on_H] = 3
        Z_base_H[~set_H_on_H & ~set_Q_on_H] = 3

        probs_Q_Q = base_probs_Q[algo_name]
        probs_Q_H = 1.0 - probs_Q_Q
        set_Q_on_Q = probs_Q_Q >= tau
        set_H_on_Q = probs_Q_H >= tau
        Z_base_Q = np.zeros_like(Z_Q)
        Z_base_Q[set_H_on_Q & ~set_Q_on_Q] = 1
        Z_base_Q[set_Q_on_Q & ~set_H_on_Q] = 2
        Z_base_Q[set_H_on_Q & set_Q_on_Q] = 3
        Z_base_Q[~set_H_on_Q & ~set_Q_on_Q] = 3

        Z_combined_base = np.zeros_like(Z_H)
        H_possible_b = (Z_base_H == 1) | (Z_base_H == 3) | (Z_base_Q == 1) | (Z_base_Q == 3)
        Q_possible_b = (Z_base_Q == 2) | (Z_base_Q == 3) | (Z_base_H == 2) | (Z_base_H == 3)

        Z_combined_base[(H_possible_b) & (~Q_possible_b)] = 1
        Z_combined_base[(Q_possible_b) & (~H_possible_b)] = 2
        Z_combined_base[(H_possible_b) & (Q_possible_b)] = 3
        
        topologies_to_plot[algo_name] = Z_combined_base

    return topologies_to_plot
