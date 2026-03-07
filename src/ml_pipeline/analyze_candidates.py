"""
Inference module for real astrophysical objects.

This module applies the Conformal Prediction and Out-Of-Distribution (OOD)
frameworks to observable astrophysical constraints (e.g., GW170817).
It employs Monte Carlo sampling over observational posteriors to deduce the
aggregate physical nature (Hadronic, Quark, Ambiguous, or OOD) of candidates.
"""

import numpy as np
import pandas as pd
from src.const import CONSTANTS


def analyze_candidates(models):
    """
    Computes statistical inferences on a set of known astrophysical candidates.

    Args:
        models (dict): The loaded, trained, and calibrated model architectures.
    """
    print("\n" + "=" * 115)
    print(f"{'INFERENCE ON ASTROPHYSICAL CANDIDATES':^115}")
    print("=" * 115)

    if "A" not in models or "Geo" not in models:
        print("[Error] Required models ('A' and 'Geo') not found for inference.")
        return

    features_A = CONSTANTS["ML_FEATURES"]["A"]
    features_Geo = CONSTANTS["ML_FEATURES"]["Geo"]

    candidates = [
        {
            "Name": "GW170817",
            "M": 1.40,
            "sM": 0.10,
            "R": 11.90,
            "sR": 1.40,
            "L": 190,
            "sL": 120,
        },
        {
            "Name": "PSR J0740+66",
            "M": 2.08,
            "sM": 0.07,
            "R": 12.35,
            "sR": 0.75,
            "L": 0,
            "sL": 0,
        },
        {
            "Name": "PSR J0030+04",
            "M": 1.44,
            "sM": 0.15,
            "R": 13.02,
            "sR": 1.06,
            "L": 0,
            "sL": 0,
        },
        {
            "Name": "HESS J1731",
            "M": 0.77,
            "sM": 0.17,
            "R": 10.40,
            "sR": 0.78,
            "L": 0,
            "sL": 0,
        },
        {
            "Name": "GW190814(sec)",
            "M": 2.59,
            "sM": 0.09,
            "R": 12.00,
            "sR": 3.00,
            "L": 0,
            "sL": 0,
        },
    ]

    print(
        f"{'Candidate':<16} | {'Model':<5} | {'% OOD':<7} | {'% Hadronic':<11} | {'% Quark':<9} | {'% Ambiguous':<11} | {'Final Verdict'}"
    )
    print("-" * 115)

    for star in candidates:
        n_mc = 10000
        raw_m = np.random.normal(float(star["M"]), float(star["sM"]), n_mc)  # type: ignore
        raw_r = np.random.normal(float(star["R"]), float(star["sR"]), n_mc)  # type: ignore
        raw_l = np.random.normal(float(star["L"]), float(star["sL"]), n_mc)  # type: ignore

        valid_mask = (raw_r > 8.0) & (raw_m > 0.1)
        has_tidal = float(star["L"]) > 0

        if has_tidal:
            valid_mask = valid_mask & (raw_l >= 1.0)

        m_s, r_s, l_s = raw_m[valid_mask], raw_r[valid_mask], raw_l[valid_mask]
        if len(m_s) == 0:
            continue

        if has_tidal:
            X_mc = pd.DataFrame(
                {"Mass": m_s, "Radius": r_s, "LogLambda": np.log10(l_s)}
            )[features_A]
            model_pkg = models["A"]
            m_name = "A"
        else:
            X_mc = pd.DataFrame({"Mass": m_s, "Radius": r_s})[features_Geo]
            model_pkg = models["Geo"]
            m_name = "Geo"

        clf = model_pkg["classifier"]
        iso = model_pkg["ood_detector"]
        tau = model_pkg["conformal_tau"]

        ood_preds = iso.predict(X_mc)
        is_ood = ood_preds == -1
        pct_ood = np.mean(is_ood) * 100.0

        inliers_X = X_mc[~is_ood]

        if len(inliers_X) > 0:
            probs = clf.predict_proba(inliers_X)[:, 1]

            set_quark = probs >= tau
            set_hadronic = (1.0 - probs) >= tau

            only_quark = set_quark & ~set_hadronic
            only_hadronic = set_hadronic & ~set_quark
            ambiguous = set_quark & set_hadronic
            empty = ~set_quark & ~set_hadronic

            pct_q = np.mean(only_quark) * 100.0
            pct_h = np.mean(only_hadronic) * 100.0
            pct_a = np.mean(ambiguous | empty) * 100.0
        else:
            pct_q = pct_h = pct_a = 0.0

        if pct_ood > 50.0:
            verdict = "OOD"
        elif pct_a > 30.0:
            verdict = "AMBIGUOUS"
        elif pct_q > pct_h:
            verdict = "QUARK"
        else:
            verdict = "HADRONIC"

        print(
            f"{star['Name']:<16} | {m_name:<5} | {pct_ood:>6.1f}% | {pct_h:>10.1f}% | {pct_q:>8.1f}% | {pct_a:>10.1f}% | {verdict}"
        )

    print("=" * 115 + "\n")
