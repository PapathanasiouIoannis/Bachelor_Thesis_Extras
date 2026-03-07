"""
Interactive 3D Phase Space Topology Mapping.

This module resolves the highly dimensional EOS inference problem by mapping
macro-physical candidates into a 3D phase space (Radius, Mass, LogLambda).
It strictly applies Split Conformal Prediction thresholds and Isolation Forest
boundaries to partition the volume into physical topologies.
Additionally, it projects precise 1-sigma observational error ellipsoids via
affine parametric transformations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualize.eval_utils import evaluate_grid


def generate_parametric_ellipsoid(center, cov_matrix, scale=1.538, resolution=50):
    """
    Constructs a 3D parametric surface grid for an error ellipsoid mathematically
    derived from the observation's complete covariance structure.

    Args:
        center: Array [R_0, M_0, L_0] representing the mean observation.
        cov_matrix: 3x3 covariance matrix.
        scale: The chi-distribution scaling factor corresponding to the 1-sigma
               contour in 3D (approx 1.538).
        resolution: Grid resolution for angles theta and phi.

    Returns:
        X, Y, Z (numpy.ndarray): Paratmetric coordinate grids forming the surface.
    """
    # 1. Eigendecomposition to determine orientation and scales
    evals, evecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues/vectors to maintain orientational consistency
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # 2. Spherical prior grid
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    THETA, PHI = np.meshgrid(theta, phi)

    x_s = np.sin(THETA) * np.cos(PHI)
    y_s = np.sin(THETA) * np.sin(PHI)
    z_s = np.cos(THETA)

    # 3. Affine Transformation
    # Scale by eigenvalues and scale factor, then apply the rotation (eigenvector matrix),
    # and finally shift to the central candidate coordinate.
    radii = np.sqrt(np.maximum(evals, 0)) * scale

    X = np.zeros_like(x_s)
    Y = np.zeros_like(y_s)
    Z = np.zeros_like(z_s)

    for i in range(resolution):
        for j in range(resolution):
            point_s = np.array([x_s[i, j], y_s[i, j], z_s[i, j]])
            scaled_point = radii * point_s
            rotated_point = np.dot(evecs, scaled_point)

            X[i, j] = center[0] + rotated_point[0]
            Y[i, j] = center[1] + rotated_point[1]
            Z[i, j] = center[2] + rotated_point[2]

    return X, Y, Z


def plot_3d_topology():
    """
    Generates and saves the Interactive 3D Plotly visual artifact mapping
    the full topology and precise structural ellipsoids.
    """
    print("\n" + "=" * 60)
    print(f"{'RENDERING 3D INTERACTIVE TOPOLOGY':^60}")
    print("=" * 60)

    model_path = os.path.join("models", "ml_models_geo_A.pkl")
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Run training pipeline first.")
        return

    models = joblib.load(model_path)
    model_pkg = models["A"]  # 3D Feature Space: Mass, Radius, LogLambda

    # 1. Generate the Dense 3D Spatial Grid
    # We sample a block that encompasses the typical EOS phase space
    print("> Constructing 3D Feature Manifold...")
    r_vals = np.linspace(8.0, 16.0, 30)
    m_vals = np.linspace(0.5, 3.0, 30)
    l_vals = np.linspace(0.0, 4.0, 30)

    R_grid, M_grid, L_grid = np.meshgrid(r_vals, m_vals, l_vals)

    flat_R = R_grid.flatten()
    flat_M = M_grid.flatten()
    flat_L = L_grid.flatten()

    df_grid = pd.DataFrame({"Mass": flat_M, "Radius": flat_R, "LogLambda": flat_L})

    # 2. Evaluate Topology (Void, Hadronic, Quark, Ambiguous)
    # Reutilize eval_utils.evaluate_grid logic for the flat dimension
    print("> Evaluating Grid through Split Conformal Boundaries...")
    Z_flat_ensemble = evaluate_grid(model_pkg, df_grid, resolution=None)
    
    # Evaluate Base Models for Hard-Cut Topology (P(Q) >= 0.5)
    print("> Evaluating Base Models...")
    from src.visualize.eval_utils import evaluate_base_models_grid
    base_probs = evaluate_base_models_grid(model_pkg, df_grid)
    
    is_ood = Z_flat_ensemble == 0
    tau = model_pkg["conformal_tau"]
    
    Z_base_models = {}
    for name, probs in base_probs.items():
        Z_base = np.zeros_like(Z_flat_ensemble)
        
        probs_H = 1.0 - probs
        probs_Q = probs
        
        set_Q = probs_Q >= tau
        set_H = probs_H >= tau
        
        Z_base[set_H & ~set_Q] = 1
        Z_base[set_Q & ~set_H] = 2
        Z_base[set_H & set_Q] = 3
        Z_base[~set_H & ~set_Q] = 3
        
        Z_base[is_ood] = 0
        Z_base_models[name] = Z_base

    # 3. Initialize Interactive Figure
    fig = go.Figure()

    colors = {
        1: "#1f77b4",  # Safe Hadronic (Blue)
        2: "#d62728",  # Safe Quark (Red)
        3: "#9467bd",  # Ambiguous (Purple)
    } 

    names = {1: "Strictly Hadronic", 2: "Strictly Quark", 3: "Zone of Ambiguity"}
    
    # Function to add traces for a specific model's Z_grid
    def add_topology_traces(Z_flat, model_label, visible=True):
        traces = []
        for zone, color in colors.items():
            mask = Z_flat == zone
            if not np.any(mask):
                # Add an empty scatter to keep trace count consistent for the dropdown
                t = go.Scatter3d(x=[], y=[], z=[], showlegend=False, visible=visible)
                traces.append(fig.add_trace(t).data[-1])
                continue

            t = go.Scatter3d(
                x=flat_R[mask],
                y=flat_M[mask],
                z=flat_L[mask],
                mode="markers",
                marker={"size": 4, "color": color, "opacity": 0.15},
                name=f"{names[zone]} ({model_label})",
                showlegend=True,
                visible=visible
            )
            traces.append(fig.add_trace(t).data[-1])
        return len(traces)

    print("> Mapping Topologies into Figure...")
    
    # Keep track of trace indices for the menu
    trace_counts = {}
    
    # 1. Ensemble (Visible by Default)
    trace_counts["Ensemble"] = add_topology_traces(Z_flat_ensemble, "Ensemble", visible=True)
    
    # 2. Base Models (Hidden by Default)
    for model_name, Z_base in Z_base_models.items():
        trace_counts[model_name] = add_topology_traces(Z_base, model_name, visible=False)

    # 4. Inject 1-Sigma Precise Observational Ellipsoids
    print("> Injecting Parametric Observational Covariances...")
    gw170817_center = np.array([11.9, 1.40, np.log10(190)])  # [R, M, LogLambda]

    sR, sM, sL = 1.4, 0.1, 0.2
    cov_gw170817 = np.array(
        [
            [sR**2, 0.05 * sR * sM, 0.8 * sR * sL],
            [0.05 * sR * sM, sM**2, 0.1 * sM * sL],
            [0.8 * sR * sL, 0.1 * sM * sL, sL**2],
        ]
    )

    X_surf, Y_surf, Z_surf = generate_parametric_ellipsoid(
        gw170817_center, cov_gw170817
    )

    fig.add_trace(
        go.Surface(
            x=X_surf,
            y=Y_surf,
            z=Z_surf,
            colorscale="Greens",
            opacity=0.6,
            showscale=False,
            name="GW170817 (1-Sigma)",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[gw170817_center[0]],
            y=[gw170817_center[1]],
            z=[gw170817_center[2]],
            mode="markers",
            marker={"size": 8, "color": "green", "symbol": "cross"},
            name="GW170817 Mean",
            showlegend=False,
        )
    )
    
    # Always keep the ellipsoids visible across all states
    num_ellipsoid_traces = 2 

    # 5. Create the Interactive Dropdown Menu
    buttons = []
    
    # We have Ensemble + 3 Base Models = 4 Model sets * 3 Topology Traces = 12 Traces
    # Plus 2 traces for the Ellipsoid = 14 Total Traces
    total_model_traces = int(sum(trace_counts.values()))
    
    current_idx = 0
    for model_name in ["Ensemble", "XGBoost", "LightGBM", "MLP"]:
        visibility = [False] * total_model_traces + [True] * num_ellipsoid_traces
        
        # Turn on the traces for the current model
        count = trace_counts[model_name]
        for i in range(current_idx, current_idx + count):
            visibility[i] = True
        
        current_idx += count
            
        buttons.append(
            dict(
                label=model_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Interactive Conformal Phase Space Topology ({model_name})"}]
            )
        )

    # 6. Figure Aesthetics Calibration
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.01,
            y=0.99,
            showactive=True,
            buttons=buttons
        )],
        scene={
            "xaxis_title": "Radius (km)",
            "yaxis_title": "Mass (M<sub>⊙</sub>)",
            "zaxis_title": "Log<sub>10</sub>(Λ)",
            "camera": {"eye": {"x": 1.8, "y": 1.5, "z": 0.8}},
        },
        title="Interactive Conformal Phase Space Topology (Ensemble)",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "3d_interactive_topology.html")
    fig.write_html(out_path)
    print(
        f"[SUCCESS] Topologies mathematically mapped. Artifact exported to {out_path}."
    )


if __name__ == "__main__":
    plot_3d_topology()
