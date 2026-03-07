import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from src.visualize.style_config import set_paper_style, COLORS
from src.const import CONSTANTS

# Optional: Interactive plotting
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_microphysics_3d(df):
    """
    Generates a 3D Scatter plot of Microphysics parameters with KDE wall projections.


    Axes:
    - X: Central Energy Density (Eps_Central)
    - Y: Speed of Sound Squared (CS2_Central)
    - Z: Topological Slope (Slope14)
    """
    set_paper_style()
    print("\n--- Generating 3D Microphysics Manifold (Intersection Shadows) ---")

    # 1. Data Prep
    plot_df = df.dropna(subset=["Eps_Central", "CS2_Central", "Slope14"]).copy()
    h_data = plot_df[plot_df["Label"] == 0]
    q_data = plot_df[plot_df["Label"] == 1]

    # Load Limits from Constants
    eps_lim = CONSTANTS["PLOT_EPS_LIM"]
    cs2_lim = CONSTANTS["PLOT_CS2_LIM"]
    slope_lim = CONSTANTS["PLOT_SLOPE_LIM"]

    # ==========================================
    # DENSITY CALCULATION ENGINE (Ghost-Busted)
    # ==========================================
    def get_density_grid(x, y, x_lim, y_lim, grid_size=100):
        """Calculates 2D KDE with strict masking to prevent 'ghost' blobs."""
        # Subsample for speed
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x_s, y_s = x.iloc[idx], y.iloc[idx]
        else:
            x_s, y_s = x, y

        # Create meshgrid
        xx = np.linspace(x_lim[0], x_lim[1], grid_size)
        yy = np.linspace(y_lim[0], y_lim[1], grid_size)
        XX, YY = np.meshgrid(xx, yy)
        positions = np.vstack([XX.ravel(), YY.ravel()])

        try:
            # 1. TIGHT BANDWIDTH KDE
            # Prevents probability from bleeding into unphysical regions
            kernel = gaussian_kde(np.vstack([x_s, y_s]))
            kernel.set_bandwidth(kernel.factor * 0.5)

            Z = np.reshape(kernel(positions).T, XX.shape)

            # 2. STRICT BOUNDING BOX MASK
            # Forces density to zero if it's outside the actual data range
            x_min, x_max = x_s.min(), x_s.max()
            y_min, y_max = y_s.min(), y_s.max()

            # 5% buffer to keep edge points visible
            x_buff = (x_max - x_min) * 0.05
            y_buff = (y_max - y_min) * 0.05

            mask = (
                (XX < x_min - x_buff)
                | (XX > x_max + x_buff)
                | (YY < y_min - y_buff)
                | (YY > y_max + y_buff)
            )
            Z[mask] = 0.0

            # Normalize
            if Z.max() > 0:
                Z = Z / Z.max()

            return XX, YY, Z
        except Exception:
            return XX, YY, np.zeros_like(XX)

    # ==========================================
    # PART A: MATPLOTLIB (Static PDF)
    # ==========================================
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # High Visibility Scatter
    n_scatter = 3000
    h_sample = h_data.sample(min(n_scatter, len(h_data)), random_state=42)
    q_sample = q_data.sample(min(n_scatter, len(q_data)), random_state=42)

    ax.scatter(
        h_sample["Eps_Central"],
        h_sample["CS2_Central"],
        h_sample["Slope14"],
        c=COLORS["H_main"],
        s=15,
        alpha=0.5,
        edgecolors="none",
        label="Hadronic",
    )

    ax.scatter(
        q_sample["Eps_Central"],
        q_sample["CS2_Central"],
        q_sample["Slope14"],
        c=COLORS["Q_main"],
        s=15,
        alpha=0.5,
        edgecolors="none",
        label="Quark",
    )

    # --- Wall Projections ---
    print("Calculating Wall Intersections (PDF)...")

    def plot_mpl_wall(ax, x_h, y_h, x_q, y_q, z_dir, z_offset, x_lim, y_lim):
        XX, YY, Z_h = get_density_grid(x_h, y_h, x_lim, y_lim)
        _, _, Z_q = get_density_grid(x_q, y_q, x_lim, y_lim)

        # High threshold to remove faint fog
        thresh = 0.15
        Z_h[Z_h < thresh] = np.nan
        Z_q[Z_q < thresh] = np.nan

        # Plot Hadronic
        ax.contourf(
            XX,
            YY,
            Z_h,
            zdir=z_dir,
            offset=z_offset,
            levels=5,
            colors=[COLORS["H_main"]],
            alpha=0.6,
        )

        # Plot Quark
        ax.contourf(
            XX,
            YY,
            Z_q,
            zdir=z_dir,
            offset=z_offset,
            levels=5,
            colors=[COLORS["Q_main"]],
            alpha=0.6,
        )

    # Floor (z=Min Slope): Eps vs CS2
    plot_mpl_wall(
        ax,
        h_data["Eps_Central"],
        h_data["CS2_Central"],
        q_data["Eps_Central"],
        q_data["CS2_Central"],
        "z",
        slope_lim[0],
        eps_lim,
        cs2_lim,
    )

    # Back Wall (y=Max CS2): Eps vs Slope
    plot_mpl_wall(
        ax,
        h_data["Eps_Central"],
        h_data["Slope14"],
        q_data["Eps_Central"],
        q_data["Slope14"],
        "y",
        cs2_lim[1],
        eps_lim,
        slope_lim,
    )

    # Side Wall (x=Min Eps): CS2 vs Slope
    plot_mpl_wall(
        ax,
        h_data["CS2_Central"],
        h_data["Slope14"],
        q_data["CS2_Central"],
        q_data["Slope14"],
        "x",
        eps_lim[0],
        cs2_lim,
        slope_lim,
    )

    # Aesthetics
    ax.set_xlim(eps_lim)
    ax.set_ylim(cs2_lim)
    ax.set_zlim(slope_lim)
    ax.set_xlabel(r"$\varepsilon_c$ [MeV/fm$^3$]", labelpad=12)
    ax.set_ylabel(r"$c_s^2$", labelpad=12)
    ax.set_zlabel(r"Slope $dR/dM$", labelpad=12)
    ax.set_title(r"Microphysics Manifold (With Projections)", y=1.02)
    ax.view_init(elev=30, azim=-60)

    # Constraints on Floor
    ax.plot(
        [eps_lim[0], eps_lim[1]],
        [1.0, 1.0],
        [slope_lim[0], slope_lim[0]],
        color="black",
        linestyle="--",
        lw=1.5,
        zorder=10,
    )

    # Hide Panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["H_main"],
            label="Hadronic",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["Q_main"],
            label="Quark",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("plots/fig_microphysics_3d.pdf")
    plt.close()

    # ==========================================
    # PART B: PLOTLY (Interactive HTML)
    # ==========================================
    if PLOTLY_AVAILABLE:
        print("Generating Interactive Microphysics Plot (Sharpened)...")
        fig_html = go.Figure()

        n_web = 6000
        h_web = h_data.sample(min(n_web, len(h_data)), random_state=42)
        q_web = q_data.sample(min(n_web, len(q_data)), random_state=42)

        # 1. Sharper Points (Higher Opacity)
        fig_html.add_trace(
            go.Scatter3d(
                x=h_web["Eps_Central"],
                y=h_web["CS2_Central"],
                z=h_web["Slope14"],
                mode="markers",
                marker=dict(size=3, color=COLORS["H_main"], opacity=0.8),
                name="Hadronic",
            )
        )
        fig_html.add_trace(
            go.Scatter3d(
                x=q_web["Eps_Central"],
                y=q_web["CS2_Central"],
                z=q_web["Slope14"],
                mode="markers",
                marker=dict(size=3, color=COLORS["Q_main"], opacity=0.8),
                name="Quark",
            )
        )

        # 2. Ghost-Busted Walls
        def add_plotly_wall(fig, x_h, y_h, x_q, y_q, wall_type, x_range, y_range):
            XX, YY, Z_h = get_density_grid(x_h, y_h, x_range, y_range, grid_size=80)
            _, _, Z_q = get_density_grid(x_q, y_q, x_range, y_range, grid_size=80)

            # Mask low density
            thresh = 0.10
            Z_h[Z_h < thresh] = np.nan
            Z_q[Z_q < thresh] = np.nan

            if wall_type == "floor":  # Z Constant
                z_h = np.full_like(XX, slope_lim[0])
                z_q = np.full_like(XX, slope_lim[0] + 0.2)
                fig.add_trace(
                    go.Surface(
                        x=XX,
                        y=YY,
                        z=z_h,
                        surfacecolor=Z_h,
                        colorscale="Greens",
                        showscale=False,
                        opacity=0.6,
                    )
                )
                fig.add_trace(
                    go.Surface(
                        x=XX,
                        y=YY,
                        z=z_q,
                        surfacecolor=Z_q,
                        colorscale="RdPu",
                        showscale=False,
                        opacity=0.6,
                    )
                )

            elif wall_type == "back":  # Y Constant
                y_h = np.full_like(XX, cs2_lim[1])
                y_q = np.full_like(XX, cs2_lim[1] - 0.02)
                fig.add_trace(
                    go.Surface(
                        x=XX,
                        y=y_h,
                        z=YY,
                        surfacecolor=Z_h,
                        colorscale="Greens",
                        showscale=False,
                        opacity=0.6,
                    )
                )
                fig.add_trace(
                    go.Surface(
                        x=XX,
                        y=y_q,
                        z=YY,
                        surfacecolor=Z_q,
                        colorscale="RdPu",
                        showscale=False,
                        opacity=0.6,
                    )
                )

            elif wall_type == "side":  # X Constant
                x_h = np.full_like(XX, eps_lim[0])
                x_q = np.full_like(XX, eps_lim[0] + 20)
                fig.add_trace(
                    go.Surface(
                        x=x_h,
                        y=XX,
                        z=YY,
                        surfacecolor=Z_h,
                        colorscale="Greens",
                        showscale=False,
                        opacity=0.6,
                    )
                )
                fig.add_trace(
                    go.Surface(
                        x=x_q,
                        y=XX,
                        z=YY,
                        surfacecolor=Z_q,
                        colorscale="RdPu",
                        showscale=False,
                        opacity=0.6,
                    )
                )

        add_plotly_wall(
            fig_html,
            h_data["Eps_Central"],
            h_data["CS2_Central"],
            q_data["Eps_Central"],
            q_data["CS2_Central"],
            "floor",
            eps_lim,
            cs2_lim,
        )

        add_plotly_wall(
            fig_html,
            h_data["Eps_Central"],
            h_data["Slope14"],
            q_data["Eps_Central"],
            q_data["Slope14"],
            "back",
            eps_lim,
            slope_lim,
        )

        add_plotly_wall(
            fig_html,
            h_data["CS2_Central"],
            h_data["Slope14"],
            q_data["CS2_Central"],
            q_data["Slope14"],
            "side",
            cs2_lim,
            slope_lim,
        )

        fig_html.update_layout(
            title="Interactive Model D Space (Sharpened)",
            scene=dict(
                xaxis_title="Density",
                yaxis_title="Sound Speed",
                zaxis_title="Slope",
                xaxis=dict(range=eps_lim),
                yaxis=dict(range=cs2_lim),
                zaxis=dict(range=slope_lim),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        fig_html.write_html("plots/fig_microphysics_3d_interactive.html")

    print("[Success] Saved Microphysics 3D Plots.")
