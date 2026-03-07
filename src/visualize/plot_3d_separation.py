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


def plot_3d_separation(df):
    """
    Generates a 3D Manifold Plot (Mass-Radius-Tidal) with projected density contours.

    """
    set_paper_style()
    print("\n--- Generating 3D Manifold Plot (Intersection Shadows) ---")

    # 1. Data Prep
    if "LogLambda" not in df.columns:
        df["LogLambda"] = np.log10(df["Lambda"])

    h_data = df[df["Label"] == 0]
    q_data = df[df["Label"] == 1]

    # Load Limits from Constants
    r_lim = CONSTANTS["PLOT_R_LIM"]
    m_lim = CONSTANTS["PLOT_M_LIM"]
    l_lim = CONSTANTS["PLOT_L_LIM"]

    # ==========================================
    # DENSITY CALCULATION ENGINE
    # ==========================================
    def get_density_grid(x, y, x_lim, y_lim, grid_size=100):
        """
        Calculates 2D KDE with strict masking to prevent 'ghost' blobs.
        """
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
            # 1. Calculate KDE with TIGHT bandwidth
            # '0.5' factor prevents the blobby look seen in your screenshot
            kernel = gaussian_kde(np.vstack([x_s, y_s]))
            kernel.set_bandwidth(kernel.factor * 0.5)

            Z = np.reshape(kernel(positions).T, XX.shape)

            # 2. STRICT BOUNDING BOX MASK
            # Forces density to zero if it's outside the actual data range
            x_min, x_max = x_s.min(), x_s.max()
            y_min, y_max = y_s.min(), y_s.max()

            # Tiny buffer (5%) just to keep the edge points visible
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

    # High visibility scatter
    n_scatter = 10000
    h_sample = h_data.sample(min(n_scatter, len(h_data)), random_state=42)
    q_sample = q_data.sample(min(n_scatter, len(q_data)), random_state=42)

    ax.scatter(
        h_sample["Radius"],
        h_sample["Mass"],
        h_sample["LogLambda"],
        c=COLORS["H_main"],
        s=15,
        alpha=0.5,
        edgecolors="none",
        label="Hadronic",
    )

    ax.scatter(
        q_sample["Radius"],
        q_sample["Mass"],
        q_sample["LogLambda"],
        c=COLORS["Q_main"],
        s=15,
        alpha=0.5,
        edgecolors="none",
        label="Quark",
    )

    print("Calculating Wall Intersections (PDF)...")

    def plot_mpl_wall(ax, x_h, y_h, x_q, y_q, z_dir, z_offset, x_lim, y_lim):
        XX, YY, Z_h = get_density_grid(x_h, y_h, x_lim, y_lim)
        _, _, Z_q = get_density_grid(x_q, y_q, x_lim, y_lim)

        # High threshold to cut off faint fog
        thresh = 0.15
        Z_h[Z_h < thresh] = np.nan
        Z_q[Z_q < thresh] = np.nan

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

    # Project onto Walls
    plot_mpl_wall(
        ax,
        h_data["Radius"],
        h_data["Mass"],
        q_data["Radius"],
        q_data["Mass"],
        "z",
        l_lim[0],
        r_lim,
        m_lim,
    )

    plot_mpl_wall(
        ax,
        h_data["Radius"],
        h_data["LogLambda"],
        q_data["Radius"],
        q_data["LogLambda"],
        "y",
        m_lim[1],
        r_lim,
        l_lim,
    )

    plot_mpl_wall(
        ax,
        h_data["Mass"],
        h_data["LogLambda"],
        q_data["Mass"],
        q_data["LogLambda"],
        "x",
        r_lim[0],
        m_lim,
        l_lim,
    )

    # Standard Formatting
    ax.set_xlim(r_lim)
    ax.set_ylim(m_lim)
    ax.set_zlim(l_lim)
    ax.set_xlabel(r"Radius $R$ [km]", labelpad=12)
    ax.set_ylabel(r"Mass $M$ [$M_{\odot}$]", labelpad=12)
    ax.set_zlabel(r"Log Tidal $\log_{10}\Lambda$", labelpad=12)
    ax.set_title(r"Topological Phase Space (With Projections)", y=1.02)
    ax.view_init(elev=30, azim=135)

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
    plt.savefig("plots/fig_12_3d_manifold.pdf")
    plt.close()

    # ==========================================
    # PART B: PLOTLY (Interactive HTML)
    # ==========================================
    if PLOTLY_AVAILABLE:
        print("Generating Interactive HTML (Ghost-Busting Enabled)...")
        fig_html = go.Figure()

        # 1. Main Volume Scatter
        n_web = 10000  # Increased for better definition
        h_web = h_data.sample(min(n_web, len(h_data)), random_state=42)
        q_web = q_data.sample(min(n_web, len(q_data)), random_state=42)

        fig_html.add_trace(
            go.Scatter3d(
                x=h_web["Radius"],
                y=h_web["Mass"],
                z=h_web["LogLambda"],
                mode="markers",
                marker=dict(
                    size=3, color=COLORS["H_main"], opacity=0.8
                ),  # High opacity for sharpness
                name="Hadronic",
            )
        )
        fig_html.add_trace(
            go.Scatter3d(
                x=q_web["Radius"],
                y=q_web["Mass"],
                z=q_web["LogLambda"],
                mode="markers",
                marker=dict(size=3, color=COLORS["Q_main"], opacity=0.8),
                name="Quark",
            )
        )

        # 2. Add KDE Surfaces (With Strict Masking)
        def add_plotly_wall(fig, x_h, y_h, x_q, y_q, wall_type, x_range, y_range):
            XX, YY, Z_h = get_density_grid(x_h, y_h, x_range, y_range, grid_size=80)
            _, _, Z_q = get_density_grid(x_q, y_q, x_range, y_range, grid_size=80)

            # CUTOFF: Remove low-probability fog (The "Ghost" Fix)
            thresh = 0.10
            Z_h[Z_h < thresh] = np.nan
            Z_q[Z_q < thresh] = np.nan

            if wall_type == "floor":  # Z Constant
                z_h = np.full_like(XX, l_lim[0])
                z_q = np.full_like(XX, l_lim[0] + 0.02)
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
                y_h = np.full_like(XX, m_lim[1])
                y_q = np.full_like(XX, m_lim[1] - 0.02)
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
                x_h = np.full_like(XX, r_lim[0])
                x_q = np.full_like(XX, r_lim[0] + 0.05)
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

        # Add Surfaces
        add_plotly_wall(
            fig_html,
            h_data["Radius"],
            h_data["Mass"],
            q_data["Radius"],
            q_data["Mass"],
            "floor",
            r_lim,
            m_lim,
        )

        add_plotly_wall(
            fig_html,
            h_data["Radius"],
            h_data["LogLambda"],
            q_data["Radius"],
            q_data["LogLambda"],
            "back",
            r_lim,
            l_lim,
        )

        add_plotly_wall(
            fig_html,
            h_data["Mass"],
            h_data["LogLambda"],
            q_data["Mass"],
            q_data["LogLambda"],
            "side",
            m_lim,
            l_lim,
        )

        fig_html.update_layout(
            title="Interactive Manifold (Sharpened)",
            scene=dict(
                xaxis_title="Radius (km)",
                yaxis_title="Mass (M_sun)",
                zaxis_title="Log Tidal",
                xaxis=dict(range=r_lim),
                yaxis=dict(range=m_lim),
                zaxis=dict(range=l_lim),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        fig_html.write_html("plots/fig_12_3d_interactive.html")

    print("[Success] Saved 3D PDF and HTML.")
