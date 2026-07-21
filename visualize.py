"""
Visualization module for the slope stability LE solver.

Produces:
  - plot_cross_section(): full cross-section figure — stratigraphy (colored
    layers), slip surface (circular or composite), water table, external
    loads, reinforcement, and an FS results box.
  - plot_search_heatmap(): contour map of minimum FS over a (xc, yc) grid —
    visualizes the "Grid & Radius" / automatic search results.
  - plot_pso_convergence(): PSO best-FS-per-generation convergence curve.

Uses matplotlib only (no interactive widgets) so figures can be saved as PNG
and dropped straight into a report, or embedded later in a real GUI.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from le_solver import (interp_polyline, circle_intersections, generate_slices,
                        factor_of_safety, GAMMA_W)

# A colorblind-friendly, muted palette for soil layers (matplotlib "tab10"-ish
# but earthier). Cycles if there are more layers than colors.
LAYER_COLORS = ["#c9a66b", "#8d7048", "#a3b18a", "#6b8f71", "#7c6a8e", "#b08968"]


def _layer_polygon_points(ground_pts, layers, x_min, x_max, n=200):
    """Return, for each layer, the (x, y_top, y_bot) arrays needed to fill
    between the layer's top and bottom boundary across [x_min, x_max]."""
    xs = np.linspace(x_min, x_max, n)
    top_ys = np.array([interp_polyline(ground_pts, x) for x in xs])
    out = []
    cur_top = top_ys.copy()
    for layer in layers:
        if layer.lower_boundary is None:
            bot_ys = np.full_like(xs, cur_top.min() - max(5.0, 0.2 * (cur_top.max() - cur_top.min() + 1)))
        else:
            bot_ys = np.array([interp_polyline(layer.lower_boundary, x) for x in xs])
        out.append((xs, cur_top.copy(), bot_ys.copy(), layer.material))
        cur_top = bot_ys
    return out


def _composite_surface_xy(ground_pts, xc, yc, R, weak_boundary=None, n=400):
    """Sample (x, y) of the actual slip surface (circular, clipped to a weak
    boundary if given) between its ground-surface intersections.
    weak_boundary: None, a single polyline, or a list of polylines."""
    inter = circle_intersections(ground_pts, xc, yc, R)
    if inter is None:
        return None
    x_left, x_right = inter
    xs = np.linspace(x_left, x_right, n)

    weak_list = None
    if weak_boundary is not None:
        first_elem = weak_boundary[0]
        weak_list = [weak_boundary] if isinstance(first_elem[0], (int, float)) else list(weak_boundary)

    ys = []
    for x in xs:
        under = R ** 2 - (x - xc) ** 2
        if under <= 0:
            ys.append(np.nan)
            continue
        y_circle = yc - math.sqrt(under)
        y = y_circle
        if weak_list is not None:
            y_top = interp_polyline(ground_pts, x)
            best_y = y_circle
            for wb in weak_list:
                y_weak = min(interp_polyline(wb, x), y_top - 1e-6)
                if y_top - y_weak >= 0.05 and y_weak > best_y:
                    best_y = y_weak
            y = best_y
        ys.append(y)
    return xs, np.array(ys)


def plot_cross_section(ground_pts, layers, xc, yc, R, fs_results=None,
                        piezo_pts=None, surcharges=None, point_loads=None,
                        reinforcements=None, weak_boundary=None, n_slices=40,
                        show_slices=True, title="Mat cat va mat truot",
                        save_path=None, figsize=(11, 7)):
    """
    fs_results: dict like {"fellenius": 1.08, "bishop": 1.16, ...} to annotate.
    """
    x_min, x_max = ground_pts[0][0], ground_pts[-1][0]
    # Focus the plotted window around the slide + a margin, not the full
    # (often very long, for boundary-condition reasons) ground polyline.
    inter = circle_intersections(ground_pts, xc, yc, R)
    if inter:
        pad = max(5.0, 0.25 * (inter[1] - inter[0]))
        x_min_plot = max(x_min, inter[0] - pad)
        x_max_plot = min(x_max, inter[1] + pad)
    else:
        x_min_plot, x_max_plot = x_min, x_max

    fig, ax = plt.subplots(figsize=figsize)

    # --- stratigraphy (filled layers) ---
    bands = _layer_polygon_points(ground_pts, layers, x_min_plot, x_max_plot)
    seen_materials = {}
    for i, (xs, top_ys, bot_ys, material) in enumerate(bands):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        label = material.name if material.name not in seen_materials else None
        seen_materials[material.name] = True
        ax.fill_between(xs, bot_ys, top_ys, color=color, alpha=0.85,
                         label=label, zorder=1, linewidth=0)

    # --- ground surface line ---
    xs_ground = np.linspace(x_min_plot, x_max_plot, 200)
    ys_ground = [interp_polyline(ground_pts, x) for x in xs_ground]
    ax.plot(xs_ground, ys_ground, color="black", linewidth=1.5, zorder=5)

    # --- weak boundary (if composite) ---
    if weak_boundary is not None:
        ys_weak = [interp_polyline(weak_boundary, x) for x in xs_ground]
        ax.plot(xs_ground, ys_weak, color="#a33", linewidth=1.0, linestyle=":",
                 zorder=4, label="Ranh gioi tang yeu")

    # --- piezometric surface ---
    if piezo_pts is not None:
        ys_pz = [interp_polyline(piezo_pts, x) for x in xs_ground]
        ax.plot(xs_ground, ys_pz, color="#1f77b4", linewidth=1.3, linestyle="--",
                 zorder=6, label="Muc nuoc ngam (piezometric)")

    # --- slip surface ---
    surf = _composite_surface_xy(ground_pts, xc, yc, R, weak_boundary)
    if surf is not None:
        xs_s, ys_s = surf
        ax.plot(xs_s, ys_s, color="#d62728", linewidth=2.2, zorder=7,
                 label="Mat truot")
        # faint full circle for reference (helps show the trial-circle geometry)
        theta = np.linspace(0, 2 * math.pi, 200)
        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta),
                 color="#d62728", linewidth=0.6, linestyle=":", alpha=0.4, zorder=2)
        ax.plot([xc], [yc], marker="+", color="#d62728", markersize=9, zorder=7)

    # --- slice discretization ---
    if show_slices:
        slices = generate_slices(ground_pts, xc, yc, R, n_slices, layers, piezo_pts,
                                  surcharges, point_loads, weak_boundary)
        if slices:
            for s in slices:
                ax.plot([s.x_mid, s.x_mid], [s.y_base, s.y_top],
                        color="gray", linewidth=0.4, alpha=0.6, zorder=3)

    # --- surcharge loads (arrows along the top) ---
    if surcharges:
        for sc in surcharges:
            xs_load = np.linspace(sc.x_start, sc.x_end, 6)
            for xl in xs_load:
                y_top = interp_polyline(ground_pts, xl)
                ax.annotate("", xy=(xl, y_top), xytext=(xl, y_top + 2.2),
                            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2), zorder=8)
            y_top_mid = interp_polyline(ground_pts, 0.5 * (sc.x_start + sc.x_end))
            ax.text(0.5 * (sc.x_start + sc.x_end), y_top_mid + 2.5, f"q={sc.q:.0f} kPa",
                    color="#9467bd", ha="center", fontsize=8, zorder=8)

    # --- point loads ---
    if point_loads:
        for pl in point_loads:
            y_top = interp_polyline(ground_pts, pl.x)
            ax.annotate("", xy=(pl.x, y_top), xytext=(pl.x, y_top + 3.0),
                        arrowprops=dict(arrowstyle="->", color="#e377c2", lw=2), zorder=8)
            ax.text(pl.x, y_top + 3.2, f"P={pl.P:.0f} kN/m", color="#e377c2",
                    ha="center", fontsize=8, zorder=8)

    # --- reinforcement (drawn as short horizontal bars at the crossing point) ---
    if reinforcements:
        for r in reinforcements:
            if surf is not None:
                y_r = np.interp(r.x, xs_s, ys_s)
            else:
                y_r = interp_polyline(ground_pts, r.x) - 2.0
            ax.plot([r.x - 1.5, r.x + 1.5], [y_r, y_r], color="#2ca02c",
                     linewidth=3, zorder=9)
            ax.text(r.x, y_r - 1.0, f"T={r.T:.0f} kN/m", color="#2ca02c",
                    ha="center", fontsize=8, zorder=9)

    # --- FS results box ---
    if fs_results:
        lines = []
        name_map = {"fellenius": "Fellenius", "bishop": "Bishop",
                     "spencer": "Spencer", "mp": "Morgenstern-Price"}
        for k, v in fs_results.items():
            label = name_map.get(k, k)
            lines.append(f"{label}: FS = {v:.3f}" if v is not None else f"{label}: khong hoi tu")
        box_text = "\n".join(lines)
        ax.text(0.02, 0.97, box_text, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def plot_search_heatmap(ground_pts, layers, xc_range, yc_range, r_range,
                          n_slices=20, piezo_pts=None, method="fellenius",
                          title="Ban do FS theo tam vong tron thu (R toi uu tai moi diem)",
                          save_path=None, figsize=(9, 7)):
    """
    For each (xc, yc) grid point, scan over r_range and keep the MINIMUM valid
    FS found (mirrors how grid-and-radius search results are usually shown:
    one FS-contour per center point, representing its most critical radius).
    """
    xcs = np.linspace(*xc_range)
    ycs = np.linspace(*yc_range)
    rs = np.linspace(*r_range)

    FS = np.full((len(ycs), len(xcs)), np.nan)
    for i, yc in enumerate(ycs):
        for j, xc in enumerate(xcs):
            best = None
            for R in rs:
                fs = factor_of_safety(ground_pts, xc, yc, R, layers, method=method,
                                       n_slices=n_slices, piezo_pts=piezo_pts)
                if fs is not None and fs > 0 and fs < 10:
                    if best is None or fs < best:
                        best = fs
            if best is not None:
                FS[i, j] = best

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.contourf(xcs, ycs, FS, levels=20, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"FS ({method}, nho nhat theo R)")

    if np.all(np.isnan(FS)):
        ax.set_title(title + " (khong co diem hop le)")
    else:
        i_min, j_min = np.unravel_index(np.nanargmin(FS), FS.shape)
        ax.plot(xcs[j_min], ycs[i_min], marker="*", color="black", markersize=16,
                markeredgecolor="white", zorder=5)
        ax.text(xcs[j_min], ycs[i_min], f"  FS_min={FS[i_min, j_min]:.3f}",
                fontsize=10, va="center")
        ax.set_title(title)

    ax.set_xlabel("xc (m)")
    ax.set_ylabel("yc (m)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def plot_pso_convergence(history, title="Hoi tu PSO", save_path=None, figsize=(7, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(history)), history, color="#2c7fb8", linewidth=1.8)
    ax.set_xlabel("The he (generation)")
    ax.set_ylabel("FS tot nhat (best fitness)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig
