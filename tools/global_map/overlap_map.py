# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#!/usr/bin/env python3
"""
Plot a tile map colored by overlap (#distinct scenes) per tile.

Directory layout expected:
  save_dir/map_location/<x0>_<y0>/*.npy
where each .npy is named "<scene_token>_<timestamp>.npy"
"""

import os
import os.path as osp
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def _iter_tile_dirs(root: str):
    """Yield (tile_dir_path, x0, y0) for subdirs named '<x0>_<y0>'."""
    if not osp.isdir(root):
        return
    for name in os.listdir(root):
        p = osp.join(root, name)
        if not osp.isdir(p):
            continue
        if "_" not in name:
            continue
        try:
            x_str, y_str = name.split("_", 1)
            x0 = float(x_str)
            y0 = float(y_str)
        except Exception:
            continue
        yield p, x0, y0


def _parse_scene_from_filename(fname: str) -> str:
    """Extract scene_token from '<scene_token>_<timestamp>.npy'."""
    if not fname.endswith(".npy"):
        return ""
    base = osp.splitext(osp.basename(fname))[0]
    if "_" not in base:
        return ""
    scene_token, _ = base.rsplit("_", 1)
    return scene_token


def compute_tile_overlap_counts(
    save_dir: str,
    map_location: str,
) -> Tuple[Dict[Tuple[float, float], int], int]:
    """
    Returns:
      tile_overlap: {(x0, y0): num_distinct_scenes_in_tile}
      total_scenes: number of distinct scenes found overall
    """
    root = osp.join(save_dir, map_location)
    tile_scenes: Dict[Tuple[float, float], Set[str]] = defaultdict(set)
    all_scenes: Set[str] = set()

    for tile_dir, x0, y0 in _iter_tile_dirs(root):
        scenes_here: Set[str] = set()
        for f in os.listdir(tile_dir):
            s = _parse_scene_from_filename(f)
            if s:
                scenes_here.add(s)
                all_scenes.add(s)
        if scenes_here:
            tile_scenes[(x0, y0)].update(scenes_here)

    tile_overlap = {k: len(v) for k, v in tile_scenes.items()}
    return tile_overlap, len(all_scenes)


def build_grid_from_tiles(
    tile_overlap: Dict[Tuple[float, float], int],
    tile_size_x: float,
    tile_size_y: float,
):
    """
    Build a dense 2D grid (for pcolormesh) from sparse tile overlaps.

    Returns:
      X_edges: (nx+1,) sorted array of x edges
      Y_edges: (ny+1,) sorted array of y edges
      Z: (ny, nx) array of overlap counts aligned with edges
    """
    if not tile_overlap:
        return np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.zeros((1, 1), dtype=int)

    xs = sorted(set([x0 for (x0, _) in tile_overlap.keys()]))
    ys = sorted(set([y0 for (_, y0) in tile_overlap.keys()]))

    X_edges = np.array(xs + [xs[-1] + tile_size_x], dtype=float)
    Y_edges = np.array(ys + [ys[-1] + tile_size_y], dtype=float)

    nx = len(xs)
    ny = len(ys)
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: j for j, y in enumerate(ys)}

    Z = np.zeros((ny, nx), dtype=int)
    for (x0, y0), count in tile_overlap.items():
        i = x_idx[x0]
        j = y_idx[y0]
        Z[j, i] = count

    return X_edges, Y_edges, Z


def _autosized_fig(nx: int, ny: int, tile_scale_in: float, min_size=(6, 5)):
    """Return (fig_width, fig_height) in inches based on tiles and desired per-tile size."""
    w = max(min_size[0], nx * tile_scale_in)
    h = max(min_size[1], ny * tile_scale_in)
    return (w, h)

def plot_overlap(
    X_edges: np.ndarray,
    Y_edges: np.ndarray,
    Z: np.ndarray,
    out_path: str = None,
    title: str = "",
    annotate: bool = False,
    dpi: int = 200,
    tile_scale_in: float = 0.8,     # <-- inches per tile (increase for bigger tiles)
    annotate_bbox: bool = True,     # <-- draw a white box behind numbers
    annotate_fixed_size: int = 0,   # <-- if >0, use this font size instead of autoscaling
    grid_linewidth: float = 0.2,    # thinner borders so numbers stand out
):
    """
    Draw a pcolormesh heatmap of tile overlaps with equal aspect and colorbar.
    Tiles/labels scale with grid size so annotations are readable.
    """
    nx = len(X_edges) - 1
    ny = len(Y_edges) - 1

    # Figure size scales with number of tiles
    fig_w, fig_h = _autosized_fig(nx, ny, tile_scale_in)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    mesh = ax.pcolormesh(X_edges, Y_edges, Z, edgecolors='k', linewidth=grid_linewidth)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("# distinct scenes in tile")

    if annotate:
        # Auto font size based on pixels per tile (if not fixed)
        if annotate_fixed_size > 0:
            fsz = annotate_fixed_size
        else:
            # pixels per tile ~= dpi * tile_scale_in
            px_per_tile = dpi * tile_scale_in
            # Convert to a font size that stays readable but not huge
            fsz = int(np.clip(px_per_tile * 0.35, 8, 22))

        # Put counts at tile centers
        Xc = 0.5 * (X_edges[:-1] + X_edges[1:])
        Yc = 0.5 * (Y_edges[:-1] + Y_edges[1:])
        bbox_props = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8) if annotate_bbox else None

        for j, yc in enumerate(Yc):
            for i, xc in enumerate(Xc):
                val = Z[j, i]
                if val > 0:
                    ax.text(
                        xc, yc, str(val),
                        ha='center', va='center',
                        fontsize=fsz,
                        weight='bold',
                        bbox=bbox_props
                    )

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()

# Example main() customization
def main():
    # ---- Edit these values ----
    save_dir = "data/av2_gmaps/lidar_map"
    map_location = "WDC" # "singapore-onenorth"  # "singapore-hollandvillage" # "boston-seaport"  #"singapore-queenstown"
    tile_size_x = 50.0
    tile_size_y = 50.0

    out_path = f"{map_location}_overlap.png" # "overlap_sg_queenstown.png"  # None to show interactively
    title = f"Tile overlap in {map_location}"
    dpi = 200

    # Readability knobs:
    tile_scale_in = 1.2     # increase to make each tile bigger on the figure
    annotate = True
    annotate_bbox = True
    annotate_fixed_size = 0  # set e.g. 16 to force a fixed font size
    grid_linewidth = 0.15
    # ---------------------------

    tile_overlap, total_scenes = compute_tile_overlap_counts(save_dir, map_location)
    if not tile_overlap:
        print("No tiles found. Nothing to plot.")
        return

    X_edges, Y_edges, Z = build_grid_from_tiles(tile_overlap, tile_size_x, tile_size_y)
    title = f"{title} (scenes: {total_scenes})"

    plot_overlap(
        X_edges, Y_edges, Z,
        out_path=out_path,
        title=title,
        annotate=annotate,
        dpi=dpi,
        tile_scale_in=tile_scale_in,
        annotate_bbox=annotate_bbox,
        annotate_fixed_size=annotate_fixed_size,
        grid_linewidth=grid_linewidth,
    )

if __name__ == "__main__":
    main()
