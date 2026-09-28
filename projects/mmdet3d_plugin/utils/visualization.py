# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import numpy as np

def save_bev_pointcloud_jpg(
        points_xyz: np.ndarray,
        out_path: str,
        *,
        xlim: tuple[float, float] = (-50.0, 50.0),
        ylim: tuple[float, float] = (-50.0, 50.0),
        point_size: float = 1.0,  # marker area in points^2
        alpha: float = 0.03,  # per-point opacity in [0,1]
        dpi: int = 300,
        axis_off: bool = True,
        figsize: tuple[float, float] = (6.0, 6.0),
):
    """
    Save a top-down BEV (x-y) plot of a point cloud to a .jpg:
      - black background
      - white filled circles
      - per-point opacity to make density visible

    If points_xyz is empty (N==0), still writes a black image.

    Args:
        points_xyz: (N,3) numpy array [x,y,z]. May be empty with shape (0,3).
        out_path: output file path, should end with .jpg.
        xlim/ylim: view limits in meters. Used for empty clouds as well.
        point_size: matplotlib scatter marker size (area in points^2).
        alpha: marker opacity in [0,1].
        dpi: output resolution.
        axis_off: hide axes/ticks for a clean render.
        figsize: (width,height) in inches.
    """
    import matplotlib.pyplot as plt
    if points_xyz is None:
        raise ValueError("points_xyz is None")
    points_xyz = np.asarray(points_xyz)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N,3), got {points_xyz.shape}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # Black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Plot points if present
    if points_xyz.shape[0] > 0:
        x = points_xyz[:, 0].astype(np.float64, copy=False)
        y = points_xyz[:, 1].astype(np.float64, copy=False)

        # Auto limits if user passed None (but defaults are fixed for empty-friendly behavior)
        _xlim = xlim
        _ylim = ylim
        if _xlim is None:
            xmin, xmax = float(np.min(x)), float(np.max(x))
            pad = 0.02 * max(1e-6, xmax - xmin)
            _xlim = (xmin - pad, xmax + pad)
        if _ylim is None:
            ymin, ymax = float(np.min(y)), float(np.max(y))
            pad = 0.02 * max(1e-6, ymax - ymin)
            _ylim = (ymin - pad, ymax + pad)

        ax.scatter(
            x, y,
            s=point_size,
            c="white",
            marker="o",
            alpha=alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
    else:
        # Empty cloud: still set limits so the output has consistent framing
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    if axis_off:
        ax.axis("off")

    fig.savefig(out_path, format="jpg", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)