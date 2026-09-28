# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

from typing import List, Sequence, Union, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GridMaskPcl(nn.Module):
    """
    GridMask augmentation for BEV point clouds (x,y plane).

    - Input: batch as list/tuple of point clouds, each (N, 3) or (N, >=3) with xyz in first 3 dims.
    - Masking happens in BEV (x,y). Points inside masked stripes are removed.
    - Same mask parameters are applied to the whole batch.
    - No rotate, no offset.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        use_x: mask stripes along x
        use_y: mask stripes along y
        ratio: masked stripe width as fraction of period d (0..1)
        mode: 0 -> drop points in stripes, 1 -> keep only points in stripes
        prob: probability of applying augmentation (when training)
        d_min: minimum period in meters
        d_max: maximum period in meters (default: min(x_span, y_span))
    """

    def __init__(
        self,
        point_cloud_range: Sequence[float],
        use_x: bool = True,
        use_y: bool = True,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.0,
        d_min: float = 1.0,
        d_max: Optional[float] = None,
    ):
        super().__init__()
        assert len(point_cloud_range) == 6, "point_cloud_range must be [x_min,y_min,z_min,x_max,y_max,z_max]"
        self.x_min, self.y_min, _, self.x_max, self.y_max, _ = [float(v) for v in point_cloud_range]

        span_x = self.x_max - self.x_min
        span_y = self.y_max - self.y_min
        assert span_x > 0 and span_y > 0, "Invalid point_cloud_range spans."

        self.use_x = bool(use_x)
        self.use_y = bool(use_y)

        self.ratio = float(ratio)
        assert 0.0 <= self.ratio <= 1.0

        self.mode = int(mode)
        assert self.mode in (0, 1)

        self.st_prob = float(prob)
        self.prob = float(prob)

        self.d_min = float(d_min)
        self.d_max = float(d_max) if d_max is not None else float(min(span_x, span_y))
        assert self.d_min > 0
        assert self.d_max > self.d_min, f"d_max ({self.d_max}) must be > d_min ({self.d_min})."

        # last sampled mask params (for visualization/debug)
        self._last_params = None  # dict(d=..., l=..., st_x=..., st_y=...)

    def set_prob(self, epoch: int, max_epoch: int):
        self.prob = self.st_prob * float(epoch) / float(max_epoch)

    @torch.no_grad()
    def forward(self, pcls: Union[Sequence[torch.Tensor], List[torch.Tensor], tuple]) -> List[torch.Tensor]:
        if not isinstance(pcls, (list, tuple)):
            raise TypeError("GridMaskBEVPointCloud expects a list/tuple of point clouds (batch).")

        if (not self.training) or (torch.rand((), device="cpu").item() > self.prob):
            return list(pcls)

        # sample one mask config for whole batch
        d = torch.empty((), device="cpu").uniform_(self.d_min, self.d_max).item()  # meters
        eps = 1e-6
        l = max(min(d * self.ratio, d - eps), eps)  # stripe width in meters
        st_x = torch.empty((), device="cpu").uniform_(0.0, d).item()
        st_y = torch.empty((), device="cpu").uniform_(0.0, d).item()

        self._last_params = dict(d=d, l=l, st_x=st_x, st_y=st_y)

        out: List[torch.Tensor] = []
        for pts in pcls:
            if not torch.is_tensor(pts):
                raise TypeError("Each point cloud must be a torch.Tensor.")
            if pts.numel() == 0:
                out.append(pts)
                continue
            if pts.ndim != 2 or pts.size(-1) < 3:
                raise ValueError("Each point cloud must have shape (N, 3) or (N, >=3) with xyz in the first 3 dims.")

            x = pts[:, 0]
            y = pts[:, 1]

            striped = torch.zeros((pts.size(0),), dtype=torch.bool, device=pts.device)

            if self.use_x:
                rx = torch.remainder(x - self.x_min - st_x, d)  # [0, d)
                striped |= (rx < l)

            if self.use_y:
                ry = torch.remainder(y - self.y_min - st_y, d)  # [0, d)
                striped |= (ry < l)

            keep = striped if self.mode == 1 else ~striped
            out.append(pts[keep])

        return out

    @torch.no_grad()
    def visualize_bev(
        self,
        pcls: Union[Sequence[torch.Tensor], torch.Tensor],
        title: str = "BEV point cloud",
        s: float = 1.0,
        alpha: float = 0.5,
        color_by: str = "batch",  # "batch" or "z"
        ax=None,
        show: bool = True,
    ):
        """
        Plot a BEV scatter (x,y) with matplotlib. Plots and (optionally) shows; does not save.

        Args:
            pcls: list/tuple of point clouds or a single point cloud tensor.
            title: plot title.
            s: marker size.
            alpha: marker alpha.
            color_by: "batch" -> different colors per batch element; "z" -> color by z height.
            ax: optional existing matplotlib Axes.
            show: if True, calls plt.show().
        """
        if not isinstance(pcls, (list, tuple)):
            pcls = [pcls]

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
            created_fig = True

        for bi, pts in enumerate(pcls):
            if not torch.is_tensor(pts) or pts.numel() == 0:
                continue

            xy = pts[:, :2].detach().cpu().numpy()

            if color_by == "z" and pts.size(-1) >= 3:
                c = pts[:, 2].detach().cpu().numpy()
                ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha, c=c)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha, label=f"b{bi}")

        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.5)

        if color_by != "z" and len(pcls) > 1:
            ax.legend(loc="upper right", markerscale=4, frameon=True)

        # annotate last mask params if available
        if self._last_params is not None:
            d = self._last_params["d"]
            l = self._last_params["l"]
            st_x = self._last_params["st_x"]
            st_y = self._last_params["st_y"]
            ax.text(
                0.02,
                0.02,
                f"d={d:.2f}m, l={l:.2f}m, st_x={st_x:.2f}, st_y={st_y:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        if show:
            plt.show()

        return ax
