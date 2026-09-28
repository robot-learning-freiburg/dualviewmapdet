# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import os
import numpy as np
import torch
import cv2
import io
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from tools.visualization.color_mapping import argoverse2_cmap
from tools.visualization.color_mapping import cmap
from tools.visualization.options import RENDER_BEV_JOINT, SCORE_THRESH, MAP_SCORE_THRESH, DET_COLOR_TRACKID, \
    DET_GT_GREY, DET_RENDER_VEL, RENDER_BLACK_BEV, LINE_WIDTH

RANGE_NUSC = (55, 55)
RANGE_ARGOVERSE2 = (55, 55) # for long-range: (150, 150)

color_mapping = cmap

class BEVRender:
    def __init__(
            self,
            plot_choices,
            out_dir,
            dataset_type
    ):
        self.plot_choices = plot_choices
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        self.joint_dir = os.path.join(out_dir, "bev")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.joint_dir, exist_ok=True)

        if dataset_type == "NuScenes3DDataset":
            self.xlim = RANGE_NUSC[0]
            self.ylim = RANGE_NUSC[1]
            self.rotate_bev = False
        elif dataset_type == "Argoverse2DatasetT":
            self.xlim = RANGE_ARGOVERSE2[0]
            self.ylim = RANGE_ARGOVERSE2[1]
            self.rotate_bev = True
            global color_mapping
            color_mapping[:len(argoverse2_cmap)] = argoverse2_cmap
        else:
            raise NotImplementedError

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        self.axes.set_xlim(- self.xlim, self.xlim)
        self.axes.set_ylim(- self.ylim, self.ylim)
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.axis('off')

        # Set background to black
        if RENDER_BLACK_BEV:
            self.fig.patch.set_facecolor("black")  # Figure background
            self.axes.set_facecolor("black")  # Axes background

        # BEV grid overlay
        # Use the smaller half-range so circles fit in both x/y
        r_max = min(self.xlim, self.ylim)  # half-range in "meters"
        step_m = 10  # every 10 meters, like in your code
        grey = (127 / 255.0, 127 / 255.0, 127 / 255.0)
        lw = LINE_WIDTH
        z = 0  # draw beneath everything else

        # Concentric circles
        r = step_m
        while r <= r_max:
            self.axes.add_patch(Circle((0.0, 0.0), r, fill=False,
                                       edgecolor=grey, linewidth=lw, zorder=z, alpha=0.9))
            r += step_m

        # Crosshair lines through origin
        self.axes.plot([-self.xlim, self.xlim], [0, 0], color=grey, linewidth=lw, zorder=z, alpha=0.9)
        self.axes.plot([0, 0], [-self.ylim, self.ylim], color=grey, linewidth=lw, zorder=z, alpha=0.9)

    def render(
            self,
            data,
            result,
            index,
    ):
        self.reset_canvas()

        self.draw_detection_pred(result)
        self.draw_track_pred(result)

        if not RENDER_BEV_JOINT:
            save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
            self.save_fig(save_path_pred)
            self.reset_canvas()

        self.draw_detection_gt(data)
        self.draw_track_gt(data)

        if not RENDER_BEV_JOINT:
            save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
            self.save_fig(save_path_gt)
            return save_path_gt, save_path_pred
        else:
            save_path_joint = os.path.join(self.joint_dir, str(index).zfill(4) + '.jpg')
            self.save_fig(save_path_joint)
            return save_path_joint

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)

        if self.rotate_bev:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                        transparent=False, facecolor="black")
            buf.seek(0)
            img = Image.open(buf).transpose(Image.ROTATE_90)  # 90° CCW
            img = img.convert("RGB")  # strip alpha for JPEG
            img.save(filename)
            buf.close()
        else:
            plt.savefig(filename)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det'] or "gt_bboxes_3d" not in data:
            return

        for i in range(data['gt_labels_3d'].shape[0]):
            label = data['gt_labels_3d'][i]
            if label == -1:
                continue

            if DET_COLOR_TRACKID:
                color = color_mapping[data['instance_inds'][i] % len(color_mapping)]
            else:
                color = color_mapping[label % len(color_mapping)][::-1]
            if DET_GT_GREY:
                color = np.array([166, 166, 166]) / 255.0

            # draw corners
            corners = box3d_to_corners(data['gt_bboxes_3d'])[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=LINE_WIDTH, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=LINE_WIDTH, linestyle='-')

            if DET_RENDER_VEL:
                # draw velocity vector
                vel_x, vel_y = data['gt_bboxes_3d'][i, 7], data['gt_bboxes_3d'][i, 8]
                self.axes.arrow(
                    center[0], center[1],
                    vel_x, vel_y,
                    head_width=0.5, head_length=0.5,
                    fc=color, ec=color, linewidth=LINE_WIDTH
                )

    def draw_detection_pred(self, result):
        if not self.plot_choices['draw_pred'] or not self.plot_choices['det'] or "boxes_3d" not in result:
            return

        BOX_ALPHA = 1.00
        VEL_ALPHA = 1.00

        def to_rgba01(c, a):
            c = np.array(c, dtype=float)
            if c.max() > 1.0:  # convert 0–255 -> 0–1 if needed
                c /= 255.0
            return (float(c[0]), float(c[1]), float(c[2]), float(a))

        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH:
                continue
            label = result['labels_3d'][i]

            if DET_COLOR_TRACKID:
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
            else:
                color = color_mapping[label % len(color_mapping)][::-1]

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            rgba_box = to_rgba01(color, BOX_ALPHA)
            self.axes.plot(x, y,
                           color=rgba_box,  # or color=color, alpha=BOX_ALPHA
                           linewidth=LINE_WIDTH, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y,
                           color=rgba_box,  # or color=color, alpha=BOX_ALPHA
                           linewidth=LINE_WIDTH, linestyle='-')

            if DET_RENDER_VEL:
                # draw velocity vector
                vel_x, vel_y = bboxes[i, 7], bboxes[i, 8]
                rgba_vel = to_rgba01(color, VEL_ALPHA)
                self.axes.arrow(center[0], center[1],
                                vel_x, vel_y,
                                head_width=0.5, head_length=0.5,
                                fc=rgba_vel, ec=rgba_vel,
                                linewidth=LINE_WIDTH,
                                alpha=VEL_ALPHA)  # alpha multiplies with RGBA alpha

    def draw_track_gt(self, data):
        """
        Draw all ground-truth boxes from the current sequence in BEV (nuScenes: y-forward, x-right).
        Keeps a per-scene history of frames and reprojects all historical boxes into the
        *current* LiDAR frame using their stored lidar2global and the current frame's lidar2global.

        Expected `result` keys:
          - gt_bboxes_3d: Tensor [N,10] = x y z w l h yaw vx vy vz   (current LiDAR frame)
          - gt_labels_3d: Tensor [N]
          - instance_inds: Tensor/ndarray/list [N]
          - scene_token: str
          - lidar2global: ndarray [4,4]
        """
        if not self.plot_choices['track'] or "gt_bboxes_3d" not in data:
            return

        # -------- helpers --------
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def transform_points(T, pts):
            """
            Apply 4x4 transform T to points.
            pts: (..., 3) -> (..., 3)
            """
            shp = pts.shape
            pts_flat = pts.reshape(-1, 3)
            hom = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)], axis=1).T  # 4xM
            out = (T @ hom)[:3].T
            return out.reshape(shp)

        # -------- scene bookkeeping --------
        scene_token = data.get('scene_token', None)
        if scene_token is None:
            return

        if not hasattr(self, '_last_scene_token_gt'):
            self._last_scene_token_gt = None
        if not hasattr(self, '_track_history_frames_gt'):
            # list of frames; each is dict(boxes [N,10], ids [N], scores [N], lidar2global [4,4])
            self._track_history_frames_gt = []

        if scene_token != self._last_scene_token_gt:
            # New sequence -> clear history
            self._track_history_frames_gt = []
            self._last_scene_token_gt = scene_token

        # -------- store current frame --------
        boxes_np = to_numpy(data['gt_bboxes_3d'])
        labels_np = to_numpy(data['gt_labels_3d'])
        ids_in = data.get('instance_inds', None)
        ids_np = to_numpy(ids_in).astype(np.int64)
        T_l2g_cur = to_numpy(data['lidar2global']).astype(np.float64)

        self._track_history_frames_gt.append(dict(
            boxes=boxes_np,
            ids=ids_np,
            labels=labels_np,
            lidar2global=T_l2g_cur.copy()
        ))

        # -------- draw all frames reprojected into current LiDAR --------
        T_g2l_cur = np.linalg.inv(T_l2g_cur)
        centers_by_iid = {}

        poly_idx = np.array([0, 3, 7, 4, 0], dtype=np.int64)

        for frame in self._track_history_frames_gt:
            boxes_f = frame['boxes']
            ids_f = frame['ids']
            labels_f = frame['labels']
            T_l2g_f = frame['lidar2global']

            # score filter at draw time
            mask = labels_f != -1
            if not np.any(mask):
                continue
            boxes_f = boxes_f[mask]
            ids_f = ids_f[mask]

            if boxes_f.shape[0] == 0:
                continue

            # 1) corners in that frame's LiDAR
            corners_f = box3d_to_corners(boxes_f)  # [N, 8, 3]

            # 2) transform into Global, then into current LiDAR (compose once)
            T_f2cur = T_g2l_cur @ T_l2g_f  # LiDAR_f -> Global -> LiDAR_cur
            corners_cur = transform_points(T_f2cur, corners_f)  # [N, 8, 3]

            # 3) draw polygons + forward tick using original indexing logic
            for i in range(corners_cur.shape[0]):
                c = corners_cur[i]  # [8,3]
                poly = c[poly_idx]  # [5,3] -> your original [0,3,7,4,0]

                x = poly[:, 0]
                y = poly[:, 1]
                iid = int(ids_f[i])
                color = color_mapping[iid % len(color_mapping)]

                # box outline
                self.axes.plot(x, y, color=color, linewidth=LINE_WIDTH, linestyle='-')

                # forward-direction tick (original: midpoint of corners[2:4] -> mean of corners[0:4])
                forward_center = np.mean(poly[2:4, :2], axis=0)  # between indices 2 and 3 of the selected poly
                center_xy = np.mean(poly[0:4, :2], axis=0)
                self.axes.plot(
                    [forward_center[0], center_xy[0]],
                    [forward_center[1], center_xy[1]],
                    color=color, linewidth=LINE_WIDTH, linestyle='-'
                )

                # accumulate centers for trajectories
                centers_by_iid.setdefault(iid, []).append((center_xy[0], center_xy[1]))

        # optional: draw center trajectories across frames
        for iid, pts in centers_by_iid.items():
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            color = color_mapping[iid % len(color_mapping)]
            self.axes.plot(xs, ys, color=color, linewidth=LINE_WIDTH, linestyle='-')

    def draw_track_pred(self, result):
        """
        Draw all predicted boxes from the current sequence in BEV (nuScenes: y-forward, x-right).
        Keeps a per-scene history of frames and reprojects all historical boxes into the
        *current* LiDAR frame using their stored lidar2global and the current frame's lidar2global.

        Expected `result` keys:
          - boxes_3d: Tensor [N,10] = x y z w l h yaw vx vy vz   (current LiDAR frame)
          - scores_3d: Tensor [N]
          - labels_3d: Tensor [N]
          - instance_ids: Tensor/ndarray/list [N]
          - scene_token: str
          - lidar2global: ndarray [4,4]
        """
        if not self.plot_choices['draw_pred'] or not self.plot_choices['track'] or "boxes_3d" not in result:
            return

        # -------- helpers --------
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def transform_points(T, pts):
            """
            Apply 4x4 transform T to points.
            pts: (..., 3) -> (..., 3)
            """
            shp = pts.shape
            pts_flat = pts.reshape(-1, 3)
            hom = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=pts_flat.dtype)], axis=1).T  # 4xM
            out = (T @ hom)[:3].T
            return out.reshape(shp)

        # -------- scene bookkeeping --------
        scene_token = result.get('scene_token', None)
        if scene_token is None:
            return

        if not hasattr(self, '_last_scene_token_pred'):
            self._last_scene_token_pred = None
        if not hasattr(self, '_track_history_frames_pred'):
            # list of frames; each is dict(boxes [N,10], ids [N], scores [N], lidar2global [4,4])
            self._track_history_frames_pred = []

        if scene_token != self._last_scene_token_pred:
            # New sequence -> clear history
            self._track_history_frames_pred = []
            self._last_scene_token_pred = scene_token

        # -------- store current frame --------
        boxes_np = to_numpy(result['boxes_3d'])
        scores_np = to_numpy(result['scores_3d'])
        ids_in = result.get('instance_ids', None)
        ids_np = to_numpy(ids_in).astype(np.int64)
        T_l2g_cur = to_numpy(result['lidar2global']).astype(np.float64)

        self._track_history_frames_pred.append(dict(
            boxes=boxes_np,
            ids=ids_np,
            scores=scores_np,
            lidar2global=T_l2g_cur.copy()
        ))

        # -------- draw all frames reprojected into current LiDAR --------
        T_g2l_cur = np.linalg.inv(T_l2g_cur)
        centers_by_iid = {}

        poly_idx = np.array([0, 3, 7, 4, 0], dtype=np.int64)

        for frame in self._track_history_frames_pred:
            boxes_f = frame['boxes']
            ids_f = frame['ids']
            scores_f = frame['scores']
            T_l2g_f = frame['lidar2global']

            # score filter at draw time
            mask = scores_f >= SCORE_THRESH
            if not np.any(mask):
                continue
            boxes_f = boxes_f[mask]
            ids_f = ids_f[mask]

            if boxes_f.shape[0] == 0:
                continue

            # 1) corners in that frame's LiDAR
            corners_f = box3d_to_corners(boxes_f)  # [N, 8, 3]

            # 2) transform into Global, then into current LiDAR (compose once)
            T_f2cur = T_g2l_cur @ T_l2g_f  # LiDAR_f -> Global -> LiDAR_cur
            corners_cur = transform_points(T_f2cur, corners_f)  # [N, 8, 3]

            # 3) draw polygons + forward tick using original indexing logic
            for i in range(corners_cur.shape[0]):
                c = corners_cur[i]  # [8,3]
                poly = c[poly_idx]  # [5,3] -> your original [0,3,7,4,0]

                x = poly[:, 0]
                y = poly[:, 1]
                iid = int(ids_f[i])
                color = color_mapping[iid % len(color_mapping)]

                # box outline
                self.axes.plot(x, y, color=color, linewidth=LINE_WIDTH, linestyle='-')

                # forward-direction tick (original: midpoint of corners[2:4] -> mean of corners[0:4])
                forward_center = np.mean(poly[2:4, :2], axis=0)  # between indices 2 and 3 of the selected poly
                center_xy = np.mean(poly[0:4, :2], axis=0)
                self.axes.plot(
                    [forward_center[0], center_xy[0]],
                    [forward_center[1], center_xy[1]],
                    color=color, linewidth=LINE_WIDTH, linestyle='-'
                )

                # accumulate centers for trajectories
                centers_by_iid.setdefault(iid, []).append((center_xy[0], center_xy[1]))

        # optional: draw center trajectories across frames
        for iid, pts in centers_by_iid.items():
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            color = color_mapping[iid % len(color_mapping)]
            self.axes.plot(xs, ys, color=color, linewidth=LINE_WIDTH, linestyle='-')
