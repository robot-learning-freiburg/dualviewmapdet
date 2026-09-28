# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import numpy as np
import torch

from projects.mmdet3d_plugin import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.structures.utils import limit_period


def transform_points(points: np.ndarray, T_mat: np.ndarray) -> np.ndarray:
    """Homogeneous transform for a (N, 3) array of points."""
    pts_h = np.hstack([points.astype(np.float64), np.ones((points.shape[0], 1))])
    pts_out = (T_mat @ pts_h.T).T
    pts_out[:, :3] /= pts_out[:, [3]]
    return pts_out[:, :3].astype(np.float32)


def nuscenes_to_mmdet3d_boxes(
    nusc_boxes: np.ndarray | torch.Tensor,
) -> LiDARInstance3DBoxes:
    """
    Undo the earlier conversion and recover *mmdet3d* `LiDARInstance3DBoxes`.

    Args
    ----
    nusc_boxes : (N, 10)     [x y z w l h yaw vel_x vel_y, vel_z]
        nuScenes-style boxes used in SparseDrive
        (i.e. centres, w-l-h order, CCW yaw, velocity in *vehicle* frame).

    Returns
    -------
    LiDARInstance3DBoxes
        Boxes in the original mmdet3d LiDAR coordinate system
        (bottom-centre, l-w-h order, clockwise yaw, velocity in local frame).
    """
    # Work on a NumPy copy
    if isinstance(nusc_boxes, torch.Tensor):
        nusc_boxes = nusc_boxes.cpu().numpy()
    boxes = nusc_boxes.copy()

    # 1) dimensions: [w, l, h] ➜ [l, w, h]
    boxes[:, 3:5] = boxes[:, [4, 3]]

    # 2) centre ➜ bottom-centre (z – h/2)
    boxes[:, 2] -= boxes[:, 5] * 0.5

    # 3) new-lidar ➜ old-lidar (mmdet3d) coordinates
    oldlidar2newlidar = np.eye(4, dtype=np.float32)
    oldlidar2newlidar[:3, :3] = np.array([[0, -1, 0],
                                          [1,  0, 0],
                                          [0,  0, 1]], dtype=np.float32)
    newlidar2oldlidar = np.linalg.inv(oldlidar2newlidar)

    boxes[:, :3] = transform_points(boxes[:, :3], newlidar2oldlidar)

    # 4) yaw: CCW ➜ CW
    boxes[:, 6] = 2.0 * np.pi - boxes[:, 6]
    #boxes[:, 6] = limit_period(torch.from_numpy(boxes[:, 6]), 0.5, np.pi).numpy()

    # 5) velocity
    boxes[:, 7:9] = boxes[:, [8, 7]]
    boxes[:, 8] = -boxes[:, 8]

    # 6) wrap up as LiDARInstance3DBoxes (bottom-centre, l-w-h, CW yaw)
    return LiDARInstance3DBoxes(torch.from_numpy(boxes), box_dim=10)


def mmdet3d_to_nuscenes_boxes(
    mmdet3d_boxes: LiDARInstance3DBoxes,
) -> torch.Tensor:
    """
    Convert from *mmdet3d* `LiDARInstance3DBoxes` to nuscenes boxes that are used in SparseDrive.

    Args
    ----
    LiDARInstance3DBoxes
        Boxes in the original mmdet3d LiDAR coordinate system
        (bottom-centre, l-w-h order, clockwise yaw, velocity in local frame).
    Returns
    -------
    nusc_boxes : (N, 7) or (N, 9)      [x y z w l h yaw (vel_x vel_y)]
        nuScenes-style boxes used in SparseDrive
        (i.e. centres, w-l-h order, CCW yaw, velocity in *vehicle* frame).
    """

    oldlidar2newlidar = np.eye(4)
    oldlidar2newlidar[:3, :3] = np.array([[0, -1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]], dtype=np.float32)

    boxes = mmdet3d_boxes.clone()
    boxes = boxes.tensor.numpy()
    boxes_center = boxes[:, :3]
    boxes_center[:, 2] += boxes[:, 5] / 2  # nuScenes box location is at box center
    boxes_center = transform_points(boxes_center, oldlidar2newlidar)

    boxes[:, :3] = boxes_center

    # [l, w, h] -> [w, l, h]
    tmp_l = boxes[:, 3].copy()  # length
    tmp_w = boxes[:, 4].copy()  # width
    tmp_h = boxes[:, 5].copy()  # height
    # reorder to w, l, h
    boxes[:, 3] = tmp_w
    boxes[:, 4] = tmp_l
    boxes[:, 5] = tmp_h

    # LiDARInstance3DBoxes of mmdet3d rotate clockwise while nuScenes rotates counterclockwise
    # see https://github.com/nutonomy/nuscenes-devkit/issues/830
    # In addition, LiDARInstance3DBoxes are rotated by 180 degree (yaw) compared to nuScenes boxes
    boxes[:, 6] = 2 * np.pi - boxes[:, 6]

    # velocity
    if boxes.shape[1] > 7:
        boxes[:, 7:9] = boxes[:, [8, 7]]
        boxes[:, 7] = -boxes[:, 7]

    return torch.from_numpy(boxes).to(mmdet3d_boxes.tensor.device)

