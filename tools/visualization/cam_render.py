# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import os
import numpy as np
import cv2
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from tools.visualization.bev_render import (
    color_mapping,
    SCORE_THRESH,
)
from tools.visualization.options import RENDER_CAM_NAME, DET_COLOR_TRACKID, DET_GT_GREY, LINE_WIDTH, DET_GT_CAM

CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

CAM_NAMES_ARGOVERSE2 = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT_CENTER',
    'CAM_FRONT_RIGHT',
    'CAM_SIDE_RIGHT',
    'CAM_REAR_RIGHT',
    'CAM_REAR_LEFT',
    'CAM_SIDE_LEFT',
]
CAM_NAMES_ARGOVERSE2_converter = [
    'CAM_REAR_LEFT',
    'CAM_SIDE_LEFT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_CENTER',
    'CAM_FRONT_RIGHT',
    'CAM_SIDE_RIGHT',
    'CAM_REAR_RIGHT'
]

CAM_IMG_SIZES_NUSC = [(1600, 900), (1600, 900), (1600, 900), (1600, 900), (1600, 900), (1600, 900)]
CAM_IMG_SIZES_ARGOVERSE2 = [(2048, 1550), (2048, 1550), (2048, 1550), (1550, 2048), (2048, 1550), (2048, 1550), (2048, 1550)]

class CamRender:
    def __init__(
            self,
            plot_choices,
            out_dir,
            dataset_type,
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        os.makedirs(self.pred_dir, exist_ok=True)
        self.dataset_type = dataset_type

        if dataset_type == "NuScenes3DDataset":
            self.CAM_IMG_SIZES = CAM_IMG_SIZES_NUSC
            self.CAM_NAMES = CAM_NAMES_NUSC
            self.CAM_NAMES_converter = CAM_NAMES_NUSC_converter
        elif dataset_type == "Argoverse2DatasetT":
            self.CAM_IMG_SIZES = CAM_IMG_SIZES_ARGOVERSE2
            self.CAM_NAMES = CAM_NAMES_ARGOVERSE2
            self.CAM_NAMES_converter = CAM_NAMES_ARGOVERSE2_converter
        else:
            raise NotImplementedError

    def reset_canvas(self):
        plt.close()

        if self.dataset_type in ["NuScenes3DDataset"]:
            self.fig, self.axes = plt.subplots(2, 3, figsize=(160 / 3, 20))
            for ax in self.axes.flatten():
                ax.set_axis_off()
                ax.grid(False)
            plt.tight_layout()
            self.axes_map = {}
            return

        elif self.dataset_type == "Argoverse2DatasetT":
            # --- Compute aspect-aware GridSpec from actual image sizes ---
            # Determine which cam goes big on the right and the six on the left
            main_cam = 'CAM_FRONT_CENTER'
            left_cam_order = [
                'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_RIGHT',
                'CAM_REAR_RIGHT', 'CAM_REAR_LEFT', 'CAM_SIDE_LEFT',
            ]

            # Read one frame’s image sizes (W,H) per cam
            def _img_size_for_cam(cam):
                idx = self.CAM_NAMES_converter.index(cam)
                path = self._last_data['img_filename'][idx] if hasattr(self, "_last_data") else None
                if path and os.path.exists(path):
                    img = Image.open(path)
                    return img.size  # (W, H)
                # Fallback to declared sizes
                return self.CAM_IMG_SIZES[idx]

            # Stash latest data once per call chain (set in render())
            # Call self._set_last_data(data) at the top of render() before reset_canvas()
            # Or: lightly adapt render() to set self._last_data=data before calling reset_canvas().

            # Left block images all use their own aspect, but the grid cells must have uniform row heights.
            # Use the *median* left-cam (W,H) as the representative per-cell size.
            left_wh = [_img_size_for_cam(c) for c in left_cam_order]
            left_w = np.median([w for (w, h) in left_wh])
            left_h = np.median([h for (w, h) in left_wh])

            # Big right image size
            big_w, big_h = _img_size_for_cam(main_cam)

            # GridSpec ratios so that:
            # - two rows each have height proportional to left_h
            # - three left columns each have width proportional to left_w
            # - the right column has width so that height=2*left_h implies width = (2*left_h)*(big_w/big_h)
            height_ratios = [left_h, left_h]
            right_col_width = (2.0 * left_h) * (big_w / float(big_h))
            width_ratios = [left_w, left_w, left_w, right_col_width]

            # Figure size proportional to total logical width/height (scale down by a factor)
            scale = 80.0  # tweak if you want larger/smaller figures
            fig_w = (3 * left_w + right_col_width) / scale
            fig_h = (2 * left_h) / scale

            self.fig = plt.figure(figsize=(fig_w, fig_h))
            gs = GridSpec(
                nrows=2, ncols=4, figure=self.fig,
                width_ratios=width_ratios, height_ratios=height_ratios
            )

            # Build axes
            ax_big = self.fig.add_subplot(gs[:, 3])
            left_axes = [self.fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

            # Turn off axes and set aspect handling
            def _prep(ax):
                ax.set_axis_off()
                ax.grid(False)
                ax.set_aspect('equal', adjustable='box')
                ax.set_anchor('NW')  # stick to top-left so there’s no extra padding above/below

            _prep(ax_big)
            for ax in left_axes: _prep(ax)

            # Map cams to axes
            self.axes_map = {main_cam: ax_big}
            for ax, cam in zip(left_axes, left_cam_order):
                self.axes_map[cam] = ax

            # Keep a flat list if other code expects self.axes
            self.axes = left_axes + [ax_big]

            # Remove inter-axes gaps and outer margins
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        else:
            raise NotImplementedError

    def render(
            self,
            data,
            result,
            index,
    ):
        self._last_data = data
        self.reset_canvas()
        self.render_image_data(data, index)
        self.draw_detection_pred(data, result)
        if DET_GT_CAM:
            self.draw_detection_gt(data)
        self.draw_motion_pred(data, result)
        self.draw_planning_pred(data, result)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path)
        return save_path

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        if RENDER_CAM_NAME:
            return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            return image

    def update_image(self, image, cam):
        """Render image data for each camera."""
        ax = self.get_axis_by_cam(cam)
        h, w = image.shape[:2]
        ax.imshow(image, aspect='equal')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # y-down like image coords
        ax.set_anchor('NW')
        ax.axis('off');
        ax.grid(False)

    def get_axis_by_cam(self, cam):
        if self.dataset_type == "Argoverse2DatasetT":
            return self.axes_map[cam]
        # default (NuScenes): keep your old indexing logic via 2x3 grid
        j = self.CAM_NAMES.index(cam)
        return self.axes[j // 3, j % 3]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def render_image_data(self, data, index):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(self.CAM_NAMES):
            idx = self.CAM_NAMES_converter.index(cam)
            img_path = data['img_filename'][idx]
            image = self.load_image(img_path, cam)
            self.update_image(image, cam)

    def draw_detection_gt(self, data):
        if not self.plot_choices['det'] or "gt_bboxes_3d" not in data:
            return

        bboxes = data['gt_bboxes_3d']
        for j, cam in enumerate(self.CAM_NAMES):
            idx = self.CAM_NAMES_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[:3, 3]
            rot = Quaternion(matrix=extrinsic[:3, :3], atol=1e-07)
            imsize = self.CAM_IMG_SIZES[idx]
            ax = self.get_axis_by_cam(cam)

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

                center = bboxes[i, 0: 3]
                box_dims = bboxes[i, 3: 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        ax,
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(color, color, color),
                        linewidth=LINE_WIDTH,
                    )

            ax.set_xlim(0, imsize[0])
            ax.set_ylim(imsize[1], 0)

    def draw_detection_pred(self, data, result):
        if not self.plot_choices['draw_pred'] or not self.plot_choices['det'] or "boxes_3d" not in result:
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(self.CAM_NAMES):
            idx = self.CAM_NAMES_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[:3, 3]
            rot = Quaternion(matrix=extrinsic[:3, :3], atol=1e-07)
            imsize = self.CAM_IMG_SIZES[idx]
            ax = self.get_axis_by_cam(cam)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH:
                    continue
                label = result['labels_3d'][i]

                if DET_COLOR_TRACKID:
                    color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                else:
                    color = color_mapping[label % len(color_mapping)][::-1]

                center = bboxes[i, 0: 3]
                box_dims = bboxes[i, 3: 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        ax,
                        view=cam_intrinsic,
                        normalize=True,
                        colors=(color, color, color),
                        linewidth=LINE_WIDTH,
                    )

            ax.set_xlim(0, imsize[0])
            ax.set_ylim(imsize[1], 0)
