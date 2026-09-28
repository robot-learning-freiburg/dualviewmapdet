# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

from tools.visualization.bev_render import BEVRender
from tools.visualization.cam_render import CamRender
from tools.visualization.options import RENDER_BEV_JOINT, plot_choices, START, END, INTERVAL, DET_RANGE, FPS


class Visualizer:
    def __init__(
            self,
            args,
            plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)

        cfg = Config.fromfile(args.config)
        self.dataset = build_dataset(cfg.data.val)
        if args.result_path is not None:
            self.results = mmcv.load(args.result_path)
        else:
            self.results = None
        self.bev_render = BEVRender(plot_choices, self.out_dir, dataset_type=cfg.dataset_type)
        self.cam_render = CamRender(plot_choices, self.out_dir, dataset_type=cfg.dataset_type)

    def add_vis(self, index):
        data = self.dataset.get_data_info(index)
        if self.results is not None:
            result = self.results[index]['img_bbox']
            # For tracking visualization
            result['scene_token'] = data['scene_token']
            result['lidar2global'] = data['lidar2global']
            range_filter_data = [
                (data, 'gt_bboxes_3d', ['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds']),
                (result, 'boxes_3d', ['boxes_3d', 'labels_3d', 'scores_3d', 'cls_scores', 'instance_ids'])
            ]
        else:
            result = dict()
            range_filter_data = [
                (data, 'gt_bboxes_3d', ['gt_bboxes_3d', 'gt_labels_3d', 'instance_inds']),
            ]

        # Range filter on gt and pred
        for d, key, fields in range_filter_data:
            mask = np.linalg.norm(d[key][:, :3], axis=1) <= DET_RANGE
            for f in fields:
                d[f] = d[f][mask]

        bev_paths = self.bev_render.render(data, result, index)
        cam_pred_path = self.cam_render.render(data, result, index)
        self.combine(bev_paths, cam_pred_path, index)

    def combine(self, bev_paths, cam_pred_path, index):
        cam_image = cv2.imread(cam_pred_path)
        if not RENDER_BEV_JOINT:
            bev_gt_path, bev_pred_path = bev_paths
            bev_gt = cv2.imread(bev_gt_path)
            bev_pred = cv2.imread(bev_pred_path)
            bev_gt = cv2.resize(bev_gt, (cam_image.shape[0], cam_image.shape[0]))
            bev_pred = cv2.resize(bev_pred, (cam_image.shape[0], cam_image.shape[0]))
            merge_image = cv2.hconcat([cam_image, bev_pred, bev_gt])
        else:
            bev_joint_path = bev_paths
            bev_joint = cv2.imread(bev_joint_path)
            bev_joint = cv2.resize(bev_joint, (cam_image.shape[0], cam_image.shape[0]))
            merge_image = cv2.hconcat([cam_image, bev_joint])

        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width // downsample, height //
                                   downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result-path',
                        default=None,
                        help='prediction result to visualize'
                             'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir',
        default='vis',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END, INTERVAL)):
        if visualizer.results is not None and idx > len(visualizer.results):
            break
        visualizer.add_vis(idx)

    visualizer.image2video(fps=FPS)


if __name__ == '__main__':
    main()