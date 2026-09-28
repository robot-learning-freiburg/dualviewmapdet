# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from mmdet3d
#   https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/mmdet3d/datasets/pipelines/test_time_aug.py
# Copyright (c) 2018-2019 Open-MMLab, licensed under the  Apache License 2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import mmcv
import warnings
from copy import deepcopy

import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_rotate,
                 pts_scale_ratio,
                 pts_rotate,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.img_rotate = img_rotate \
            if isinstance(img_rotate, list) else[img_rotate]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]
        self.pts_rotate = pts_rotate \
            if isinstance(pts_rotate, list) else [pts_rotate]

        assert mmcv.is_list_of(self.img_scale, float) # Note: changed from tuple to float
        assert mmcv.is_list_of(self.img_rotate, float)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)
        assert mmcv.is_list_of(self.pts_rotate, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        flip_aug = [False, True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for rotate in self.img_rotate:
                for pts_scale_ratio in self.pts_scale_ratio:
                    for pts_rotate in self.pts_rotate:
                        for flip in flip_aug:
                            for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                                for pcd_vertical_flip in pcd_vertical_flip_aug:
                                    for direction in self.flip_direction:
                                        # results.copy will cause bug
                                        # since it is shallow copy
                                        _results = deepcopy(results)
                                        _results['scale'] = scale
                                        _results['rotate'] = rotate
                                        _results['flip'] = flip
                                        _results['pcd_scale_factor'] = \
                                            pts_scale_ratio
                                        _results['pcd_rotate'] = \
                                            pts_rotate
                                        _results['flip_direction'] = direction
                                        _results['pcd_horizontal_flip'] = \
                                            pcd_horizontal_flip
                                        _results['pcd_vertical_flip'] = \
                                            pcd_vertical_flip

                                        # Custom
                                        ## Scale
                                        H, W = _results['aug_config']["original"]["H"], _results['aug_config']["original"]["W"]
                                        fH, fW = _results['aug_config']["original"]["final_dim"]
                                        resize = scale
                                        resize_dims = (int(W * resize), int(H * resize))
                                        newW, newH = resize_dims
                                        crop_h = (
                                                int((1 - np.mean(_results['aug_config']["original"]["bot_pct_lim"])) * newH)
                                                - fH
                                        )
                                        crop_w = int(max(0, newW - fW) / 2)
                                        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
                                        _results['aug_config']['resize'] = resize
                                        _results['aug_config']['resize_dims'] = resize_dims
                                        _results['aug_config']['crop'] = crop

                                        ## Rotate
                                        _results['aug_config']['rotate'] = rotate

                                        ## Flip
                                        _results['aug_config']['flip'] = flip

                                        ## Scale BEV
                                        _results['aug_config']['scale_bda'] = pts_scale_ratio

                                        ## Rotate BEV
                                        _results['aug_config']['rotate_bda'] = pts_rotate

                                        ## Flip BEV, reuse pcd flip for that
                                        _results['aug_config']['flip_dx'] = pcd_horizontal_flip
                                        _results['aug_config']['flip_dy'] = pcd_vertical_flip

                                        data = self.transforms(_results)
                                        aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str