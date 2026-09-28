# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# This source code of class BEVDataAug is derived from BEVNeXt
#   https://github.com/woxihuanjiangguo/BEVNeXt/blob/9b0e4ad33ed3e82dc9cee9f0f66ffd1899095026/mmdet3d/datasets/pipelines/loading.py#L1329
# Copyright (c) 2024 Zhenxin Li, licensed under the Apache License 2.0,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
import torch.nn.functional as F


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            H, W = imgs[i].shape[:2]
            if H > W: # Special case for Argoverse2 front camera
                resize, resize_dims, crop = self._sample_augmentation_f(imgs[i])
                aug_config_f = {"resize_dims": resize_dims,
                                "resize": resize,
                                "crop": crop}
                img_f, mat_f = self._img_transform(
                    np.uint8(imgs[i]), aug_config_f,
                )
                imgs[i] = Image.fromarray(np.uint8(img_f))
            else:
                mat_f = np.eye(4)

            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            mat = mat @ mat_f
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] = (
                    mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def _sample_augmentation_f(self, img):
        """For the Argoverse2 front camera (from Far3D)."""
        H, W = img.shape[:2]
        fH, fW = W, H

        resize = np.round(((H + 50) / W), 2)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize, resize_dims, crop


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ) # lidar2bev_aug
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
            results["lidar2cam"][view] = (
                    results["lidar2cam"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d


@PIPELINES.register_module()
class BEVDataAug(object):

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            if gt_boxes.shape[-1] > 7:
                gt_boxes[:, 7:] = (
                        rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        if "gt_bboxes_3d" in results:
            gt_bboxes_3d = torch.Tensor(np.array(results["gt_bboxes_3d"]))
        else:
            gt_bboxes_3d = None

        rotate_bda = results["aug_config"]["rotate_bda"]
        scale_bda = results["aug_config"]["scale_bda"]
        flip_dx = results["aug_config"]["flip_dx"]
        flip_dy = results["aug_config"]["flip_dy"]

        bda_mat = torch.zeros(4, 4)  # lidar2bev_aug
        bda_mat[3, 3] = 1
        gt_bboxes_3d, bda_rot = self.bev_transform(gt_bboxes_3d, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        bda_mat_inv = np.linalg.inv(bda_mat)  # bev_aug2lidar

        results["bda"] = bda_mat

        # Adapt GTs
        if "gt_bboxes_3d" in results:
            results['gt_bboxes_3d'] = gt_bboxes_3d.numpy()
        if "gt_map_pts" in results:
            gt_map_pts =  torch.Tensor(results['gt_map_pts'])  # (num_elements, permut, 20, 2)
            gt_map_pts = F.pad(gt_map_pts, (0, 1), value=0)  # (num_elements, permut, 20, 3)
            gt_map_pts = F.pad(gt_map_pts, (0, 1), value=1)  # (num_elements, permut, 20, 4)
            gt_map_pts = (bda_mat @ gt_map_pts.unsqueeze(-1)).squeeze(-1)
            gt_map_pts = gt_map_pts[..., :2]
            results['gt_map_pts'] = gt_map_pts.numpy()

        # Adapt lidar points
        if "points" in results and results["points"] is not None:
            pts = results["points"]
            pts_tensor = pts.tensor  # (N, C), first 3 are xyz in lidar frame
            xyz = pts_tensor[:, :3]
            ones = xyz.new_ones((xyz.shape[0], 1))
            xyz_h = torch.cat([xyz, ones], dim=1)  # (N, 4)
            xyz_h = (bda_mat @ xyz_h.t()).t()
            pts_tensor[:, :3] = xyz_h[:, :3]
            pts.tensor = pts_tensor
            results["points"] = pts

        # Adapt lidar transformation matrices
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                    results["lidar2img"][view] @ bda_mat_inv
            )
            results["lidar2cam"][view] = (
                    results["lidar2cam"][view] @ bda_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ bda_mat_inv

        return results

def revert_bev_transform(gt_bboxes_3d, bda_mat):
    """
    Revert the scaling + rotation + flip augmentation applied by
    BEVDataAug.bev_transform().

    Parameters
    ----------
    gt_bboxes_3d : (N, ≥7) torch.Tensor | np.ndarray
        Augmented boxes:  [x, y, z, dx, dy, dz, heading, (vx, vy, …)]
    bda_mat      : (4, 4) torch.Tensor | np.ndarray
        `results["bda"]` – its upper-left 3 × 3 block is the composite
        flip ◦ scale ◦ rotation matrix used in bev_transform.

    Returns
    -------
    same type as input
        Boxes restored to the original (pre-augmentation) lidar frame.
    """

    # ---- make sure everything is a NumPy array first ------------------------
    to_torch = isinstance(gt_bboxes_3d, torch.Tensor)
    gt_np  = gt_bboxes_3d.detach().cpu().numpy() if to_torch else np.asarray(gt_bboxes_3d)
    bda_np = bda_mat.detach().cpu().numpy()        if isinstance(bda_mat, torch.Tensor) else np.asarray(bda_mat)

    # ---- inverse composite matrix ------------------------------------------
    rot_scl_flip   = bda_np[:3, :3]          # 3×3
    inv_rot_scl    = np.linalg.inv(rot_scl_flip)

    # uniform scale factor (same for x, y, z because scale_mat was isotropic)
    scale_ratio    = np.linalg.norm(rot_scl_flip[:, 0])  # ||first column||

    boxes = gt_np.copy()

    # 1. centres --------------------------------------------------------------
    boxes[:, :3] = (inv_rot_scl @ boxes[:, :3].T).T      # inverse rigid+flip+scale

    # 2. sizes ----------------------------------------------------------------
    boxes[:, 3:6] /= scale_ratio                         # remove isotropic scale

    # 3. headings -------------------------------------------------------------
    # Represent heading as a unit direction vector, invert the transform,
    # then recover atan2.
    heading        = boxes[:, 6]
    dir_aug        = np.stack([np.cos(heading), np.sin(heading), np.zeros_like(heading)], axis=1)
    dir_orig       = (inv_rot_scl @ dir_aug.T).T
    boxes[:, 6]    = np.arctan2(dir_orig[:, 1], dir_orig[:, 0])

    # 4. optional velocities (or any 2-D vectors stored after heading) --------
    if boxes.shape[1] > 7:
        # inv_R2 = inv_rot_scl[:2, :2]
        boxes[:, 7:10] = (inv_rot_scl @ boxes[:, 7:10].T).T

    # ---- return in the same type the user passed in -------------------------
    if to_torch:
        return torch.from_numpy(boxes).to(gt_bboxes_3d.device).type_as(gt_bboxes_3d)
    return boxes


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
