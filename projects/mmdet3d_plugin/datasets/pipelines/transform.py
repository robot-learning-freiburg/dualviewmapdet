# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is partially derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
#
# This source code of class ObjectRangeFilter is derived from mmdet3d
#   https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/mmdet3d/datasets/pipelines/transforms_3d.py
# Copyright (c) 2018-2019 Open-MMLab, licensed under the  Apache License 2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
from typing import Optional, List

import numpy as np
import torch
import open3d.ml.torch as ml3d
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from PIL import Image, ImageOps
from projects.mmdet3d_plugin.ops.visibility_pkg import visibility

from projects.mmdet3d_plugin.core.points.lidar_points import LiDARPoints
from projects.mmdet3d_plugin.core.points.base_points import BasePoints


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(
        self,
        downsample=1,
        max_depth=60,
        occlusion_filter=False,
        occlusion_filter_threshold=2.4,
        occlusion_filter_radius=9,
        pcl_key="points",
        out_key="gt_depth",
        add_xyz: bool = False,
        invalid_value: float = -1.0,
    ):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth
        self.occlusion_filter = occlusion_filter
        self.occlusion_filter_threshold = occlusion_filter_threshold
        self.occlusion_filter_radius = occlusion_filter_radius
        self.pcl_key = pcl_key
        self.out_key = out_key

        self.add_xyz = add_xyz
        self.invalid_value = invalid_value

    def __call__(self, input_dict):
        points = input_dict[self.pcl_key]
        if isinstance(points, LiDARPoints):
            points = points.tensor

        # points_np: (N, 3) in LiDAR frame
        points_np = points.numpy()[..., :3]
        # points_h: (N, 3, 1) for matmul
        points_h = points_np[..., None]

        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points_h, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]

            pts_2d_round = np.round(pts_2d)
            depths_metric = pts_2d[:, 2]  # metric depth used for filtering

            mask = np.logical_and.reduce([
                pts_2d_round[:, 1] >= 0,
                pts_2d_round[:, 1] < H,
                pts_2d_round[:, 0] >= 0,
                pts_2d_round[:, 0] < W,
                pts_2d_round[:, 2] >= 0.1,
                depths_metric <= self.max_depth,
            ])
            U = pts_2d_round[mask, 0].astype(np.int32)
            V = pts_2d_round[mask, 1].astype(np.int32)
            depths = pts_2d[mask, 2]

            # If add_xyz: gather the corresponding LiDAR xyz for the same masked points
            if self.add_xyz:
                xyz = points_np[mask]  # (M, 3)

            # Sort so that nearer points overwrite farther ones (far -> near)
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)

            if self.add_xyz:
                xyz = xyz[sort_idx]  # keep aligned with (U,V,depths)

            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)

                if self.add_xyz:
                    # depth + xyz => (4, h, w)
                    depthxyz = np.ones([4, h, w], dtype=np.float32) * self.invalid_value
                    depthxyz[0, v, u] = depths
                    depthxyz[1, v, u] = xyz[:, 0]
                    depthxyz[2, v, u] = xyz[:, 1]
                    depthxyz[3, v, u] = xyz[:, 2]
                else:
                    depth_map = np.ones([h, w], dtype=np.float32) * self.invalid_value
                    depth_map[v, u] = depths

                if self.occlusion_filter:
                    assert downsample == 1  # need to adapt intrinsics below if other scale is used (not implemented)
                    # Occlusion filter operates on depth only. We post-mask xyz accordingly.
                    depth_map_filter = torch.zeros([h, w], dtype=torch.float)
                    depth_map_filter += 1000.
                    uv = torch.stack([torch.Tensor(u), torch.Tensor(v)], dim=-1)
                    depth_t = torch.Tensor(depths)
                    depth_map_filter = visibility.depth_image(
                        uv.int().contiguous(),
                        depth_t.contiguous(),
                        depth_map_filter,
                        uv.shape[0],
                        w,
                        h,
                    )
                    depth_map_filter[(depth_map_filter == 1000.)] = 0.

                    K = torch.Tensor(input_dict["cam_intrinsic"][i][:3, :3])  # (3,3)
                    intr = K.contiguous().view(-1)  # 9 elems
                    depth_map_no_occlusion = torch.zeros_like(depth_map_filter)
                    depth_map_no_occlusion = visibility.visibility2(
                        depth_map_filter,
                        intr,
                        depth_map_no_occlusion,
                        w,
                        h,
                        self.occlusion_filter_threshold,
                        self.occlusion_filter_radius,
                    )
                    depth_map_no_occlusion = depth_map_no_occlusion.cpu().numpy()  # (h,w), 0 invalid
                    depth_map_no_occlusion[depth_map_no_occlusion == 0] = self.invalid_value

                    if self.add_xyz:
                        # Start from our depth+xyz, then invalidate pixels removed by occlusion filter
                        out = depthxyz
                        keep = depth_map_no_occlusion > 0
                        out[0, :, :] = self.invalid_value
                        out[0, keep] = depth_map_no_occlusion[keep].astype(np.float32)
                        # Invalidate xyz where not kept
                        out[1:, ~keep] = self.invalid_value
                        gt_depth[j].append(out)
                    else:
                        # Keep behavior close to original (but still returns 0 invalid)
                        gt_depth[j].append(depth_map_no_occlusion)
                else:
                    gt_depth[j].append(depthxyz if self.add_xyz else depth_map)

        input_dict[self.out_key] = [np.stack(x) for x in gt_depth]
        return input_dict


@PIPELINES.register_module()
class NormalizeMultiScaleDepthMap(object):
    """Normalize a (multi-scale) depth map to [0, 1] and add a validity mask.
    Supports optional additional channels (e.g. xyz).

    Expected input format:
        results[depth_key] is a list over scales.
        Each scale element is either:
          - (N_views, H, W) depth in meters with invalid_value for invalid
          - (N_views, C, H, W) where channel 0 is depth in meters and remaining channels are extra per-pixel attrs
            (e.g. xyz in LiDAR coords), also using invalid_value for invalid pixels.

    Output:
        results[out_key] is a list over scales.
        Each scale element is np.ndarray of shape (N_views, C_out, H, W) where:
            - depth_norm is first channel
            - optional depth Fourier features (if enabled) next
            - optional extra channels (e.g. xyz) next (normalized by max_depth)
            - mask is last channel

    Args:
        depth_key (str): Key of the input depth pyramid.
        out_key (str | None): Key to write output to. If None, overwrite depth_key.
        max_depth (float): Depth range max in meters used for normalization.
        invalid_value (float): Value indicating invalid depth (default: -1.0).
        depth_pos_embed_num_freq (int | None): Number of frequencies to use for positional encoding of depth (if used).
    """

    def __init__(
        self,
        depth_key: str = "pcl_map_depth",
        out_key: str = None,
        max_depth: float = 50.0,
        invalid_value: float = -1.0,
        depth_pos_embed_num_freq: Optional[int] = None,
    ):
        self.depth_key = depth_key
        self.out_key = out_key if out_key is not None else depth_key
        assert max_depth > 0
        self.max_depth = max_depth
        self.invalid_value = invalid_value
        self.depth_pos_embed_num_freq = depth_pos_embed_num_freq

        assert self.max_depth > 0, "max_depth must be > 0"

    def __call__(self, results: dict) -> dict:
        depth_pyr = results[self.depth_key]
        out_pyr = []

        for depth_scale in depth_pyr:
            # Accept:
            #   (H,W) -> (1,H,W)
            #   (N,H,W)
            #   (N,C,H,W)
            if depth_scale.ndim == 2:
                depth_scale = depth_scale[None, ...]  # (1,H,W)

            if depth_scale.ndim == 3:
                # depth only: (N,H,W)
                depth = depth_scale
                extra = None
            elif depth_scale.ndim == 4:
                # depth + extras: (N,C,H,W), depth is channel 0
                depth = depth_scale[:, 0]  # (N,H,W)
                extra = depth_scale[:, 1:] if depth_scale.shape[1] > 1 else None
            else:
                raise AssertionError(
                    "Each scale must be shaped (N,H,W), (H,W), or (N,C,H,W). "
                    f"Got shape {depth_scale.shape}."
                )

            # Valid pixels from depth channel
            valid = depth != self.invalid_value
            depth_norm = np.zeros_like(depth, dtype=np.float32)
            depth_norm[valid] = depth[valid] / self.max_depth

            mask = valid.astype(np.float32)  # (N,H,W)

            # Normalize extra channels (e.g. xyz) by max_depth as well (keeps scale comparable)
            extra_norm = None
            if extra is not None:
                v = valid[:, None, :, :].astype(np.float32)  # (N,1,H,W)
                extra_norm = (extra / self.max_depth) * v

            # Depth Fourier features (only on depth_norm)
            feats = [depth_norm[:, None, :, :]]  # (N,1,H,W)
            if self.depth_pos_embed_num_freq is not None:
                for L in range(self.depth_pos_embed_num_freq):
                    fourier = depth_norm * np.pi * 2 ** L
                    feats.append(np.sin(fourier)[:, None, :, :])
                    feats.append(np.cos(fourier)[:, None, :, :])

            if extra_norm is not None:
                feats.append(extra_norm)

            mask_ch = mask[:, None, :, :]  # (N,1,H,W)

            # Concatenate and mask-out invalid
            out = np.concatenate(feats + [mask_ch], axis=1)  # (N, C_out, H, W)
            out = out * mask_ch  # keep mask channel as 0/1, other channels zeroed where invalid
            out[:, -1] = mask  # restore mask explicitly (since mask_ch * mask_ch = mask anyway)

            out_pyr.append(out)

        results[self.out_key] = out_pyr
        results["depth_norm_cfg"] = dict(
            max_depth=self.max_depth,
            invalid_value=self.invalid_value,
        )
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"depth_key='{self.depth_key}', out_key='{self.out_key}', "
            f"max_depth={self.max_depth}, invalid_value={self.invalid_value}, "
            f"depth_pos_embed_num_freq={self.depth_pos_embed_num_freq})"
        )


@PIPELINES.register_module()
class AddNearestDepthSpreadAndDistToValid(object):
    """
    Adds two extra channels to a normalized (multi-scale) depth tensor produced by NormalizeMultiScaleDepthMap:

      1) depth_nearest (no blur):
         For each pixel, take the nearest valid pixel (mask==1) and copy its depth value,
         but only if the nearest valid pixel is within max_dist pixels.
         Otherwise depth_nearest = 0.

      2) dist_to_valid (normalized):
         dist = distance to nearest valid pixel (pixels)
         dist = clip(dist, 0, max_dist)
         dist_norm = dist / max_dist

    Input format per scale:
      results[depth_key][s] is np.ndarray (N_views, C, H, W)
        - channel 0: depth_norm in [0,1] (0 for invalid)
        - last channel: mask in {0,1}
        - any channels in between are preserved

    Output format per scale:
      (N_views, C+2, H, W), inserting the two new channels right before the mask:
        [... existing non-mask channels ..., depth_nearest, dist_to_valid, mask]
    """

    def __init__(
        self,
        depth_key: str = "pcl_map_depth",
        out_key: Optional[str] = None,
        max_dist: float = 20.0,
    ):
        self.depth_key = depth_key
        self.out_key = out_key if out_key is not None else depth_key

        assert max_dist > 0, "max_dist must be > 0"
        self.max_dist = float(max_dist)

        from scipy.ndimage import distance_transform_edt
        self._distance_transform_edt = distance_transform_edt

    def _nearest_spread_and_dist(self, depth_norm: np.ndarray, valid: np.ndarray):
        """
        depth_norm: (H,W) float32, normalized depth (0 where invalid)
        valid:      (H,W) bool, True where valid

        returns:
          depth_nearest: (H,W) float32
          dist_norm:     (H,W) float32 in [0,1]
        """
        H, W = depth_norm.shape
        invalid = ~valid

        # distance to nearest valid (computed on invalid mask)
        # return_indices gives indices of nearest "background" in 'invalid'
        # Here: invalid is True where invalid; background (False) is valid pixels.
        dist, (iy, ix) = self._distance_transform_edt(
            invalid, return_distances=True, return_indices=True
        )
        dist = dist.astype(np.float32)

        # nearest valid depth for every pixel is depth_norm[iy, ix]
        depth_nearest = depth_norm[iy, ix].astype(np.float32)


        # Enforce that valid pixels keep their own depth (good hygiene)
        depth_nearest[valid] = depth_norm[valid]

        # Apply radius: only spread within max_dist
        within = dist <= self.max_dist
        depth_nearest = depth_nearest * within.astype(np.float32)

        # dist normalized (clipped)
        dist_clip = np.clip(dist, 0.0, self.max_dist)
        dist_norm = (dist_clip / self.max_dist).astype(np.float32)

        return depth_nearest.astype(np.float32), dist_norm

    def __call__(self, results: dict) -> dict:
        pyr: List[np.ndarray] = results[self.depth_key]
        out_pyr: List[np.ndarray] = []

        for scale in pyr:
            assert scale.ndim == 4, f"Expected (N_views,C,H,W), got {scale.shape}"
            x = scale.astype(np.float32, copy=False)
            N, C, H, W = x.shape
            assert C >= 2, "Need at least [depth, mask] channels."

            depth = x[:, 0:1, :, :]     # (N,1,H,W) normalized depth
            mask = x[:, -1:, :, :]      # (N,1,H,W) validity mask
            non_mask = x[:, :-1, :, :]  # (N,C-1,H,W) keep everything except mask

            depth_nearest_ch = np.zeros((N, 1, H, W), dtype=np.float32)
            dist_ch = np.zeros((N, 1, H, W), dtype=np.float32)

            for i in range(N):
                valid = (mask[i, 0] > 0.5)             # (H,W) bool
                d0 = depth[i, 0].astype(np.float32)    # (H,W)

                dn, distn = self._nearest_spread_and_dist(d0, valid)
                depth_nearest_ch[i, 0] = dn
                dist_ch[i, 0] = distn

            # Insert: [..., depth_nearest, dist_to_valid, mask]
            out = np.concatenate([non_mask, depth_nearest_ch, dist_ch, mask], axis=1).astype(np.float32)

            out_pyr.append(out)

        results[self.out_key] = out_pyr
        results["depth_nearest_dist_cfg"] = dict(
            max_dist=self.max_dist,
            inserted_channels=["depth_nearest", "dist_to_valid"],
            mask_is_last=True,
        )
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"depth_key='{self.depth_key}', out_key='{self.out_key}', "
            f"max_dist={self.max_dist})"
        )



@PIPELINES.register_module()
class VoxelizePointCloud(object):
    """Voxelize a point cloud into a sparse voxel set (unique occupied voxels).

    This pipeline step converts XYZ points into **voxel-space coordinates**
    (continuous indices in the voxel grid), filters points outside the configured
    grid bounds, and then **pools points per voxel** using Open3D-ML's
    `VoxelPooling` (average position + average features per voxel).

    Expected input format:
        results[pcl_key]:
            - torch.Tensor of shape (N, 3 + F), where the first 3 columns are
              xyz (meters) in the current LiDAR frame, and the remaining columns
              are optional point features

    Output:
        results["voxel_coords"]:
            torch.Tensor of shape (P, 3) with pooled voxel-space coordinates
            (float). These are *not* guaranteed to be integers because the pooled
            position is an average of all points that fell into the voxel.
        results["voxel_feats"]:
            torch.Tensor of shape (P, 3 + F') where the first 3 channels are the
            **normalized pooled voxel coordinates** (divided by grid_size),
            concatenated with pooled features (if any).

    Args:
        grid_config: defines voxel grid space.
        pcl_key (str): Key in `results` where the input point cloud is stored.
    """

    def __init__(
        self,
        grid_config,
        pcl_key: str = "pcl_map",
    ):
        self.grid_config = grid_config
        self.create_grid_infos(**grid_config)
        self.pcl_key = pcl_key
        self.voxel_pooling = ml3d.layers.VoxelPooling(position_fn='average', feature_fn='average')

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def __call__(self, results: dict) -> dict:
        """Run voxelization and attach sparse voxel outputs to `results`."""
        points = results[self.pcl_key]
        if isinstance(points, LiDARPoints):
            points = points.tensor

        points_lidar = points[..., :3]  # get xyz

        # convert coordinates into the voxel space coordinates
        points_voxel = ((points_lidar - self.grid_lower_bound.to(points_lidar)) /
                        self.grid_interval.to(points_lidar))

        # filter out points that are outside of the voxel grid volume
        self.grid_size = self.grid_size.to(points_lidar.device)
        kept = (points_voxel[:, 0] >= 0) & (points_voxel[:, 0] < self.grid_size[0]) & \
               (points_voxel[:, 1] >= 0) & (points_voxel[:, 1] < self.grid_size[1]) & \
               (points_voxel[:, 2] >= 0) & (
                           points_voxel[:, 2] < self.grid_size[2])
        points_voxel = points_voxel[kept]
        #points_feat = points_feat[kept] TODO: change this if features are added to points
        # Create dummy features for now
        points_feat = torch.zeros(points_voxel.shape[0], 0, device=points_voxel.device)

        if points_voxel.numel() == 0:
            results["voxel_coords"] = points_voxel.new_zeros((0, 3))
            results["voxel_feats"] = points_voxel.new_zeros((0, 3))  # coords-only
            return results

        # Voxelization
        points_voxel_pooled, points_feat_pooled = self.voxel_pooling(points_voxel, points_feat.to(torch.float32),
                                                                voxel_size=1.0)
        points_voxel_pooled_center = points_voxel_pooled.long()

        results["voxel_coords"] = points_voxel_pooled  # (P, 3)
        # Concat normalized coords to features
        points_voxel_pooled_normalized = points_voxel_pooled / self.grid_size
        # PreSight further adds a hit counter: https://github.com/yuantianyuan01/PreSight/blob/main/online-mapping/plugin/datasets/pipelines/prior_points.py#L130
        results["voxel_feats"] = torch.cat([points_voxel_pooled_normalized, points_feat_pooled], dim=1)  # (P, C)

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"grid_config='{self.grid_config}', pcl_key='{self.pcl_key}')"
        )


@PIPELINES.register_module()
class PclMapToLiDARPoints:
    """Convert point cloud to LiDARPoints mmdet class.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        pcl_map = data["pcl_map"]
        points = LiDARPoints(pcl_map, points_dim=pcl_map.shape[1])
        if "points" in data:  # not there during test
            data['points_lidar'] = data["points"] # points of current lidar scan TODO: not pretty solution since data changes
        data["points"] = points
        return data


@PIPELINES.register_module()
class PointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range, points_keys=["points"]):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.points_keys = points_keys

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        for key in self.points_keys:
            points = data[key]
            points_mask = points.in_range_3d(self.pcd_range)
            clean_points = points[points_mask]
            data[key] = clean_points
        return data


@PIPELINES.register_module()
class PointShuffle:
    def __init__(self, points_keys=["points"]):
        self.points_key = points_keys

    def __call__(self, data):
        for key in self.points_key:
            data[key].shuffle()
        return data


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        input_dict["lidar2cam"] = np.float32(
            np.stack(input_dict["lidar2cam"])
        )
        input_dict["cam2lidar"] = np.float32(
            np.stack(np.linalg.inv(input_dict["lidar2cam"]))
        )
        if "bda" in input_dict:
            input_dict["bda"] = np.float32(input_dict["bda"])
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = np.abs(input_dict["cam_intrinsic"][..., 0, 0])
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        if "pcl_map_depth" in input_dict:
            pcl_map_depth = input_dict["pcl_map_depth"][0]  # only single scale
            input_dict["pcl_map_depth"] = DC(to_tensor(pcl_map_depth), stack=True)

        if "voxel_coords" in input_dict and "voxel_feats" in input_dict:
            input_dict["voxel_coords"] = DC(input_dict["voxel_coords"])
            input_dict["voxel_feats"] = DC(input_dict["voxel_feats"])

        if "points" in input_dict:
            assert isinstance(input_dict["points"], BasePoints)
            input_dict["points"] = DC(input_dict["points"].tensor)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


# Copied from https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/mmdet3d/datasets/pipelines/transforms_3d.py
@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        # bev_range = self.pcd_range[[0, 1, 3, 4]] # mmdet3d
        bev_range = self.pcd_range[[1, 0, 4, 3]] # nuScenes

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        mask = (
                (gt_bboxes_3d[:, 0] > bev_range[0])
                & (gt_bboxes_3d[:, 1] > bev_range[1])
                & (gt_bboxes_3d[:, 0] < bev_range[2])
                & (gt_bboxes_3d[:, 1] < bev_range[3])
        )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        # limit rad to [-pi, pi]
        # gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str

