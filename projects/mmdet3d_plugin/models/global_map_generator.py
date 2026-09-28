# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
)
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import open3d.ml.torch as ml3d

from projects.mmdet3d_plugin.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

__all__ = ["GlobalMapGenerator"]



NUSCENES_MAPS = {
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-queenstown",
    "singapore-onenorth",
}

AV2_MAPS = {
    "ATX",  # Austin, Texas
    "DTW",  # Detroit, Michigan
    "MIA",  # Miami, Florida
    "PAO",  # Palo Alto, California
    "PIT",  # Pittsburgh, PA
    "WDC",  # Washington, DC
}

@DETECTORS.register_module()
class GlobalMapGenerator(BaseDetector):
    def __init__(
        self,
        img_norm_cfg,
        voxel_pooling=False,
        voxel_size=0.4,
        voxel_pooling_position_fn="average",
        voxel_pooling_feature_fn="nearest_neighbor",
        point_format="xyzirgbc",
        extra_width_obj_boxes=0.4,
        tile_size=50.0,
        use_cpu=False,
        save_dir=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(GlobalMapGenerator, self).__init__(init_cfg=init_cfg)
        self.img_norm_cfg = img_norm_cfg
        if voxel_pooling:
            self.voxel_size = voxel_size
            self.voxel_pooling = ml3d.layers.VoxelPooling(position_fn=voxel_pooling_position_fn,
                                                          feature_fn=voxel_pooling_feature_fn)
        else:
            self.voxel_pooling = None

        assert point_format in ["xyzirgbc"]
        self.point_format = point_format
        self.extra_width_obj_boxes = extra_width_obj_boxes
        self.tile_size = tile_size
        self.use_cpu = use_cpu
        self.save_dir = save_dir

        self.dumpy_layer = torch.nn.Linear(1, 1)
        self.points_with_feat_seq = None
        self.curr_scene_tokens = None

    def init_weights(self):
        pass

    @force_fp32(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            return self.forward_train(img, return_loss=return_loss, **kwargs)
        else:
            return self.forward_train(img, return_loss=return_loss, **kwargs)

    def forward_train(self, img_batch, return_loss=True, **data):
        batch_size = img_batch.shape[0]
        H, W = img_batch.shape[3:]
        num_cams = img_batch.shape[1]
        device_cuda = img_batch.device
        device = device_cuda
        if self.use_cpu:
            device = torch.device('cpu')
        if self.curr_scene_tokens is None:
            self.curr_scene_tokens = ["" for _ in range(batch_size)]

        batch_indices = []
        if self.save_dir is not None:
            for batch_idx in range(batch_size):
                scene_token = data["img_metas"][batch_idx]["scene_token"]
                dir = os.path.join(os.path.join(self.save_dir, "processed"), str(scene_token))
                # process sequence if this node already processed some of the seq. or dir does not already exist
                if self.curr_scene_tokens[batch_idx] == scene_token or not os.path.isdir(dir):
                    # Create empty dir to let other nodes know that this node is processing this sequence
                    os.makedirs(dir, exist_ok=True)
                    batch_indices.append(batch_idx)
                    self.curr_scene_tokens[batch_idx] = scene_token

        if self.points_with_feat_seq is None:
            self.points_with_feat_seq = [[] for _ in range(batch_size)]
        batch_size = len(batch_indices)
        if batch_size > 0:
            img_batch = img_batch[batch_indices]  # (B, 6, C, H, W)
            img_batch_merge = img_batch.view(-1, 3, H, W)  # (B*6, C, H, W)
            img_rgb_batch = img_batch_merge  # (B*6, C, H, W)
            img_rgb_batch = img_rgb_batch.view(batch_size, num_cams, -1, img_rgb_batch.shape[-2], img_rgb_batch.shape[-1])  # (B, 6, C, H, W)

        for batch_idx in batch_indices:
            # Get xys points and reflectance
            points = data['points'][batch_idx].to(device)  # (P, 5)
            points_reflect = points[:, 3].unsqueeze(1)  # (P, 1)
            points = points[:, :3]  # (P, 3) do not use reflectance and ring index for xyz

            # Project points to image
            points_hom = F.pad(points, (0, 1), value=1)  # (P, 4) homogeneous
            lidar2img = data["projection_mat"][batch_idx].to(device)  # (6, 4, 4)
            lidar2img = lidar2img.unsqueeze(1)  # (6, 1, 4, 4)
            points_hom = points_hom.unsqueeze(0).unsqueeze(-1)  # (1, P, 4, 1)
            points_img = lidar2img @ points_hom  # (6, P, 4, 1)
            points_img = points_img.squeeze(-1)  # (6, P, 4)
            points_img[:, :, :2] /= points_img[:, :, 2:3]

            # Compute mask that masks out projected points that are not within the image
            mask = torch.ones(points_img.shape[0:2], dtype=torch.bool).to(device)  # (6, P)
            mask = torch.logical_and(mask, points_img[:, :, 1] >= 0)
            mask = torch.logical_and(mask, points_img[:, :, 1] <= H-1)
            mask = torch.logical_and(mask, points_img[:, :, 0] >= 0)
            mask = torch.logical_and(mask, points_img[:, :, 0] <= W-1)
            mask = torch.logical_and(mask, points_img[:, :, 2] >= 0.1)  # depth 0.1m

            # Get projected 2d points
            points_img_2d = points_img[:, :, :2].clone()  # (6, P, 2)

            # Normalize points_i_2d_2d to be in the range [-1, 1]
            points_img_2d[:, :, 0] = (points_img_2d[:, :, 0] / (W - 1)) * 2 - 1
            points_img_2d[:, :, 1] = (points_img_2d[:, :, 1] / (H - 1)) * 2 - 1
            grid = points_img_2d.unsqueeze(2)  # (6, num_points, 1, 2)

            img_rgb = img_rgb_batch[batch_idx]

            # Sample img features at projected points
            sampled_rgb = F.grid_sample(img_rgb, grid, mode='bilinear', align_corners=True)  # (6, C, P, 1)
            sampled_rgb = sampled_rgb.squeeze(-1)  # (6, C, P)
            sampled_rgb = sampled_rgb.permute(0, 2, 1)  # (6, P, C)

            # Transform into global coord. system
            T_lidar2global = torch.from_numpy(data["img_metas"][batch_idx]["T_global"]).to(device)  # (4,4)
            points_hom = F.pad(points, (0, 1), value=1).to(torch.float64)  # (P, 4) homogeneous
            points_global = (T_lidar2global @ points_hom.T).T  # (P, 4)
            points_global[:, :3] = points_global[:, :3] / points_global[:, 3:4]
            points_global = points_global[:, :3].to(torch.float32)

            # Concat 3d coords with features in global coord system
            points_with_reflect_rgb = points_global.expand(sampled_rgb.shape[0], -1, -1)  # (6, P, 3)
            points_reflect = points_reflect.expand(sampled_rgb.shape[0], -1, -1) # (6, P, 3)
            points_with_reflect_rgb = torch.cat((points_with_reflect_rgb, points_reflect, sampled_rgb), dim=-1)  # (6, P, 3+C)

            # Create LiDARInstance3DBoxes and which points are within which box
            gt_bboxes_3d = data['gt_bboxes_3d'][
                batch_idx]  # (BB, 9) xyz lwh(x_size,y_size,z_size) yaw vel_x vel_y
            gt_bboxes_3d = gt_bboxes_3d[:, :7]
            # gt_bboxes_3d[:, 3:6] *= 1.0  # object points are not always within bbox
            # LiDARInstance3DBoxes of mmdet3d rotate clockwise while nuScenes rotates counterclockwise
            # see https://github.com/nutonomy/nuscenes-devkit/issues/830
            # In addition, LiDARInstance3DBoxes are rotated by 180 degree (yaw) compared to nuScenes boxes
            gt_bboxes_3d[:, 6] = 2 * torch.pi - gt_bboxes_3d[:, 6] + torch.pi
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
            gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.extra_width_obj_boxes)
            box_idx = gt_bboxes_3d.points_in_boxes(points.to(gt_bboxes_3d.device)).to(device)  # (P)
            gt_bboxes_3d = gt_bboxes_3d.to(device)

            # Store the object class id for each point and -1 for the static env
            instance_id = data["img_metas"][batch_idx]["instance_id"]  # track id of boxes (num_boxes)
            points_class = -1 * torch.ones((points_with_reflect_rgb.shape[0], points_with_reflect_rgb.shape[1]), dtype=torch.int32).to(device)  # (6, P)
            for idx, instance_id_i in enumerate(instance_id):
                class_id = data['gt_labels_3d'][batch_idx][idx].item()
                points_instance_id_i_mask = (box_idx == idx)  # (P)
                points_instance_id_i_mask = points_instance_id_i_mask.expand(num_cams, -1)  # (6, P)
                points_class[points_instance_id_i_mask] = class_id
            points_with_reflect_rgb_class = torch.cat([points_with_reflect_rgb, points_class.unsqueeze(-1)], dim=-1)  # (6, P, 3+C)

            # Mask out points that are not visible in cameras
            points_with_reflect_rgb_class = points_with_reflect_rgb_class[mask].cpu().numpy()  # (P, 3+C)

            # Add point clouds to sequence list
            self.points_with_feat_seq[batch_idx].append(points_with_reflect_rgb_class)

            # self.plot_projected_points(img_batch[batch_idx], points_img, mask)

            is_last = data["img_metas"][batch_idx]["is_last"]
            # Save sequence
            if is_last:

                points_with_feat_acc = np.concatenate([x for x in self.points_with_feat_seq[batch_idx]], axis=0)  # (P, 3+C)
                if self.voxel_pooling is not None:
                    points_with_feat_acc = self.downsample_point_cloud_with_feat(points_with_feat_acc)

                if self.save_dir is not None:
                    map_location = data["img_metas"][batch_idx]["map_location"]
                    scene_token = data["img_metas"][batch_idx]["scene_token"]
                    timestamp = data["img_metas"][batch_idx]["timestamp"]
                    paths = self.save_pointcloud_tiles(
                        point_cloud_with_feat=points_with_feat_acc,
                        tile_size_x=self.tile_size,
                        tile_size_y=self.tile_size,
                        save_dir=self.save_dir,
                        map_location=map_location,
                        scene_token=scene_token,
                        timestamp=str(timestamp),
                    )
                    print(f"Saved {len(paths)} tiles.")

                    # Mark scene as processed
                    dir = os.path.join(os.path.join(self.save_dir, "processed"), str(scene_token))
                    os.makedirs(dir, exist_ok=True)
                    with open(os.path.join(dir, f"done.txt"), 'w') as output:
                        output.write(" ")

                # self.plot_projected_points(img_batch[batch_idx], points_img, mask)
                # pca = self.plot_pca_img_feat(img_feat)
                # self.pca_on_points(points_with_feat_acc, pca)

                self.points_with_feat_seq[batch_idx] = []

        t = torch.zeros((1, 1)).to(device_cuda)
        loss = self.dumpy_layer(t)

        if return_loss:
            return {"dummy_loss": loss}
        else:
            return [{"img_bbox": None} for i in range(batch_size)]

    def downsample_point_cloud_with_feat(self, points_with_feat_acc):
        points_with_feat_acc = torch.tensor(points_with_feat_acc)
        positions = points_with_feat_acc[:, :3]
        features = points_with_feat_acc[:, 3:]
        pooled_positions, pooled_features = self.voxel_pooling(positions, features, voxel_size=self.voxel_size)
        points_with_feat_acc = torch.cat((pooled_positions, pooled_features), dim=1).numpy()
        return points_with_feat_acc

    def _fmt_coord(self, v: float) -> str:
        """Format a coordinate nicely for folder names: integers as '12', others as '12.5'."""
        if np.isfinite(v) and abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        # Limit to a sensible number of decimals to avoid long folder names.
        return f"{v:.4f}".rstrip("0").rstrip(".")

    def save_pointcloud_tiles(
            self,
            point_cloud_with_feat: np.ndarray,
            tile_size_x: float,
            tile_size_y: float,
            save_dir: str,
            map_location: str,
            scene_token: str,
            timestamp: str,
            make_parents: bool = True,
            return_paths: bool = True,
    ) -> Optional[Dict[Tuple[float, float], str]]:
        """
        Split an accumulated nuScenes/av2 point cloud into XY tiles and save each tile separately.

        Parameters
        ----------
        point_cloud_with_feat : np.ndarray
            Array of shape (P, 3 + C). Columns 0:3 are XYZ in *global* coordinates.
            Columns 3: are arbitrary per-point features (optional, variable C).
        tile_size_x : float
            Tile width along the global X axis (meters).
        tile_size_y : float
            Tile height along the global Y axis (meters).
        save_dir : str
            Root directory where tiles will be stored.
        map_location : str
            One of {'boston-seaport','singapore-hollandvillage','singapore-queenstown','singapore-onenorth'}.
        scene_token : str
            scene token for this sequence.
        timestamp : str
            Timestamp (e.g., sample timestamp) identifying this saved sequence.
        make_parents : bool, optional
            If True, create missing directories as needed.
        return_paths : bool, optional
            If True, return a dict mapping (x0, y0) -> saved file path.

        Returns
        -------
        Optional[Dict[Tuple[float, float], str]]
            Dict mapping tile origins (x0, y0) to the saved .npy paths if return_paths=True,
            otherwise None.

        Notes
        -----
        - Tiles use half-open intervals: X in [x0, x0 + tile_size_x), Y in [y0, y0 + tile_size_y).
        - Only non-empty tiles are saved.
        - Files are saved under: save_dir / map_location / "{x0}_{y0}" / "{scene_token}_{timestamp}.npy"
        """
        # --- Basic checks ---
        if point_cloud_with_feat.ndim != 2 or point_cloud_with_feat.shape[1] < 3:
            raise ValueError("point_cloud_with_feat must be (P, 3 + C) with at least 3 columns for XYZ.")
        if tile_size_x <= 0 or tile_size_y <= 0:
            raise ValueError("tile_size_x and tile_size_y must be positive.")
        if map_location not in NUSCENES_MAPS and map_location not in AV2_MAPS:
            raise ValueError(f"map_location '{map_location}' is invalid. Must be one of {NUSCENES_MAPS} or {AV2_MAPS}.")

        P = point_cloud_with_feat.shape[0]
        if P == 0:
            # Nothing to do
            return {} if return_paths else None

        # --- Extract XY (global) ---
        xyz = point_cloud_with_feat[:, :3]
        xs = xyz[:, 0]
        ys = xyz[:, 1]

        # --- Compute integer tile indices via floor division ---
        # ix, iy are the tile grid indices (can be negative); tile origin is ix*tile_size_x, iy*tile_size_y
        ix = np.floor(xs / tile_size_x).astype(np.int64)
        iy = np.floor(ys / tile_size_y).astype(np.int64)

        # --- Group points by (ix, iy) without Python loops over P ---
        # Use a structured array for unique grouping.
        pairs = np.core.records.fromarrays([ix, iy], names="ix,iy")
        unique_pairs, inverse = np.unique(pairs, return_inverse=True)

        # Prepare output mapping
        saved_paths: Dict[Tuple[float, float], str] = {}

        # Ensure base map directory exists
        base_dir = os.path.join(save_dir, map_location)
        if make_parents:
            os.makedirs(base_dir, exist_ok=True)

        # --- Save each non-empty tile ---
        for k in range(len(unique_pairs)):
            # Selection
            sel = (inverse == k)

            # Compute tile origin in *global* coordinates
            ix_k = int(unique_pairs[k].ix)
            iy_k = int(unique_pairs[k].iy)
            x0 = ix_k * tile_size_x
            y0 = iy_k * tile_size_y

            # Folder name is "<x0>_<y0>", where x0,y0 are the tile origins
            x_str = self._fmt_coord(float(x0))
            y_str = self._fmt_coord(float(y0))
            tile_dir = os.path.join(base_dir, f"{x_str}_{y_str}")
            if make_parents:
                os.makedirs(tile_dir, exist_ok=True)

            filename = f"{scene_token}_{timestamp}.npy"
            path = os.path.join(tile_dir, filename)

            # Slice points belonging to this tile and save
            tile_points = point_cloud_with_feat[sel]
            # Enforce half-open bounds exactly (guards against any numerical oddities)
            # Keep only points where x in [x0, x0+tile_size_x) and y in [y0, y0+tile_size_y)
            # This is typically redundant given floor() grouping, but safe.
            mask = (
                    (tile_points[:, 0] >= x0) & (tile_points[:, 0] < x0 + tile_size_x) &
                    (tile_points[:, 1] >= y0) & (tile_points[:, 1] < y0 + tile_size_y)
            )
            tile_points = tile_points[mask]
            if tile_points.shape[0] == 0:
                # After strict bounds, tile ended up empty; skip saving.
                continue

            np.save(path, tile_points)
            saved_paths[(float(x0), float(y0))] = path

        return saved_paths if return_paths else None

    def plot_projected_points(self, img, points_img, mask):
        device = img.device

        img_std = torch.tensor(self.img_norm_cfg["std"]).to(device) / 255.0
        img_mean = torch.tensor(self.img_norm_cfg["mean"]).to(device) / 255.0
        img_unnorm = img * img_std[:, None, None] + img_mean[:, None, None]  # (6, 3, H, W)

        num_cams = img.shape[0]
        for cam_i in range(num_cams):
            points_img_cam_i = points_img[cam_i, :, :3]  # (num_points, 2)
            points_img_cam_i = points_img_cam_i[mask[cam_i]]
            depth_cam_i = points_img_cam_i[:, 2].cpu().numpy()  # (num_points)

            img_cam_i = img_unnorm[cam_i]  # (3, H, W)
            img_cam_i = img_cam_i.cpu().numpy().transpose((1, 2, 0))  # (H, W, 3)
            plt.figure(figsize=(10, 5))
            plt.imshow(img_cam_i)
            plt.scatter(points_img_cam_i[:, 0].cpu().numpy(), points_img_cam_i[:, 1].cpu().numpy(),
                        c=depth_cam_i, cmap='viridis', s=1)
            plt.colorbar(label='Depth')
            plt.show()

    def extract_feat(self, imgs):
        """Extract features from images."""
        raise NotImplementedError

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def forward_test(self, img, **data):
        raise NotImplementedError


