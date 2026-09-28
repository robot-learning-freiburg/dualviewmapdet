# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# This source code of class LoadPointsFromMultiSweeps and LoadBEVSegmentation are derived from BEVFusion
#   https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/datasets/pipelines/loading.py
# Copyright (c) 2023 MIT HAN Lab licensed under the Apache License 2.0,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import itertools
import os
from typing import Tuple, Dict, Any, List, Optional
from collections import OrderedDict

import math
import numpy as np
import mmcv
import pandas as pd
from mmdet.datasets.builder import PIPELINES
import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import open3d as o3d
import hashlib

from projects.mmdet3d_plugin.core.points import BasePoints
from projects.mmdet3d_plugin.core.points import get_points_type

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [img.astype(np.float32) for img in img]
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img_i for img_i in img]
        results["img_shape"] = [img_i.shape for img_i in img]
        results["ori_shape"] = [img_i.shape for img_i in img]
        # Set initial values for default meta_keys
        results["pad_shape"] = [img_i.shape for img_i in img]
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        file_client_args=dict(backend="disk"),
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if str(pts_filename).endswith(".feather"): # Argoverse2
            pcl = pd.read_feather(pts_filename)
            points = np.column_stack([pcl[c].to_numpy(np.float32)
                             for c in ('x', 'y', 'z', 'intensity', 'laser_number', 'offset_ns')])
            return points
        else:
            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)
            try:
                pts_bytes = self.file_client.get(pts_filename)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
            except ConnectionError:
                mmcv.check_file_exist(pts_filename)
                if pts_filename.endswith(".npy"):
                    points = np.load(pts_filename)
                else:
                    points = np.fromfile(pts_filename, dtype=np.float32)

            return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
             points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )

        results["points"] = points
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
            self,
            sweeps_num=10,
            load_dim=5,
            use_dim=[0, 1, 2, 4],
            pad_empty_sweeps=False,
            remove_close=False,
            test_mode=False,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=2.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                # don't allow to sample the earliest frame, match with Tianwei's implementation.
                choices = np.random.choice(
                    len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim]
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                        points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadGlobalMap:
    """
    Load point clouds from a tile-based global map around the current ego pose.

    This pipeline step:
      1) Computes the set of tile folder IDs covering a square region around the current ego xy-position.
      2) Loads all `.npy` point cloud files inside those tiles.
      3) Filters out:
         - points from the current scene (`scene_token`)
         - scenes that are too close in time (`min_timediff`)
         - scenes not in the current split to exclude val scenes during training (`only_from_split`)
      4) Optionally applies a refined global alignment transform (old-global -> refined-global) to correct wrong global poses.
      5) Optionally removes object points (keeps only static environment) based on class channel `c`.
      6) Transforms the resulting global points into the current LiDAR frame and writes `results["map_pcl"]`.

    Expected directory structure (tile-based global map):
        save_dir /
            map_location /
                "{x0}_{y0}" /
                    "{scene_token}_{timestamp}.npy"

    Refined pose file (optional):
        lidar_pose /
            map_location /
                "T_globalold_to_globalstar.npy"
    where the file contains a dict-like object: {scene_token: (4,4) np.ndarray}.

    Notes:
      - If `with_dynamic_objects=False`, `point_format_input` must include the class channel `'c'`
        so static and dynamic points can be selected.
    """

    def __init__(
        self,
        save_dir: str,
        lidar_pose: Optional[str] = None,
        map_range: float = 50.0,          # meters (radius-like parameter, used to determine tile coverage)
        tile_size: float = 50.0,          # meters (edge length of a tile)
        point_format_input: str = "xyzirgbc",
        point_format_output: str = "xyzi",
        with_dynamic_objects: bool = False,
        static_objects_class_ids: Optional[List[int]] = None,
        rand_sampling_max_points: Optional[int] = None,
        voxel_size: Optional[float] = 0.2,
        sor_k: Optional[int] = 20,
        sor_std: Optional[float] = 0.8,
        min_timediff: float = 120.0,      # seconds
        only_from_split: bool = True,
        cache_size_tiles: int = 50,
        cache_tiles_to_disk: bool = True,
        load_largest_k: Optional[int] = None,
    ) -> None:
        """
        Args:
            save_dir: Root directory containing the tile-based global map.
            lidar_pose: Optional root directory containing refined pose transforms
                (old-global -> refined-global). If provided, will load
                `lidar_pose/<map_location>/T_globalold_to_globalstar.npy`.
            map_range: Spatial range (in meters) used to determine which tiles to load
                around the current ego pose (xy). Internally converted to a tile radius.
            tile_size: Tile edge length in meters. Tiles are addressed by their origin
                `{x0}_{y0}`, where `x0 = ix * tile_size`, `y0 = iy * tile_size`.
            point_format_input: String describing the input point layout in the `.npy` files.
                Must start with `"xyz"`. Example: `"xyzirgbc"`.
                If `with_dynamic_objects=False`, it must also contain `'c'` for class-based filtering.
            point_format_output: String describing the output point layout written to `results["map_pcl"]`.
                Must start with `"xyz"`. Example: `"xyzi"`.
            with_dynamic_objects: If True, keep dynamic object points. If False, keep only static environment points
                based on the `'c'` object class ID channel (< 0).
            static_objects_class_ids: Optional list of class IDs of static objects to keep.
            rand_sampling_max_points: If provided, randomly selects rand_sampling_max_points points before voxel downsampling.
            voxel_size: Optional voxel size in meters. If provided, will be used to downsample the point cloud.
            sor_k: Optional number of neighbors around point for statistical outlier removal (SOR).
            sor_std: Standard deviation ratio for SOR.
            min_timediff: Minimum absolute time difference (seconds) between the current sample
                (`results["timestamp"]`) and map tile samples to include.
            only_from_split: If True, only uses scenes listed in `results["scene_tokens_split"]`.
            cache_size_tiles: Cache size for loaded tiles.
            cache_tiles_to_disk: If processed tiles should be cached on disk.
            load_largest_k: If provided, loads only the k largest map point clouds from a tile.
        """
        self.save_dir = save_dir
        self.lidar_pose = lidar_pose
        self.map_range = map_range
        self.tile_size = tile_size
        assert point_format_input[:3] == "xyz"
        self.point_format_input = point_format_input
        assert point_format_output[:3] == "xyz"
        self.point_format_output = point_format_output
        self.cache_size_tiles = cache_size_tiles
        self.cache_tiles_to_disk = cache_tiles_to_disk
        self.load_largest_k = load_largest_k
        self.with_dynamic_objects = with_dynamic_objects
        # If we want to remove dynamic objects, we need the class channel 'c'
        if not self.with_dynamic_objects:
            assert "c" in point_format_input
        self.static_objects_class_ids = static_objects_class_ids
        if self.static_objects_class_ids is None:
            self.static_objects_class_ids = []
        self.rand_sampling_max_points = rand_sampling_max_points
        self.voxel_size = voxel_size
        if self.voxel_size is not None:
            self.voxel_pooling = ml3d.layers.VoxelPooling(
                position_fn="average",
                feature_fn="nearest_neighbor",
            )
        self.sor_k = sor_k
        self.sor_std = sor_std
        self.min_timediff = min_timediff
        self.only_from_split = only_from_split

        # On-disk cache for processed tiles (shared across workers)
        if self.cache_tiles_to_disk:
            self._tile_pcl_cache_dir = os.path.join(self.save_dir, "tile_pcl_cache")
            os.makedirs(self._tile_pcl_cache_dir, exist_ok=True)

            # Processing signature (to avoid mixing caches across different preprocessing settings)
            proc_str = (
                    f"pfi={self.point_format_input}|pfo={self.point_format_output}|"
                    f"dyn={int(self.with_dynamic_objects)}|static_ids={','.join(map(str, self.static_objects_class_ids))}|"
                    f"rand={self.rand_sampling_max_points}|vox={self.voxel_size}|sor_k={self.sor_k}|sor_std={self.sor_std}|"
                    f"min_td={self.min_timediff}|only_split={int(self.only_from_split)}|load_k={self.load_largest_k}"
            )
            self._proc_sig = hashlib.md5(proc_str.encode("utf-8")).hexdigest()

        # LRU cache (processed): key=(map_location, tile_id, selection_sig) -> merged+processed tile tensor
        # selection_sig encodes which tile files survived the per-frame filters (scene/timediff/split).
        self._tile_pcl_cache: "OrderedDict[Tuple[str, str, str], torch.Tensor]" = OrderedDict()

        # Cache refined transforms per map_location (avoid reloading dict every frame)
        self._pose_dict_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _fmt_coord(self, v: float) -> str:
        """
        Format a coordinate for folder names: integers as '12', others as '12.5' etc.

        Args:
            v: Coordinate value.

        Returns:
            A compact string representation suitable for folder names.
        """
        if np.isfinite(v) and abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        # Limit to a sensible number of decimals to avoid long folder names.
        return f"{v:.4f}".rstrip("0").rstrip(".")

    def get_tiles(self, lidar_pose_xy: np.ndarray, map_range: float, tile_size: float) -> List[str]:
        """
        Compute all tile IDs that cover a square region around the given xy-position.

        The tile grid index is computed by floor-division:
            ix = floor(x / tile_size), iy = floor(y / tile_size)
        The returned tile IDs are the tile origin coordinates formatted as:
            "{x0}_{y0}" with x0 = (ix + di) * tile_size, y0 = (iy + dj) * tile_size

        Args:
            lidar_pose_xy: LiDAR ego pose xy position in the (old) global frame, shape (2,).
            map_range: Coverage range in meters.
            tile_size: Tile edge length in meters.

        Returns:
            List of tile folder IDs: ["x0_y0", ...].
        """
        # --- Compute integer tile index of tile with the ego car via floor division ---
        # ix, iy are the tile grid indices (can be negative); tile origin is ix*tile_size_x, iy*tile_size_y
        cx, cy = float(lidar_pose_xy[0]), float(lidar_pose_xy[1])
        ix = int(np.floor(cx / tile_size))
        iy = int(np.floor(cy / tile_size))

        # How many tiles to expand in each direction
        tiles_rad = int(math.ceil(map_range / tile_size))  # ensures coverage

        tile_ids = []
        for di, dj in itertools.product(range(-tiles_rad, tiles_rad + 1),
                                        range(-tiles_rad, tiles_rad + 1)):
            x0 = (ix + di) * tile_size
            y0 = (iy + dj) * tile_size
            x_str = self._fmt_coord(float(x0))
            y_str = self._fmt_coord(float(y0))
            tile_ids.append(f"{x_str}_{y_str}")

        return tile_ids

    def transform_points(self, points: torch.Tensor, T_mat: torch.Tensor) -> torch.Tensor:
        """
        Apply a 4x4 homogeneous transform to 3D points.

        Args:
            points: Tensor of shape (P, 3) in float32/float64.
            T_mat: Homogeneous transform of shape (4, 4). Expected to map points as:
                p' = T_mat * [p, 1]^T

        Returns:
            Transformed points of shape (P, 3) as float32.
        """
        points_hom = F.pad(points, (0, 1), value=1).to(torch.float64)  # (P, 4) homogeneous
        points_transformed = (T_mat.to(torch.float64) @ points_hom.T).T  # (P, 4)
        points_transformed[:, :3] = points_transformed[:, :3] / points_transformed[:, 3:4]
        points_transformed = points_transformed[:, :3].to(torch.float32)  # (P, 3)
        return points_transformed

    def _get_pose_dict(self, map_location: str) -> Optional[Dict[str, np.ndarray]]:
        """Load + cache old-global -> globalstar transform dict for this map_location (if enabled)."""
        if self.lidar_pose is None:
            return None
        if map_location in self._pose_dict_cache:
            return self._pose_dict_cache[map_location]

        lidar_pose_path = os.path.join(self.lidar_pose, map_location, "T_globalold_to_globalstar.npy")
        pose_dict = np.load(lidar_pose_path, allow_pickle=True).item()
        self._pose_dict_cache[map_location] = pose_dict
        return pose_dict

    def _downsample_point_cloud_with_feat(self, points_with_feat: torch.Tensor, voxel_size: float) -> torch.Tensor:
        positions = points_with_feat[:, :3]
        features = points_with_feat[:, 3:]
        pooled_positions, pooled_features = self.voxel_pooling(
            positions,
            features,
            voxel_size=voxel_size,
        )
        points_with_feat_acc = torch.cat((pooled_positions, pooled_features), dim=1)
        return points_with_feat_acc

    def _selection_signature(self, selected: List[Tuple[str, float]]) -> str:
        """
        Build a stable signature for the subset of tile entries that survived filtering.
        This lets us cache the *processed* merged tile pcl for streaming training.
        """
        # Use integer milliseconds to stabilize float formatting across loads
        parts = [f"{st}:{int(round(ts * 1000.0))}" for (st, ts) in selected]
        s = "|".join(parts).encode("utf-8")
        return hashlib.md5(s).hexdigest()

    def _disk_cache_path(self, map_location: str, tile_id: str, selection_sig: str) -> str:
        """Path for the processed-tile tensor cached on disk."""
        d = os.path.join(self._tile_pcl_cache_dir, map_location, tile_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self._proc_sig}_{selection_sig}.pt")

    def _get_tile_pcl_cached(
        self,
        map_location: str,
        base_dir: str,
        tile_id: str,
        pose_dict: Optional[Dict[str, np.ndarray]],
        scene_token: str,
        timestamp: float,
        scene_tokens_split: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Return a *processed* tile point cloud for this frame context, cached by (tile_id + selection signature).

        Processing (done once per selection_sig, then cached):
          - merge selected tile files
          - mask out objects (if enabled)
          - voxel downsample (if enabled)
          - statistical outlier removal (if enabled)

        Returned tensor is in:
          - old global frame (pose_dict is None)
          - refined globalstar frame (pose_dict provided, applied during cache miss processing)
        """
        tile_path = os.path.join(base_dir, tile_id)
        if not os.path.exists(tile_path):
            return None

        # --- Apply the same per-frame filtering, but only to decide which entries contribute ---
        selected_meta: List[Tuple[str, float]] = []
        selected_paths: List[Tuple[str, float, str]] = []

        for fn in os.listdir(tile_path):
            if not fn.endswith(".npy"):
                continue
            fpath = os.path.join(tile_path, fn)
            if not os.path.isfile(fpath):
                continue

            base = os.path.basename(fn).replace(".npy", "")
            tile_scene_token, tile_ts_str = base.split("_")
            tile_ts = float(tile_ts_str)

            # Filter tiles of current scene out
            if tile_scene_token == scene_token:
                continue
            # Filter tiles of scenes out if they are too close in time
            if abs(tile_ts - timestamp) < self.min_timediff:
                continue
            # Use scene only if it is in the current dataset split
            if self.only_from_split and (tile_scene_token not in scene_tokens_split):
                continue

            selected_meta.append((tile_scene_token, tile_ts))
            selected_paths.append((tile_scene_token, tile_ts, fpath))

        if not selected_paths:
            return None

        if self.load_largest_k is not None and self.load_largest_k > 0:
            if len(selected_paths) > self.load_largest_k:
                # Attach file size for sorting (descending)
                selected_paths_with_size = []
                for tile_scene_token, tile_ts, fpath in selected_paths:
                    fsize = os.path.getsize(fpath)
                    selected_paths_with_size.append((fsize, tile_scene_token, tile_ts, fpath))

                # Sort by size desc
                selected_paths_with_size.sort(
                    key=lambda x: (-x[0], x[1], x[2])
                )

                # Keep only largest-K
                selected_paths = [
                    (tile_scene_token, tile_ts, fpath)
                    for (fsize, tile_scene_token, tile_ts, fpath)
                    in selected_paths_with_size[: self.load_largest_k]
                ]

            # IMPORTANT: selection_sig must reflect the reduced selection set
        selected_meta = [(tile_scene_token, tile_ts) for (tile_scene_token, tile_ts, _) in selected_paths]

        selected_meta.sort(key=lambda x: (x[0], x[1]))  # Sort by scene_token, then timestamp
        selection_sig = self._selection_signature(selected_meta)
        key = (map_location, tile_id, selection_sig)
        if key in self._tile_pcl_cache:
            self._tile_pcl_cache.move_to_end(key)
            return self._tile_pcl_cache[key]

        if self.cache_tiles_to_disk:
            # Disk cache lookup (shared across workers)
            disk_path = self._disk_cache_path(map_location, tile_id, selection_sig)
            if os.path.isfile(disk_path):
                pcl = torch.load(disk_path, map_location="cpu")
                self._tile_pcl_cache[key] = pcl
                self._tile_pcl_cache.move_to_end(key)
                while len(self._tile_pcl_cache) > self.cache_size_tiles:
                    self._tile_pcl_cache.popitem(last=False)
                return pcl

        # Cache miss: load + (optionally) transform + merge only the selected tile files
        selected_paths.sort(key=lambda x: (x[0], x[1]))
        selected_entries: List[torch.Tensor] = []
        for tile_scene_token, tile_ts, fpath in selected_paths:
            tile_np = np.load(fpath)
            tile_t = torch.from_numpy(tile_np)

            # Transform xyz points to refined global lidar pose once when caching the tile file
            if pose_dict is not None:
                T_globalold2globalstar = torch.from_numpy(pose_dict[tile_scene_token])
                tile_t = tile_t.clone()
                tile_t[:, :3] = self.transform_points(tile_t[:, :3], T_globalold2globalstar)  # (P, 3) xyz

            selected_entries.append(tile_t)

        # Load point clouds of map tiles and merge them (within this tile, after filtering)
        pcl = torch.cat(selected_entries, dim=0)

        # Mask out objects
        if not self.with_dynamic_objects:
            points_class = pcl[:, self.point_format_input.index("c")]
            static_env_mask = points_class < 0
            for static_objects_class_id in self.static_objects_class_ids:
                static_env_mask = torch.logical_or(static_env_mask, points_class == static_objects_class_id)
            pcl = pcl[static_env_mask]

        # Optionally downsample point cloud for efficiency
        if self.voxel_size is not None:
            if self.rand_sampling_max_points is not None and pcl.shape[0] > self.rand_sampling_max_points:
                idx = np.random.choice(pcl.shape[0], self.rand_sampling_max_points, replace=False)
                pcl = pcl[idx]
            pcl = self._downsample_point_cloud_with_feat(pcl, voxel_size=self.voxel_size)

        # Outlier filtering to remove noise, especially useful when we have many traversals of the same location
        if self.sor_k is not None and self.sor_std is not None and pcl.numel() > 0:
            pts_np = pcl[:, :3].numpy().astype(np.float64)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_np))
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=self.sor_k, std_ratio=self.sor_std)

            ind_t = torch.as_tensor(ind, device=pcl.device, dtype=torch.long)

            # Keep features aligned with the filtered points
            pcl = pcl[ind_t].clone()

            # Keep xyz identical to Open3D output (important because Open3D is the source of truth for filtered xyz)
            pcl[:, :3] = torch.from_numpy(np.asarray(pcd.points)).to(device=pcl.device, dtype=torch.float32)

        # Store processed tile pcl in LRU cache
        self._tile_pcl_cache[key] = pcl
        self._tile_pcl_cache.move_to_end(key)

        # Persist processed tile to disk (atomic write)
        if self.cache_tiles_to_disk:
            tmp_path = disk_path + f".tmp.{os.getpid()}"
            torch.save(pcl.cpu(), tmp_path)
            os.replace(tmp_path, disk_path)

        while len(self._tile_pcl_cache) > self.cache_size_tiles:
            self._tile_pcl_cache.popitem(last=False)

        return pcl

    def __call__(self, results):
        """
        Load and assemble global-map points around the current sample and attach them to `results`.

        Required keys in `results`:
            - "lidar2global": (4,4) np.ndarray, current LiDAR pose in old global frame.
            - "map_location": str, map/city identifier used as subfolder under `save_dir`.
            - "scene_token": str, identifier of the current scene/sequence.
            - "timestamp": float, timestamp (seconds) of the current sample.
            - "scene_tokens_split": List[str], scene tokens belonging to the current dataset split
              (only used if `only_from_split=True`).

        Optional keys in `results`:
            - "bda": torch.Tensor or np.ndarray (4,4), BEV data augmentation matrix. If present, the
              spatial range is divided by the inferred scale.
              (Used to keep the loaded map coverage consistent with BEV scaling.)

        Added/updated keys:
            - "pcl_map": torch.Tensor of shape (N, 3 + F), where
                - the first 3 columns are xyz in the current LiDAR frame
                - the remaining columns are features selected according to `point_format_output`

        Returns:
            The updated `results` dict.
        """
        lidar2global = results["lidar2global"]
        if "bda" in results:
            bda = results["bda"].numpy()  # lidar2bev_aug
        else:
            bda = torch.eye(4).numpy()  # lidar2bev_aug

        scale = np.linalg.norm(
            np.dot(bda[:3, :3].astype(np.float32), np.array([1, 0, 0]))
        )  # scale of BEV data aug
        lidar_pose_xy = lidar2global[:2, 3]

        # Get all relevant tile ids that cover the region around the lidar
        tile_ids = self.get_tiles(lidar_pose_xy, self.map_range / scale, self.tile_size)

        # Get all pcl files of the relevant tiles
        map_location = results["map_location"]
        scene_token = results["scene_token"]
        timestamp = results["timestamp"]
        pose_dict = self._get_pose_dict(map_location)  # None if lidar_pose disabled

        # Use scene only if it is in the current dataset split
        # (only used if only_from_split=True)
        scene_tokens_split = results["scene_tokens_split"] if self.only_from_split else []

        map_pcl: List[torch.Tensor] = []
        base_dir = os.path.join(self.save_dir, map_location)
        assert os.path.isdir(base_dir), f"Map directory {base_dir} does not exist!"

        for tile_id in tile_ids:
            # Cache *processed* per-tile point clouds (merge + mask + downsample + SOR),
            # so we do not repeat heavy preprocessing for each frame.
            tile_pcl = self._get_tile_pcl_cached(
                map_location=map_location,
                base_dir=base_dir,
                tile_id=tile_id,
                pose_dict=pose_dict,
                scene_token=scene_token,
                timestamp=timestamp,
                scene_tokens_split=scene_tokens_split,
            )
            if tile_pcl is not None and tile_pcl.numel() > 0:
                map_pcl.append(tile_pcl)

        # Load point clouds of map tiles and merge them
        if map_pcl:
            pcl = torch.cat(map_pcl, dim=0)
        else:
            pcl = torch.empty(0, len(self.point_format_input))

        # Transform point cloud to from global to lidar frame
        points_global = pcl[:, :3]  # (P, 3)
        T_lidar2global = torch.from_numpy(results["lidar2global"])
        if pose_dict is not None:
            T_globalold2globalstar_current = torch.from_numpy(pose_dict[scene_token])
            T_lidar2global = T_globalold2globalstar_current @ T_lidar2global
        T_global2lidar = torch.linalg.inv(T_lidar2global)
        points_lidar = self.transform_points(points_global, T_global2lidar)

        # Convert features to output format
        points_feat = pcl[:, 3:]  # (P, C)
        feat_out_idx = []
        for feat_name in self.point_format_output[3:]:
            assert feat_name in self.point_format_input[3:]
            feat_out_idx.append(self.point_format_input[3:].index(feat_name))
        points_feat = points_feat[:, feat_out_idx]

        results["pcl_map"] = torch.cat([points_lidar, points_feat], dim=1)
        #results["pcl_map"] = torch.zeros((0, results["pcl_map"].shape[-1])) # complete map drop

        return results



@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        map_center_x = xbound[0] + (xbound[1] - xbound[0]) / 2
        map_center_y = ybound[0] + (ybound[1] - ybound[0]) / 2
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.map_center = (map_center_x, map_center_y)
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2global = data['lidar2global']
        if 'bda' in data: # BEV data augmentation
            bda = data['bda'].numpy()  # lidar2bev_aug
        else:
            bda = torch.eye(4).numpy()  # lidar2bev_aug

        map_center = np.array([self.map_center[0], self.map_center[1], 0.0, 1.0])  # 3D homogeneous coordinates
        scale = np.linalg.norm(np.dot(bda[:3, :3].astype(np.float32), np.array([1, 0, 0])))
        map_pose = lidar2global @ map_center.astype(np.float64)
        map_pose = (map_pose[:2] / map_pose[3]).astype(np.float32)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0] / scale, self.patch_size[1] / scale)

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["map_location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_bev_map"] = labels
        return data
