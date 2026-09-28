# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from Far3D
#   https://github.com/megvii-research/Far3D/blob/main/projects/mmdet3d_plugin/datasets/argoverse2_dataset_t.py
# Copyright (c) 2024 MEGVII, licensed under the  Apache License 2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
from typing import Optional
import numpy as np
from mmdet.datasets import DATASETS
from pathlib import Path
import math
from mmcv.parallel import DataContainer as DC
import os

try:
    from .argoverse2_dataset import Argoverse2Dataset
    from av2.evaluation.detection.constants import CompetitionCategories
except ImportError:
    print("WARNING no Argoverse2")

LABEL_ATTR = (
    "tx_m","ty_m","tz_m","length_m","width_m","height_m","qw","qx","qy","qz",
)

@DATASETS.register_module()
class Argoverse2DatasetT(Argoverse2Dataset):
    try:
        CLASSES = tuple(x.value for x in CompetitionCategories)
    except:
        print("WARNING no Argoverse2")
    
    def __init__(self, with_seq_flag=False, sequences_split_num=1, keep_consistent_seq_aug=True, num_frame_losses=1, queue_length=8, random_length=0, interval_test=False, data_aug_conf=None, shuffle=True, drop_last=True, load_ann=True, work_dir=None, keep_scene_tokens_txt: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.data_aug_conf = data_aug_conf
        self.with_seq_flag = with_seq_flag
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.load_ann = load_ann
        self.keep_scene_tokens_txt = keep_scene_tokens_txt

        # Remove samples with missing camera info in test mode
        # Hacky fix to fix this issue: https://github.com/megvii-research/Far3D/issues/26
        before = len(self.data_infos)
        self.data_infos = [
            info for info in self.data_infos
            if 'cam_infos' in info
               and info['cam_infos'] is not None
               and all(ci is not None for ci in info['cam_infos'].values())
        ]
        if len(self.data_infos) < before:
            print(f"[Argoverse2DatasetT] Filtered {before - len(self.data_infos)} "
                  f"samples with missing cameras.")

        # filter by keep list (scene_id == scene_token in your pipeline)
        if self.keep_scene_tokens_txt is not None:
            path = self.keep_scene_tokens_txt
            if not isinstance(path, str) or len(path) == 0:
                raise ValueError("keep_scene_tokens_txt must be a non-empty path string or None.")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"keep_scene_tokens_txt not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                keep = {ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")}

            before_keep = len(self.data_infos)
            self.data_infos = [info for info in self.data_infos if info.get("scene_id") in keep]
            after_keep = len(self.data_infos)
            print(
                f"[Argoverse2DatasetT] keep_scene_tokens_txt: kept {len(keep)} scene tokens, "
                f"filtered samples {before_keep} -> {after_keep} "
                f"({before_keep - after_keep} removed)."
            )

        if interval_test:
            data_infos = self.data_infos
            s1, s2, s3, s4, s5 = data_infos[::5], data_infos[1::5], data_infos[2::5], data_infos[3::5], data_infos[4::5]
            data_infos = s1 + s2 + s3 + s4 + s5 
            self.data_infos = data_infos
        if with_seq_flag:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

        self.scene_tokens = set([info['scene_id'] for info in self.data_infos])

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []
        scene_id = None

        curr_sequence = -1
        for idx in range(len(self.data_infos)):
            if self.data_infos[idx]['scene_id'] != scene_id:
                # Not first frame and # of sweeps is 0 -> new sequence
                scene_id = self.data_infos[idx]['scene_id']
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                
                self.flag = np.array(new_flags, dtype=np.int64)


    def get_city_name_from_sequence_path(self, sequence_path: Path) -> Optional[str]:
        """
        Extract the city name from the *_ground_height_surface____CITY.npy file
        inside the sequence's map/ directory.

        Returns:
            city name (e.g. 'PIT', 'MIA', 'DTW', ...) or None if not found.
        """
        map_dir = sequence_path / "map"
        pattern = "*_ground_height_surface____*.npy"
        candidates = list(map_dir.glob(pattern))
        # take the first file (each sequence has only one)
        fname = candidates[0].name
        city = fname.split("____")[-1].replace(".npy", "")
        return city

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        city_SE3_ego = info['city_SE3_ego_lidar_t']
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = city_SE3_ego.rotation
        transform_matrix[:3, 3] = city_SE3_ego.translation

        ego_pose =  transform_matrix

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        pts_filename = Path(self.data_root) / self.split / info['scene_id']  / 'sensors' /  'lidar' / f"{info['lidar_timestamp_ns']}.feather"
        map_location = self.get_city_name_from_sequence_path(Path(self.data_root) / self.split / info['scene_id'])
        input_dict = dict(
            pts_filename=pts_filename,
            lidar2global=ego_pose,
            token=str(info['lidar_timestamp_ns']),
            scene_token=info['scene_id'],
            timestamp=info['lidar_timestamp_ns'] / 1e9,
            map_location=map_location,
            scene_tokens_split=self.scene_tokens,
        )

        if self.modality['use_camera']:
            image_paths = []
            image_raw_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            city_SE3_ego_lidar_t = info['city_SE3_ego_lidar_t']
            for cam_type, cam_info in info['cam_infos'].items():
                if cam_info is None:
                    return None
                img_timestamp.append(cam_info['cam_timestamp_ns']/ 1e9)
                image_path = self.data_root / cam_info['fpath']
                image_paths.append(image_path)
                image_raw_paths.append(cam_info['fpath'])
                # obtain lidar to image transformation matrix
                city_SE3_ego_cam_t = cam_info['city_SE3_ego_cam_t']
                ego_SE3_cam = cam_info['ego_SE3_cam']
                ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t) #ego2glo_lidar -> glo2ego_cam
                cam_SE3_ego_cam_t = ego_SE3_cam.inverse().compose(ego_cam_t_SE3_ego_lidar_t) #ego -> cam
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = cam_SE3_ego_cam_t.rotation
                transform_matrix[:3, 3] = cam_SE3_ego_cam_t.translation

                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ transform_matrix)
                intrinsics.append(viewpad[:3, :3])
                extrinsics.append(transform_matrix)
                lidar2img_rts.append(lidar2img_rt)

            if self.with_seq_flag:
                is_last = index == len(self.flag)-1 or self.flag[index] != self.flag[index + 1]
                input_dict['is_last'] = is_last

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=intrinsics,
                    lidar2cam=extrinsics,
                ))
        if self.load_ann:
            annos = self.get_ann_info(index, input_dict)
            input_dict.update(annos)
            
        return input_dict

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            if "rot3d_range" in self.data_aug_conf: # Sparse4Dv3 / SparseDrive 3D aug
                rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
            elif "rot3d_lim" in self.data_aug_conf:  # BEVNeXT / BEVDet4D 3D aug
                rotate_bda = np.random.uniform(*self.data_aug_conf['rot3d_lim'])
                scale_bda = np.random.uniform(*self.data_aug_conf['scale3d_lim'])
                flip_dx = np.random.uniform() < self.data_aug_conf['flip3d_dx_ratio']
                flip_dy = np.random.uniform() < self.data_aug_conf['flip3d_dy_ratio']
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            if "rot3d_range" in self.data_aug_conf: # Sparse4Dv3 / SparseDrive 3D aug
                rotate_3d = 0
            elif "rot3d_lim" in self.data_aug_conf:  # BEVNeXT / BEVDet4D 3D aug
                rotate_bda = 0
                scale_bda = 1.0
                flip_dx = False
                flip_dy = False

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "original": self.data_aug_conf, # for TTA
        }
        if "rot3d_range" in self.data_aug_conf:  # Sparse4Dv3 / SparseDrive 3D aug
            aug_config.update({
                "rotate_3d": rotate_3d,
            })
        elif "rot3d_lim" in self.data_aug_conf:  # BEVNeXT / BEVDet4D 3D aug
            aug_config.update({
                "rotate_bda": rotate_bda,
                "scale_bda": scale_bda,
                "flip_dx": flip_dx,
                "flip_dy": flip_dy,
            })
        return aug_config

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix





