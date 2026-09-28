# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import torch
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import NECKS, build_neck


@NECKS.register_module()
class BEVEncoder(BaseModule):

    def __init__(
            self,
            img_view_transformer,
            bev_trans_feature_scale,
            grid_config,
            img_norm_cfg,
    ):
        super(BEVEncoder, self).__init__()
        self.img_view_transformer = build_neck(img_view_transformer)
        self.bev_trans_feature_scale = bev_trans_feature_scale
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.fp16_enabled = True

    def forward(self, img, img_feature_maps, data):
        device = img.device
        batch_size = data['cam2lidar'].shape[0]
        num_cams = data['cam2lidar'].shape[1]

        # Use lidar as "ego"
        cam2lidar = data['cam2lidar']  # (B, CAMS, 4, 4)
        rot = cam2lidar[:, :, :3, :3]
        tran = cam2lidar[:, :, :3, 3]

        intrin = data['cam_intrinsic']  # (B, CAMS, 3, 3)

        # Image data aug
        # we do not need post_tran and post_rot since it there are already within cam_intrinsic, see ResizeCropFlipImage
        # Thus we, set it to tans to zero and rot to identity
        post_tran = torch.zeros(batch_size, num_cams, 3).to(device)  # (B, CAMS, 3)
        post_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_cams, 1, 1).to(device)  # (B, CAMS, 3, 3)

        # BEV data aug (lidar2bev_aug), only use bda for the MLP and not for the lifting
        # since bda is already within cam2lidar, see BEVDataAug
        if 'bda' in data:
            bda = data['bda']  # (B, 4, 4)
        else:  # if not used
            bda = torch.eye(4).repeat(batch_size, 1, 1).to(device)
        bda_identity = torch.eye(4).repeat(batch_size, 1, 1).to(device)
        bda_identity = bda_identity[:, :3, :3]
        bda = bda[:, :3, :3]  # (B, 3, 3)

        mlp_input = self.img_view_transformer.get_mlp_input(
            rot, tran, intrin, post_rot, post_tran, bda)

        prev_info = None  # Not used in view transformer

        img = img.view(batch_size, num_cams, img.shape[1], img.shape[2], img.shape[3])  # (B, CAMS, 3, H, W)
        canvas = self.get_canvas(img)

        x = [img_feature_maps[self.bev_trans_feature_scale], None]  # (B, N, C, H, W)

        # Use bda_identity since bda is already in cam2lidar tranformation
        bev_feat, depth, others, tran_feat = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda_identity, mlp_input, prev_info, canvas])

        return depth, bev_feat

    def get_canvas(self, img):
        device = img.device
        img_std = torch.tensor(self.img_norm_cfg["std"]).to(device) / 255.0
        img_mean = torch.tensor(self.img_norm_cfg["mean"]).to(device) / 255.0
        img_unnorm = img * img_std[:, None, None] + img_mean[:, None, None]  # (B, CAMS, 3, H, W)
        canvas = torch.permute(img_unnorm, (0, 1, 3, 4, 2))  # (B, N, H, W, C)
        return canvas

    def get_depth_loss(self, depth, gt_depth, img):
        canvas = None  # self.get_canvas(img)
        gt_depth = gt_depth[self.bev_trans_feature_scale]  #  (B, CAMS, H, W), scale 1/8
        gt_depth[gt_depth == -1] = 0.0  # change background value, see get_downsampled_gt_depth
        gt_depth[gt_depth < self.grid_config['depth'][0]] = 0.0
        gt_depth[gt_depth >= self.grid_config['depth'][1]] = 0.0
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth,
                                                              depth,
                                                              canvas)
        return loss_depth["loss_depth"]
