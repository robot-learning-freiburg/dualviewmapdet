# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import copy
from collections import OrderedDict
from inspect import signature
import logging

import torch
import torch.distributed as dist

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from mmdet.utils import get_root_logger

from .grid_mask import GridMask
from .grid_mask_pcl import GridMaskPcl
from ..utils.count_params import log_params_per_child
from ..utils.nan_error_mode import NaNErrorMode

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["DualViewMapDet"]


@DETECTORS.register_module()
class DualViewMapDet(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_grid_mask_pv_pcl_map=False,
        use_grid_mask_bev_pcl_map=False,
        point_cloud_range=None,
        use_deformable_func=False,
        depth_branch=None,
        pv_pcl_map_encoder=None,
        bev_pcl_map_encoder=None,
        img_bev_encoder=None,
        img_map_pv_fuser=None,
        img_map_bev_fuser=None,
        bev_head=None,
        detect_nan_values=False,
    ):
        super(DualViewMapDet, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        self.use_grid_mask_pv_pcl_map = use_grid_mask_pv_pcl_map
        self.use_grid_mask_bev_pcl_map = use_grid_mask_bev_pcl_map
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        if use_grid_mask_pv_pcl_map:
            self.grid_mask_pv_pcl_map = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7, mask_channel_index=-1
            )
        if use_grid_mask_bev_pcl_map:
            self.grid_mask_bev_pcl_map = GridMaskPcl(
                point_cloud_range, True, True, ratio=0.5, mode=1, prob=0.7, d_max=20.0,
            )

        # Point cloud map encoders
        if pv_pcl_map_encoder is not None:
            self.pv_pcl_map_encoder = build_backbone(pv_pcl_map_encoder)
            assert img_map_pv_fuser is not None, "img_map_pv_fuser must be specified when using pv_pcl_map_encoder"
        else:
            self.pv_pcl_map_encoder = None
        if bev_pcl_map_encoder is not None:
            self.bev_pcl_map_encoder = build_backbone(bev_pcl_map_encoder)
            assert not (img_map_bev_fuser is not None and img_bev_encoder is None)
        else:
            self.bev_pcl_map_encoder = None

        # Image to BEV lifting network
        if img_bev_encoder is not None:
            self.img_bev_encoder = build_neck(img_bev_encoder)
            assert not (img_map_bev_fuser is not None and bev_pcl_map_encoder is None)
        else:
            self.img_bev_encoder = None

        # Image - map feature fusion layers
        if img_map_pv_fuser is not None:
            self.img_map_pv_fuser = build_neck(img_map_pv_fuser)
        else:
            self.img_map_pv_fuser = None
        if img_map_bev_fuser:
            self.img_map_bev_fuser = build_neck(img_map_bev_fuser)
            assert img_bev_encoder is not None and bev_pcl_map_encoder is not None
        else:
            self.img_map_bev_fuser = None

        if bev_head is not None:
            self.bev_head = build_head(bev_head)
        else:
            self.bev_head = None

        self.detect_nan_values = detect_nan_values

        self.test_cfg = test_cfg

        logger = get_root_logger()
        log_params_per_child(self, logger=logger, trainable_only=False, sort="desc")
        log_params_per_child(self, logger=logger, trainable_only=True,  sort="desc", title="Trainable params per child")

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depths_pv=False, data=None):
        bs = img.shape[0]

        pcl_map_depth = data.get("pcl_map_depth", None)
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
            if pcl_map_depth is not None:
                pcl_map_depth = pcl_map_depth.flatten(end_dim=1)
        else:
            num_cams = 1

        # Image encoding
        if self.use_grid_mask:
            img = self.grid_mask(img)
        #self.visualize_imgs_depth(img, None)
        if "metas" in signature(self.img_backbone.forward).parameters:
            img_feature_maps = self.img_backbone(img, num_cams, metas=data)
        else:
            img_feature_maps = self.img_backbone(img)
        if isinstance(img_feature_maps, dict):
            img_feature_maps = list(img_feature_maps.values())

        # Perspective view and BEV point cloud map encoding
        if self.pv_pcl_map_encoder is not None:
            if self.use_grid_mask_pv_pcl_map:
                pcl_map_depth = self.grid_mask_pv_pcl_map(pcl_map_depth)
            pv_pcl_map_features = self.pv_pcl_map_encoder(pcl_map_depth)
            if isinstance(pv_pcl_map_features, dict):
                pv_pcl_map_features = list(pv_pcl_map_features.values())
        else:
            pv_pcl_map_features = None

        if self.bev_pcl_map_encoder is not None:
            if self.use_grid_mask_bev_pcl_map:
                data["points"] = self.grid_mask_bev_pcl_map(data["points"])
            bev_pcl_map_features = self.bev_pcl_map_encoder(data)
        else:
            bev_pcl_map_features = None

        # Image - map feature fusion in PV
        if self.img_map_pv_fuser is not None:
            pv_feature_maps = self.img_map_pv_fuser(img_feature_maps, pv_pcl_map_features)
        else:
            pv_feature_maps = img_feature_maps

        # PV Neck (FPN)
        if self.img_neck is not None:
            pv_feature_maps = list(self.img_neck(pv_feature_maps))
        for i, feat in enumerate(pv_feature_maps):
            pv_feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        # Auxiliary depth head
        if return_depths_pv and self.depth_branch is not None:
            depths_pv = self.depth_branch(pv_feature_maps, data.get("focal"))
        else:
            depths_pv = None

        # Image features to BEV features lifting
        if self.img_bev_encoder is not None:
            depth_lss, bev_img_features = self.img_bev_encoder(img, pv_feature_maps, data)
        else:
            depth_lss = None
            bev_img_features = None

        # Image - map feature fusion in BEV
        if self.img_map_bev_fuser is not None:
            bev_feature_maps = self.img_map_bev_fuser(bev_img_features, bev_pcl_map_features)
        elif self.bev_pcl_map_encoder is not None:
            bev_feature_maps = bev_pcl_map_features
        elif self.img_bev_encoder is not None:
            bev_feature_maps = bev_img_features
        else:
            bev_feature_maps = None

        if self.bev_head is not None:
            bev_pred = self.bev_head(bev_feature_maps)
        else:
            bev_pred = None

        # Convert features to be used for deformable aggregation
        if bev_feature_maps is not None and self.use_deformable_func:
            bev_feature_maps = [bev_feature_maps.unsqueeze(1)]  # [(B, 1(CAMS), C, H, W)]
            bev_feature_maps = feature_maps_format(bev_feature_maps)

        if self.use_deformable_func:
            pv_feature_maps = feature_maps_format(pv_feature_maps)

        return pv_feature_maps, bev_feature_maps, depths_pv, depth_lss, bev_pred

    @force_fp32(apply_to=("img",))
    def forward(self, img, return_loss=True, return_feats=False, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            if self.detect_nan_values:
                with NaNErrorMode(
                        enabled=True, raise_error=True, print_stats=True, print_nan_index=True
                ):
                    return self.forward_train(img, return_feats=return_feats, **kwargs)
            else:
                return self.forward_train(img, return_feats=return_feats, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def forward_train(self, img, return_feats=False, **data):
        ## Visualize imgs and depth map
        vis_imgs_depth = False
        if vis_imgs_depth:
            self.visualize_imgs_depth(img, data)

        pv_feature_maps, bev_feature_maps, depths_pv, depth_lss, bev_pred = self.extract_feat(img, True, data)
        model_outs = self.head(pv_feature_maps, data, bev_feature_maps)

        output = self.head.loss(model_outs, data)
        if "gt_depth" in data:
            if depths_pv is not None:
                output["loss_dense_depth"] = self.depth_branch.loss(
                    depths_pv, data["gt_depth"]
                )
            if depth_lss is not None:
                output["loss_dense_depth_lss"] = self.img_bev_encoder.get_depth_loss(depth_lss, data["gt_depth"], img)

        if bev_pred is not None:
            output.update(self.bev_head.loss(bev_pred, data))

        if return_feats:
            return pv_feature_maps, bev_feature_maps, output
        else:
            return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        ## Visualize imgs and depth map
        vis_imgs_depth = False
        if vis_imgs_depth:
            self.visualize_imgs_depth(img, data)

        pv_feature_maps, bev_feature_maps, depths_pv, depth_lss, bev_pred \
            = self.extract_feat(img, False, data)

        model_outs = self.head(pv_feature_maps, data, bev_feature_maps)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        from ..core.bbox.converter import nuscenes_to_mmdet3d_boxes
        from ..core.bbox.converter import mmdet3d_to_nuscenes_boxes
        from projects.mmdet3d_plugin.datasets.pipelines.augment import revert_bev_transform
        from projects.mmdet3d_plugin.core.post_processing.merge_augs import merge_aug_bboxes_3d

        batch_size = img[0].shape[0]
        aug_bboxes = [[] for _ in range(batch_size)]
        device = img[0].device

        # Use one instance bank for each data aug to keep the streaming manner during testing
        if not hasattr(self, 'instance_banks'):
            num_augs = len(img)
            self.instance_banks = [copy.deepcopy(self.head.det_head.instance_bank) for _ in range(num_augs)]

        for i in range(len(img)): # iterate over all augmentations
            self.head.det_head.instance_bank = self.instance_banks[i]

            aug_data = dict()
            for key, val in data.items():
                if key == 'rescale':
                    aug_data[key] = val
                else:
                    aug_data[key] = val[i]

            output = self.simple_test(img[i], **aug_data)
            for j, output_j in enumerate(output): # over batch
                aug_box = output_j['img_bbox']
                aug_box['boxes_3d'] = revert_bev_transform(aug_box['boxes_3d'], aug_data['bda'][j])
                aug_box['boxes_3d'] = nuscenes_to_mmdet3d_boxes(aug_box['boxes_3d'])
                aug_box = {k:v.to(device) for (k,v) in aug_box.items()}
                aug_bboxes[j].append(aug_box)

        output_merged_boxes = []
        for aug_bboxes_j in aug_bboxes: # over batch
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes_j, self.test_cfg, img_metas=None)
            merged_bboxes['boxes_3d'] = mmdet3d_to_nuscenes_boxes(merged_bboxes['boxes_3d'])
            output_merged_boxes.append({"img_bbox": merged_bboxes})
        return output_merged_boxes

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        # log_vars_ = dict()
        # for loss_name, loss_value in log_vars.items():
        #     k = loss_name + '_val'
        #     log_vars_[k] = loss_value

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[f'_losses_{loss_name}'] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[f'_losses_{loss_name}'] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['/loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def visualize_imgs_depth(self, img, data):
        vis_show_img = True
        vis_show_depth = True

        import numpy as np
        from matplotlib import pyplot as plt
        with torch.no_grad():
            img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                to_rgb=True)

            # Precompute mean/std tensors
            img_mean = torch.as_tensor(img_norm_cfg["mean"], device=img.device).view(3, 1, 1) / 255.0
            img_std = torch.as_tensor(img_norm_cfg["std"], device=img.device).view(3, 1, 1) / 255.0

            if img.dim() == 5:
                B, Cams = img.shape[:2]
                per_cam = True
            else:
                B, Cams = img.shape[0], 1
                per_cam = False
            for batch_idx in range(B):
                for cam_idx in range(Cams): # av2: 3
                    # De-normalize and clamp to valid image range
                    if per_cam:
                        img_unnorm = (img[batch_idx, cam_idx] * img_std + img_mean).clamp(0.0, 1.0)
                    else:
                        img_unnorm = (img[batch_idx] * img_std + img_mean).clamp(0.0, 1.0)
                    im_np = img_unnorm[[0, 1, 2], ...].permute(1, 2, 0).detach().cpu().numpy()

                    # Depth + mask invalid (<= 0) and non-finite values
                    if vis_show_depth:
                        if per_cam:
                            depth = data["pcl_map_depth"][batch_idx, cam_idx, 0].detach().cpu().numpy()  # or gt_depth
                        else:
                            depth = data["pcl_map_depth"][batch_idx, cam_idx, 0].detach().cpu().numpy()  # or gt_depth
                        invalid = (depth <= 0) | ~np.isfinite(depth)
                        depth_masked = np.ma.masked_where(invalid, depth)

                    # High-resolution figure
                    fig, ax = plt.subplots(figsize=(20, 7.27272728), dpi=200, frameon=False) # av2: 12, 7
                    ax = fig.add_axes([0, 0, 1, 1])  # fill the whole canvas (no margins)
                    if vis_show_img:
                        ax.imshow(im_np, interpolation="nearest")
                    if vis_show_depth:
                        ax.imshow(depth_masked, alpha=0.8, cmap="magma", vmin=0.0, vmax=1.0, # av2:/3,
                                  interpolation="nearest")
                    ax.set_axis_off()
                    plt.show()
