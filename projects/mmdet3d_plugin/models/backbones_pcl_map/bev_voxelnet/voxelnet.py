# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFusion
#   https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/models/fusion_models/bevfusion.py
# Copyright (c) 2023 MIT HAN Lab licensed under the Apache License 2.0,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
from typing import Dict

import torch
import torch.nn as nn
from mmdet.models import BACKBONES, build_backbone
from mmcv.cnn.utils import kaiming_init, xavier_init, constant_init
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32, auto_fp16

from projects.mmdet3d_plugin.ops import Voxelization, DynamicScatter



@BACKBONES.register_module()
class VoxelNet(BaseModule):
    def __init__(self,
                 voxelizer,
                 encoder,
                 bev_grid_size_empty,
                 voxelize_reduce=True,
                 ):
        super().__init__()
        if voxelizer.get("max_num_points", -1) > 0:
            self.voxelizer = Voxelization(**voxelizer)
        else:
            self.voxelizer = DynamicScatter(**voxelizer)
        self.encoder = build_backbone(encoder)
        self.voxelize_reduce = voxelize_reduce
        self.bev_grid_size_empty = bev_grid_size_empty

        self.fp16_enabled = False # Not sure if we should use this
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, data: Dict):
        x = data["points"]
        return self.extract_lidar_features(x)

    @auto_fp16(apply_to=("x",))
    def extract_lidar_features(self, x) -> torch.Tensor:
        batch_size = len(x)
        feats, coords, sizes = self.voxelize(x)

        # If *no voxels at all* -> return a valid empty BEV tensor
        # If too few voxels for BN (e.g., 1) -> return empty BEV tensor
        if coords.numel() == 0 or coords.shape[0] < 8:
            # pick a device/dtype to place the zeros on
            if isinstance(x, (list, tuple)) and len(x) > 0:
                ref = x[0]
                device = ref.device
                dtype = ref.dtype
            else:
                device = feats.device
                dtype = feats.dtype

            C, H, W = self.bev_grid_size_empty
            out = torch.zeros((batch_size, C, H, W), device=device, dtype=dtype)

            if self.training:
                out = out + self._ddp_touch().to(device=device, dtype=dtype)

            return out

        # normal path
        x = self.encoder(feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            res = res.contiguous()
            ret = self.voxelizer(res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                f, c = ret
                n = None

            # Skip empties (optional, but helps keep shapes sane)
            if c.numel() == 0:
                continue

            feats.append(f)

            # Hacky fix to fix wrong ordering from voxelization
            if c.shape[0] > 0:
                # Force consistency: If it looks like (z, y, x), flip it to (x, y, z)
                # so the rest of your pipeline stays predictable.
                mx = c.max(0).values
                if mx[0] < mx[2] and mx[0].max() < 64:
                    c = c[:, [2, 1, 0]]
            c = c.contiguous()

            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        if len(coords) == 0:
            # return consistent empty tensors on the right device
            # (feats dtype is float, coords int)
            # If points list can be empty, guard that too:
            device = points[0].device if len(points) > 0 else "cpu"
            feats = torch.empty((0, 0), device=device, dtype=torch.float32)
            coords = torch.empty((0, 4), device=device, dtype=torch.int32)
            sizes = torch.empty((0,), device=device, dtype=torch.int32)
            return feats, coords, sizes

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        else:
            sizes = torch.empty((0,), device=coords.device, dtype=torch.int32)

        return feats, coords, sizes

    # Fix ddp when point clouds might be empty
    def _ddp_touch(self):
        # create a scalar that depends on ALL trainable encoder params but evaluates to 0
        s = None
        for p in self.encoder.parameters():
            if p.requires_grad:
                v = p.view(-1)[0]  # cheap: one element
                s = v if s is None else (s + v)
        if s is None:
            return 0.0
        return s * 0.0

