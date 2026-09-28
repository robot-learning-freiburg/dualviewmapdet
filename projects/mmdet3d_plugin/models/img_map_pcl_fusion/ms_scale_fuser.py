# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from typing import List, Sequence

import torch
from mmcv.runner import BaseModule
from torch import nn
from mmdet.models import NECKS


@NECKS.register_module()
class MultiScaleConvFuser(BaseModule):
    """
    Multi-scale concat + conv fusion.

    For each scale i:
        fused_i = Conv3x3( concat(img_i, pv_pcl_map_i) ) -> BN -> ReLU

    Args:
        img_in_channels (Sequence[int]): channels per scale for image branch.
        pv_pcl_map_in_channels (Sequence[int]): channels per scale for pv_pcl_map branch.
        out_channels (Sequence[int]): output channels per fused scale (same length as inputs).
        use_bn (bool): whether to use BatchNorm2d.
        act_cfg (dict): activation config. Supported: {'type': 'ReLU', 'inplace': True}

    Forward:
        img_feature_maps (List[Tensor]): list of (B, C_img_i, H_i, W_i)
        pv_pcl_map_features (List[Tensor]): list of (B, C_pv_pcl_map_i, H_i, W_i)

    Returns:
        List[Tensor]: fused feature maps, each (B, out_channels[i], H_i, W_i)
    """

    def __init__(
        self,
        img_in_channels: Sequence[int],
        pv_pcl_map_in_channels: Sequence[int],
        out_channels: Sequence[int],
        use_bn: bool = True,
        act_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if len(img_in_channels) != len(pv_pcl_map_in_channels):
            raise ValueError(
                f"img_in_channels and pv_pcl_map_in_channels must have same length, got "
                f"{len(img_in_channels)} and {len(pv_pcl_map_in_channels)}."
            )

        self.img_in_channels = list(img_in_channels)
        self.pv_pcl_map_in_channels = list(pv_pcl_map_in_channels)
        self.num_scales = len(self.img_in_channels)

        if len(out_channels) != self.num_scales:
            raise ValueError(
                f"out_channels must have length {self.num_scales}, got {len(out_channels)}."
            )
        self.out_channels = list(out_channels)

        if act_cfg is None:
            act_cfg = {"type": "ReLU", "inplace": True}
        if act_cfg.get("type", "ReLU") != "ReLU":
            raise ValueError(f"Only ReLU is supported in this minimal impl, got act_cfg={act_cfg}")
        inplace = bool(act_cfg.get("inplace", True))

        self.fuse_blocks = nn.ModuleList()
        for c_img, c_pv_pcl_map, c_out in zip(
            self.img_in_channels, self.pv_pcl_map_in_channels, self.out_channels
        ):
            layers: List[nn.Module] = [
                nn.Conv2d(c_img + c_pv_pcl_map, c_out, kernel_size=3, padding=1, bias=not use_bn)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            # different activation module per scale
            layers.append(nn.ReLU(inplace=inplace))

            self.fuse_blocks.append(nn.Sequential(*layers))

        self.fp16_enabled = True

    def forward(
        self,
        img_feature_maps: List[torch.Tensor],
        pv_pcl_map_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(img_feature_maps) != self.num_scales or len(pv_pcl_map_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scales, got "
                f"{len(img_feature_maps)} (img) and {len(pv_pcl_map_features)} (pv_pcl_map)."
            )

        fused: List[torch.Tensor] = []
        for i, (img_f, pv_pcl_map_f, block) in enumerate(
            zip(img_feature_maps, pv_pcl_map_features, self.fuse_blocks)
        ):
            if img_f.shape[0] != pv_pcl_map_f.shape[0] or img_f.shape[-2:] != pv_pcl_map_f.shape[-2:]:
                raise ValueError(
                    f"Scale {i}: batch/spatial mismatch: img {tuple(img_f.shape)}, "
                    f"pv_pcl_map {tuple(pv_pcl_map_f.shape)}"
                )
            if img_f.shape[1] != self.img_in_channels[i] or pv_pcl_map_f.shape[1] != self.pv_pcl_map_in_channels[i]:
                raise ValueError(
                    f"Scale {i}: channel mismatch: expected img {self.img_in_channels[i]}, "
                    f"pv_pcl_map {self.pv_pcl_map_in_channels[i]}, got img {img_f.shape[1]}, "
                    f"pv_pcl_map {pv_pcl_map_f.shape[1]}"
                )

            x = torch.cat([img_f, pv_pcl_map_f], dim=1)
            fused.append(block(x))

        return fused
