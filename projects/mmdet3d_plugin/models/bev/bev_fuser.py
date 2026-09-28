# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFusion
#   https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/models/fusers/conv.py
# Copyright (c) 2023 MIT HAN Lab licensed under the Apache License 2.0,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
from typing import List

import torch
from mmdet.models import NECKS
from torch import nn


@NECKS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input_0: List[torch.Tensor], input_1: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_0) == len(input_1) == 1
        inputs = input_0 + input_1
        return super().forward(torch.cat(inputs, dim=1))
