# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES, ResNet


@BACKBONES.register_module()
class PVResNet(ResNet):
    """Map Encoder Module."""
    def __init__(self,
                 resnet_cfg,
                 ):
        super(PVResNet, self).__init__(**resnet_cfg)


    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        out = tuple(outs)

        return out


