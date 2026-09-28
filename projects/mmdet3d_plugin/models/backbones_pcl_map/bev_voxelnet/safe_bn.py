# projects/mmdet3d_plugin/models/utils/safe_bn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import NORM_LAYERS


@NORM_LAYERS.register_module(name="SafeBN1d")
class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BN1d that won't crash when N==1 in training.
    For N==1: use running stats (training=False) and do NOT update them.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input is (N, C) in spconv sparse features
        if input.dim() == 2 and self.training and input.size(0) == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,   # use running stats
                momentum=0.0,     # do not update running stats
                eps=self.eps,
            )
        return super().forward(input)
