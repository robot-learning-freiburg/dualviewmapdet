# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from typing import List
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import NECKS, build_neck, build_backbone


@NECKS.register_module()
class BEVFuserFPN(BaseModule):

    def __init__(self,
                 fuser,
                 pre_process,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 ):
        super(BEVFuserFPN, self).__init__()

        if fuser is not None:
            self.bev_map_fuser = build_neck(fuser)
        else:
            self.bev_map_fuser = None  # without fusion
        self.pre_process_net = build_backbone(pre_process)
        self.img_bev_encoder_backbone = \
            build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)


    def forward(self, bev_feat_0, bev_feat_1):
        if bev_feat_0 is not None and bev_feat_1 is not None:
            bev_feat = self.bev_map_fuser([bev_feat_0], [bev_feat_1])
        elif bev_feat_0 is None:
            assert self.bev_map_fuser is None
            assert bev_feat_1 is not None
            bev_feat = bev_feat_1
        elif bev_feat_1 is None:
            assert self.bev_map_fuser is None
            assert bev_feat_0 is not None
            bev_feat = bev_feat_0
        else:
            raise NotImplementedError

        if isinstance(bev_feat, List):
            assert len(bev_feat) == 1
            bev_feat = bev_feat[0]

        x = self.pre_process_net(bev_feat)[0]

        x = self.img_bev_encoder_backbone(x)
        bev_feat = self.img_bev_encoder_neck(x)  # (B, C, BEV_H, BEV_W)

        return bev_feat
