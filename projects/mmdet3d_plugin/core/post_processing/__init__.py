# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.post_processing import (merge_aug_bboxes, merge_aug_masks,
                                        merge_aug_proposals, merge_aug_scores,
                                        multiclass_nms)
from .merge_augs import merge_aug_bboxes_3d

__all__ = [
    'multiclass_nms', 'merge_aug_proposals', 'merge_aug_bboxes',
    'merge_aug_scores', 'merge_aug_masks', 'merge_aug_bboxes_3d',
]