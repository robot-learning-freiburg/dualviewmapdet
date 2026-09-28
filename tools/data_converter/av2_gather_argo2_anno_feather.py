# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from Far3D
#   https://github.com/megvii-research/Far3D/blob/main/tools/create_infos_av2/gather_argo2_anno_feather.py
# Copyright (c) 2024 MEGVII, licensed under the  Apache License 2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# "create complete infos of gts for evaluation"
from av2.utils.io import read_feather
import pandas as pd   
from pathlib import Path   
import os  

if __name__ == '__main__':
    save_path = "./data/av2/val_anno.feather" # replace with absolute path
    split_dir = Path("./data/av2/val") # replace with absolute path
    annotations_path_list = split_dir.glob("*/annotations.feather")

    seg_anno_list = []
    for annotations_path in annotations_path_list:
        
        seg_anno = read_feather(Path(annotations_path))
        log_dir = os.path.dirname(annotations_path)
        log_id = log_dir.split('/')[-1]
        print(log_id)
        seg_anno["log_id"] = log_id
        seg_anno_list.append(seg_anno)
    
    gts = pd.concat(seg_anno_list).reset_index()
    gts.to_feather(save_path)