# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from SparseDrive
# Copyright (c) 2024 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
import os
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

os.makedirs('data/kmeans_argoverse2', exist_ok=True)
os.makedirs('vis/kmeans_argoverse2', exist_ok=True)

K = 900
range_m = "50m"
range_type = "circle"

if range_m == "50m":
    perception_range = np.array([-55.0, -55.0, -5.0, 55.0, 55.0, 5.0], dtype=np.float32)  # 50m
    DIS_THRESH = 55
elif range_m == "150m":
    perception_range =  np.array([-152.4, -152.4, -5.0, 152.4, 152.4, 5.0], dtype=np.float32) # 150m
    DIS_THRESH = 152.4
else:
    raise NotImplementedError

fp = 'data/av2/av2_train_infos.pkl'
data = mmcv.load(fp)
#data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
data_infos = list(data["infos"])
center = []
for idx in tqdm(range(len(data_infos))):
    boxes = data_infos[idx]['gt3d_infos']['gt_boxes'][:,:3]
    if len(boxes) == 0:
        continue

    if range_type == "rect":
        #bev_range = point_cloud_range[[1, 0, 4, 3]]  # nuScenes coordinates (lidar is rotated 90 degree)
        bev_range = perception_range[[0, 1, 3, 4]]  # Argoverse2
        mask = (
                (boxes[:, 0] > bev_range[0])
                & (boxes[:, 1] > bev_range[1])
                & (boxes[:, 0] < bev_range[2])
                & (boxes[:, 1] < bev_range[3])
        )
        center.append(boxes[mask])
    elif range_type == "circle":
        distance = np.linalg.norm(boxes[:, :2], axis=1)
        center.append(boxes[distance < DIS_THRESH])
    else:
        raise NotImplementedError

center = np.concatenate(center, axis=0)

# Subsample for speed
max_points = 200_000
N = center.shape[0]
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
    center_sub = center[idx]
else:
    center_sub = center

print("start clustering, may take a few minutes.")
cluster = KMeans(n_clusters=K, verbose=1).fit(center).cluster_centers_
plt.scatter(cluster[:,0], cluster[:,1])
plt.savefig(f'vis/kmeans_argoverse2/det_anchor_{K}_{range_m}_{range_type}', bbox_inches='tight')
others = np.array([1,1,1,1,0,0,0,0])[np.newaxis].repeat(K, axis=0)
cluster = np.concatenate([cluster, others], axis=1)
np.save(f'data/kmeans_argoverse2/kmeans_det_{K}_{range_m}_{range_type}.npy', cluster)