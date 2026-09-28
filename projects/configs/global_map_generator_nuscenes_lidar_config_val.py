#version = 'mini'  # debug
version = 'trainval'

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None

total_batch_size = 8
num_gpus = 8
batch_size = total_batch_size // num_gpus

checkpoint_config = None

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

load_from = None
resume_from = None
workflow = [('train', 1)]
input_shape = (1600, 900)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

model = dict(
    type="GlobalMapGenerator",
    img_norm_cfg=img_norm_cfg,
    voxel_pooling=False,
    point_format="xyzirgbc",
    tile_size=50.0,
    save_dir="./data/nuscenes_gmaps/lidar_map",
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"
file_client_args = dict(backend="disk")

pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # Commented out to not filter ignore classes out
    #dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
            'points',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "is_last", "map_location"],
    ),
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

data_aug_conf = {
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,  # debug 0
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=pipeline,
        test_mode=True,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=1,  # changed from 2 to 1 to not split sequences
        keep_consistent_seq_aug=True,
        shuffle=False,
        drop_last=False,
    ),
    val=dict(),
    test=dict(),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=4e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(),
)
optimizer_config = dict()
lr_config = None
runner = dict(type='EpochBasedRunner', max_epochs=1)
