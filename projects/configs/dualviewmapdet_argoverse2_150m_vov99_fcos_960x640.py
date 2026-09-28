# ================ base config ===================

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
ssd = ""

total_batch_size = 24
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_epochs = 80
checkpoint_epoch_interval = 1
eval_epoch_interval = 20

checkpoint_config = dict(interval=checkpoint_epoch_interval)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook') # alternative to wandb
        dict(type='CustomWandbLoggerHook',
             init_kwargs=dict(project='gmp', mode="online")),  # disabled online
    ])

load_from = None
resume_from = None
workflow = [('train', eval_epoch_interval), ('val', 1)]
fp16 = dict(loss_scale='dynamic')
input_shape = (960, 640)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# ================== model ========================
class_names = [
    'ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
    'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
    'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
    'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
    'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
    'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER'
]
perception_range = [-152.4, -152.4, -5.0, 152.4, 152.4, 5.0]
max_depth = 150.0
num_classes = len(class_names)

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
with_quality_estimation = True


grid_config_pcl_map = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5.0, 3.0, 1.6],
    'depth': [1.0, 60.0, 0.5],
}
point_cloud_range = [grid_config_pcl_map['x'][0], grid_config_pcl_map['y'][0], grid_config_pcl_map['z'][0],
                     grid_config_pcl_map['x'][1], grid_config_pcl_map['y'][1], grid_config_pcl_map['z'][1]]
voxel_size= [0.2, 0.2, 0.4]

# Cam BEV net
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
numC_Trans = 256
multi_adj_frame_id_cfg = (1, 1 + 0, 1)

task_config = dict(
    with_det=True,
    with_map=False,
    with_motion_plan=False,
)

model = dict(
    type="DualViewMapDet",
    use_grid_mask=True,
    use_grid_mask_pv_pcl_map=True,
    use_grid_mask_bev_pcl_map=True,
    point_cloud_range=point_cloud_range,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        out_features=['stage2', 'stage3', 'stage4', 'stage5'],
        norm_eval=True,
        frozen_stages=1,
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpt/fcos3d_vovnet_imgbackbone-remapped.pth',
            prefix='img_backbone.'),
    ),
    pv_pcl_map_encoder=dict(
        type="PVResNet",
        resnet_cfg=dict(
            in_channels=31,
            depth=18,
            num_stages=4,
            frozen_stages=-1,
            norm_eval=False,
            style="pytorch",
            with_cp=False,
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type="BN", requires_grad=True),
            pretrained="ckpt/resnet18-5c106cde.pth",
        ),
    ),
    img_map_pv_fuser=dict(
        type='MultiScaleConvFuser',
        img_in_channels=[256, 512, 768, 1024],
        pv_pcl_map_in_channels=[64, 128, 256, 512],
        out_channels=[256, 512, 768, 1024],
    ),
    img_bev_encoder=dict(
        type="BEVEncoder",
        img_view_transformer=dict(
            type='LSSViewTransformerBEVNeXt',
            grid_config=grid_config,
            input_size=input_shape[::-1],
            in_channels=256,
            out_channels=numC_Trans,
            depthnet_cfg=dict(
                depth_in_channels=256,
                context_in_channels=256,
                mid_channels=256,
                use_dcn=True
            ),
            downsample=16,
            patch_size=16,
            loss_depth_weight=0.15,
            crf_config=dict(
                crf_num_iter=0,
                grad=False,
                crf_zeta=0.001
            ),
            accelerate=False,
        ),
        bev_trans_feature_scale=2, # 1
        grid_config=grid_config,
        img_norm_cfg=img_norm_cfg,
    ),
    bev_pcl_map_encoder=dict(
        type="VoxelNet",
        voxelizer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000],
        ),
        encoder=dict(
            type="SparseEncoder",
            in_channels=3,
            sparse_shape=[512, 512, 21],
            bev_grid_size_empty=[256, 128, 128],
            base_channels=16,
            output_channels=128,
            order=["conv", "norm", "act"],
            encoder_channels=[
                [16, 16, 32],
                [32, 32, 64],
                [64, 64],
            ],
            encoder_paddings=[
                [0, 0, 1],
                [0, 0, 1],
                [0, [1, 1, 0]],
            ],
            block_type="basicblock",
        ),
        bev_grid_size_empty=[256, 128, 128],
    ),
    img_map_bev_fuser=dict(
        type="BEVFuserFPN",
        fuser=dict(
            type='ConvFuser',
            in_channels=[embed_dims, embed_dims],
            out_channels=embed_dims,
        ),
        pre_process=dict(
            type='CustomResNet',
            numC_input=numC_Trans,
            num_layer=[2, ],
            num_channels=[numC_Trans, ],
            stride=[1, ],
            backbone_output_ids=[0, ]),
        img_bev_encoder_backbone=dict(
            type='CustomResNet',
            numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
            num_channels=[numC_Trans, numC_Trans, numC_Trans],
            stride=[1, 2, 2]),
        img_bev_encoder_neck=dict(
            type='FPN_LSS',
            in_channels=numC_Trans + numC_Trans,
            out_channels=256,
            extra_upsample=None,
        ),
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 768, 1024],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.1,
    ),
    head=dict(
        type="DualViewMapDetHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=1800,
                embed_dims=embed_dims,
                anchor="data/kmeans_argoverse2/kmeans_det_1800_150m_rect.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=1200 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "gnn",
                    "norm",
                    "deformable_bev",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable_bev",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder)
            )[2:],
            temp_graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=7,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            deformable_model_bev=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=1,
                num_cams=1,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=False,
                residual_mode="add",
                proj_mode="bev",
                grid_config=grid_config,
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=5,
                num_temp_dn_groups=3,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={},
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("BOLLARD"),
                                   class_names.index("CONSTRUCTION_CONE"),
                                   class_names.index("CONSTRUCTION_BARREL"),
                                   class_names.index("BOLLARD"),],
            ),
            decoder=dict(type="SparseBox3DDecoder"),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
        map_head=None,
        motion_plan_head=None,
    ),
)

# ================== data ========================
dataset_type = "Argoverse2DatasetT"
data_root = f"data/av2{ssd}/"
anno_root = f"data/av2{ssd}/"
file_client_args = dict(backend="disk")
if ssd == "_ssd":
    gmap_root = "./data_ssd"
else:
    gmap_root = "./data"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
        max_depth=max_depth,
    ),
    dict(type="BEVDataAug"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="LoadGlobalMap",
        save_dir=f"{gmap_root}/av2_gmaps/lidar_map",
        lidar_pose=f"lidarpose/av2_pose_graph",
        map_range=50.0,
        tile_size=50.0,
        point_format_input="xyzirgbc",
        point_format_output="xyz",
        with_dynamic_objects=False,
        static_objects_class_ids=[class_names.index("BOLLARD"),
                                  class_names.index("SIGN"),
                                  class_names.index("STOP_SIGN")],
        voxel_size=0.4,
        sor_k=20,
        sor_std=0.8,
        min_timediff=120.0,
        only_from_split=True,
        cache_size_tiles=20,
    ),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=[1],
        pcl_key="pcl_map",
        out_key="pcl_map_depth",
        add_xyz=True,
    ),
    dict(
        type="NormalizeMultiScaleDepthMap",
        depth_key="pcl_map_depth",
        max_depth=max_depth,
        depth_pos_embed_num_freq=12,
    ),
    dict(
        type="AddNearestDepthSpreadAndDistToValid",
        depth_key="pcl_map_depth",
        out_key="pcl_map_depth",
        max_dist=20.0, # pixels (radius for spreading + clip for dist)
    ),
    dict(
        type="PclMapToLiDARPoints",
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="PointShuffle",
    ),
    dict(
        type="ObjectRangeFilter",
        point_cloud_range=perception_range,
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "lidar2cam",
            "cam2lidar",
            "cam_intrinsic",
            #"bda",
            "gt_depth",
            "pcl_map_depth",
            "points",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
        max_depth=max_depth,
    ),
    dict(type="BEVDataAug"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="LoadGlobalMap",
        save_dir=f"{gmap_root}/av2_gmaps/lidar_map",
        lidar_pose=f"lidarpose/av2_pose_graph",
        map_range=50.0,
        tile_size=50.0,
        point_format_input="xyzirgbc",
        point_format_output="xyz",
        with_dynamic_objects=False,
        static_objects_class_ids=[class_names.index("BOLLARD"),
                                  class_names.index("SIGN"),
                                  class_names.index("STOP_SIGN")],
        voxel_size=0.4,
        sor_k=20,
        sor_std=0.8,
        min_timediff=120.0,
        only_from_split=False,  # use train as well
        cache_size_tiles=20,
    ),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=[1],
        pcl_key="pcl_map",
        out_key="pcl_map_depth",
        add_xyz=True,
    ),
    dict(
        type="NormalizeMultiScaleDepthMap",
        depth_key="pcl_map_depth",
        max_depth=max_depth,
        depth_pos_embed_num_freq=12,
    ),
    dict(
        type="AddNearestDepthSpreadAndDistToValid",
        depth_key="pcl_map_depth",
        out_key="pcl_map_depth",
        max_dist=20.0, # pixels (radius for spreading + clip for dist)
    ),
    dict(
        type="PclMapToLiDARPoints",
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="PointShuffle",
    ),
    dict(
        type="ObjectRangeFilter",
        point_cloud_range=perception_range,
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "lidar2cam",
            "cam2lidar",
            "cam_intrinsic",
            "gt_depth",
            "pcl_map_depth",
            "points",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="LoadGlobalMap",
        save_dir=f"{gmap_root}/av2_gmaps/lidar_map",
        lidar_pose=f"lidarpose/av2_pose_graph",
        map_range=50.0,
        tile_size=50.0,
        point_format_input="xyzirgbc",
        point_format_output="xyz",
        with_dynamic_objects=False,
        static_objects_class_ids=[class_names.index("BOLLARD"),
                                  class_names.index("SIGN"),
                                  class_names.index("STOP_SIGN")],
        voxel_size=0.4,
        sor_k=20,
        sor_std=0.8,
        min_timediff=120.0,
        only_from_split=False,  # use train as well
        cache_size_tiles=20,
    ),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=[1],
        pcl_key="pcl_map",
        out_key="pcl_map_depth",
        add_xyz=True,
    ),
    dict(
        type="NormalizeMultiScaleDepthMap",
        depth_key="pcl_map_depth",
        max_depth=max_depth,
        depth_pos_embed_num_freq=12,
    ),
    dict(
        type="AddNearestDepthSpreadAndDistToValid",
        depth_key="pcl_map_depth",
        out_key="pcl_map_depth",
        max_dist=20.0, # pixels (radius for spreading + clip for dist)
    ),
    dict(
        type="PclMapToLiDARPoints",
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="PointShuffle",
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "lidar2cam",
            "cam2lidar",
            "cam_intrinsic",
            "pcl_map_depth",
            "points",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
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
    eval_range_m=150.0,
)

data_aug_conf = {
    "resize_lim": (0.47, 0.55), # same as Far3D; similar to Sparse4Dv3-ResNet50 resize: (0.40, 0.47) * (960/704) = (0.545. 0.641);  (0.545. 0.641) * (1600/2048) = (0.426, 0.501); Note: with lower than 1.0 and full image size, it will crop parts at the bottom away
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 1550,
    "W": 2048,
    "rand_flip": True,
    #"rot3d_range": [0, 0],
    "rot3d_lim": (-22.5, 22.5),
    "scale3d_lim": (0.95, 1.05),
    "flip3d_dx_ratio": 0.5,
    "flip3d_dy_ratio": 0.5,
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,  # debug 0
    train=dict(
        **data_basic_config,
        ann_file=anno_root + f"av2_train_infos.pkl",
        split='train',
        load_interval=5,
        num_frame_losses=1,
        data_aug_conf=data_aug_conf,
        sequences_split_num=2,
        with_seq_flag=True,
        pipeline=train_pipeline,
        queue_length=1,
        test_mode=False,
        use_valid_flag=False,
        interval_test=False,
        keep_consistent_seq_aug=True,
        shuffle=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + f"av2_val_infos.pkl",
        pipeline=test_pipeline,
        queue_length=1,
        split='val',
        load_interval=5,
        interval_test=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        test_mode=True,
        shuffle=False,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + f"av2_val_infos.pkl",
        pipeline=test_pipeline,
        queue_length=1,
        split='val',
        load_interval=5,
        interval_test=False,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "img_bev_encoder": dict(lr_mult=0.4),
            "img_map_bev_fuser": dict(lr_mult=0.4),
            "pv_pcl_map_encoder": dict(lr_mult=0.4),
            "img_map_pv_fuser": dict(lr_mult=0.4),
            "bev_pcl_map_encoder": dict(lr_mult=0.4),

            #"depth_branch": dict(lr_mult=0.1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)

# ================== eval ========================
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=False,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)

evaluation = dict(interval=eval_epoch_interval, eval_mode=eval_mode)
load_from = 'work_dirs/dualviewmapdet_argoverse2_50m_vov99_fcos_704x480_ablationmain/epoch_80.pth'