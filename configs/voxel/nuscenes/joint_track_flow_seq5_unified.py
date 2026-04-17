"""Unified single-dataloader config for joint tracking + flow training.

Key optimization vs joint_track_flow_seq5_sync.py:
    - ONE dataset (NuScenesJointSeq5Dataset) with track_source='h5'
      → reads point clouds from h5 for BOTH flow and tracking
      → eliminates redundant NuScenes SDK raw .bin reads
    - ONE dataloader → no duplicate I/O for overlapping scenes
    - JointSeq5UnifiedTrainLoop → feeds the same batch to both
      train_step_flow() and train_step_track()

Data output is IDENTICAL to the dual-loader version:
    - Flow fields: pc0, pc1, pch1-3, query_points, pose_flow, gt_flow, etc.
    - Track fields: track_prev_points, track_this_points, track_wlh,
      track_box_label, track_theta
"""
_base_ = '../../default_runtime.py'

raw_data_dir = 'C:/Users/SunChanggeng/Desktop/NuScenes_correct'
flow_h5_dir = 'C:/develop/OpenSceneFlow/data/processed'
category_name = 'Car'
flow_total_epochs = 30
track_total_epochs = 20
track_start_epoch = flow_total_epochs - track_total_epochs + 1
epoch_num = flow_total_epochs
batch_size = 128
point_cloud_range = [-4.8, -4.8, -2.0, 4.8, 4.8, 2.0]
box_aware = True
use_rot = False

from train import simple_collate_fn

custom_imports = dict(
    imports=['models', 'datasets', 'engine'],
    allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='precision',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

model = dict(
    type='P2PJointSeq5Voxel',
    backbone=dict(
        type='VoxelNet',
        points_features=4,
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.15],
        grid_size=[27, 128, 128],
        output_channels=128),
    fuser=dict(type='BEVFuser'),
    tracking_head=dict(
        type='VoxelHead',
        q_distribution='laplace',
        use_rot=use_rot,
        box_aware=box_aware),
    flow_head=dict(
        type='DeltaFlowTemporalHead',
        in_channels=1024,
        hidden_channels=256,
        num_pairs=1),
    flow_loss=dict(type='DeFlowLoss'),
    joint_pair_mode='pc0_pc1',
    flow_freeze_shared_backbone=True,
    flow_backbone_lr_mult=1.0,
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot,
        input_dim=4,
    ))

# ---- UNIFIED dataset: one dataset produces both flow + tracking data ----
unified_train_dataset = dict(
    type='NuScenesJointSeq5Dataset',
    path=flow_h5_dir,
    split='train',
    sidecar_file=f'joint_seq5_{category_name.lower()}.pkl',
    pair_mode='pc0_pc1',
    preloading=False,
    category_name=category_name,
    remove_ground=False,
    input_dim=4,
    history_frames=3,
    num_candidates=4,
    require_deltaflow_fields=False,
    # KEY CHANGE: read tracking point clouds from h5 (same source as flow)
    # instead of going back to raw NuScenes .bin files via SDK
    track_source='h5',
    # KEY CHANGE: load flow data too (was True in old track-only config)
    skip_flow_loading=False,
    track_cfg=dict(
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        input_dim=4,
        regular_pc=False,
        flip=True,
    ),
)

unified_train_dataloader = dict(
    dataset=unified_train_dataset,
    batch_size=batch_size,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    pin_memory=False,
    persistent_workers=True,
    collate_fn=simple_collate_fn,
)

# Legacy alias for Runner compatibility
train_dataloader = unified_train_dataloader

val_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='NuScenesDataset',
        path=raw_data_dir,
        split='val',
        category_name=category_name,
        preloading=True,
    ),
)

val_dataloader = dict(
    dataset=val_dataset,
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    pin_memory=False,
    persistent_workers=False,
    collate_fn=simple_collate_fn,
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(
        bias_decay_mult=0.,
        norm_decay_mult=0.),
)

train_cfg = dict(
    _delete_=True,
    type='JointSeq5UnifiedTrainLoop',
    # Single dataloader — no separate flow/track loaders needed
    dataloader=unified_train_dataloader,
    track_start_epoch=track_start_epoch,
    max_epochs=epoch_num,
    val_begin=track_start_epoch,
)

val_cfg = dict()
test_cfg = None
