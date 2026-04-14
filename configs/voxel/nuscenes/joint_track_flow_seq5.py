_base_ = '../../default_runtime.py'

raw_data_dir = 'C:/Users/SunChanggeng/Desktop/NuScenes_correct'
flow_h5_dir = 'C:/develop/OpenSceneFlow/data/processed'
category_name = 'Car'
epoch_num = 50
batch_size = 1
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
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

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
        num_pairs=4),
    flow_loss=dict(type='DeFlowLoss'),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot,
        input_dim=4,
    ))

train_dataset = dict(
    type='NuScenesJointSeq5Dataset',
    path=flow_h5_dir,
    split='train',
    sidecar_file=f'joint_seq5_{category_name.lower()}_s10.pkl',
    category_name=category_name,
    remove_ground=False,
    input_dim=4,
    history_frames=3,
    num_candidates=4,
    require_deltaflow_fields=False,
    track_data_dir=raw_data_dir,
    track_version='v1.0-trainval',
    track_source='raw',
    track_cfg=dict(
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        input_dim=4,
        regular_pc=False,
        flip=True,
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=simple_collate_fn,
)

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
    collate_fn=simple_collate_fn,
)


optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)
train_cfg = dict(
    _delete_=True,
    type='JointSeq5TrainLoop',
    max_epochs=epoch_num,
)

val_cfg = dict()
test_cfg = None


