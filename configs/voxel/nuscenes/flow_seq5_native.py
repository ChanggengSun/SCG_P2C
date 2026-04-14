_base_ = '../../default_runtime.py'

flow_h5_dir = 'C:/develop/OpenSceneFlow/data/processed'
epoch_num = 10
batch_size = 32
val_batch_size = 8
point_cloud_range = [-4.8, -4.8, -2.0, 4.8, 4.8, 2.0]

from train import simple_collate_fn

custom_imports = dict(
    imports=['models', 'datasets'],
    allow_failed_imports=False,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='val/Dynamic/Mean', rule='less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

model = dict(
    type='P2PFlowSeq5Voxel',
    backbone=dict(
        type='VoxelNet',
        points_features=4,
        point_cloud_range=point_cloud_range,
        voxel_size=[0.075, 0.075, 0.15],
        grid_size=[27, 128, 128],
        output_channels=128,
    ),
    fuser=dict(type='BEVFuser'),
    flow_head=dict(
        type='DeltaFlowTemporalHead',
        in_channels=1024,
        hidden_channels=256,
        num_pairs=4,
    ),
    flow_loss=dict(type='DeFlowLoss'),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        input_dim=4,
    ),
)

train_dataset = dict(
    type='NuScenesFlowSeq5NativeDataset',
    path=flow_h5_dir,
    split='train',
    remove_ground=False,
    input_dim=4,
    history_frames=3,
)

val_dataset = dict(
    type='NuScenesFlowSeq5NativeDataset',
    path=flow_h5_dir,
    split='val',
    remove_ground=False,
    input_dim=4,
    history_frames=3,
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=simple_collate_fn,
)

val_dataloader = dict(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=simple_collate_fn,
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.00001),
    clip_grad=dict(max_norm=10, norm_type=2),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = None
