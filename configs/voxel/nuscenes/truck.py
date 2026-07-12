_base_ = '../../default_runtime.py'
data_dir = 'C:/Users/SunChanggeng/Desktop/NuScenes_correct'
h5_train_dir = 'C:/develop/OpenSceneFlow/data/processed/train'
h5_val_dir = 'C:/develop/OpenSceneFlow/data/processed/val'
category_name = 'Truck'
batch_size = 256
point_cloud_range = [-9.6, -9.6, -3.0, 9.6, 9.6, 3.0]
box_aware = True
use_rot = False
# ========== 浠?train.py 瀵煎叆 collate 鍑芥暟 ==========
from train import simple_collate_fn


model = dict(
    type='P2PVoxel',
    backbone=dict(type='VoxelNet',
                  points_features=4,
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.15, 0.15, 0.3],
                  grid_size=[21, 128, 128],
                  output_channels=128
                  ),
    fuser=dict(type='BEVFuser'),
    head=dict(
        type='VoxelHead',
        q_distribution='laplace',  # ['laplace', 'gaussian']
        use_rot=use_rot,
        box_aware=box_aware
    ),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot,
        input_dim = 4 #xyz,intensity
    )
)

train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='NuScenesH5Dataset',
        path=data_dir,
        h5_dir=h5_train_dir,
        split='train_track',
        category_name=category_name,
        preloading=True,
        preload_offset=10,
    ),
    cfg=dict(
        num_candidates=4,
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        input_dim = 4, #xyz,intensity
        regular_pc=False,
        flip=True
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='NuScenesH5Dataset',
        path=data_dir,
        h5_dir=h5_val_dir,
        split='val',
        category_name=category_name,
        preloading=True
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=2,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=simple_collate_fn,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=simple_collate_fn,
)

