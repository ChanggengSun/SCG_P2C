import argparse
import pickle
from pathlib import Path

import h5py
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def _normalize_category_name(category_name: str) -> str:
    return str(category_name).strip().lower()


def _scene_names_for_split(mode: str, split_name: str):
    split_name = str(split_name).strip().lower()
    if mode == 'v1.0-trainval':
        if split_name == 'train':
            return set(splits.train)
        if split_name == 'val':
            return set(splits.val)
        raise ValueError(f'Unsupported split_name={split_name} for mode={mode}')
    if mode == 'v1.0-mini':
        if split_name == 'train':
            return set(splits.mini_train)
        if split_name == 'val':
            return set(splits.mini_val)
        raise ValueError(f'Unsupported split_name={split_name} for mode={mode}')
    if mode == 'v1.0-test':
        if split_name == 'test':
            return set(splits.test)
        raise ValueError(f'Unsupported split_name={split_name} for mode={mode}')
    raise ValueError(f'Unsupported mode={mode}')


def _tracking_to_general_class_map():
    return {
        'bicycle': {'vehicle.bicycle'},
        'bus': {'vehicle.bus.bendy', 'vehicle.bus.rigid'},
        'car': {'vehicle.car'},
        'motorcycle': {'vehicle.motorcycle'},
        'pedestrian': {
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.police_officer',
        },
        'trailer': {'vehicle.trailer'},
        'truck': {'vehicle.truck'},
    }


def _load_flow_index(split_dir: Path):
    index_file = split_dir / 'index_flow.pkl'
    if not index_file.exists():
        raise FileNotFoundError(f'Flow index file not found: {index_file}')
    with open(index_file, 'rb') as f:
        index = pickle.load(f)

    flow_ts_by_scene = {}
    for scene_id, ts in index:
        ts_int = int(ts)
        flow_ts_by_scene.setdefault(scene_id, set()).add(ts_int)
    return flow_ts_by_scene


def _load_scene_timestamps(split_dir: Path, scene_id: str):
    h5_file = split_dir / f'{scene_id}.h5'
    if not h5_file.exists():
        return None, None
    with h5py.File(h5_file, 'r') as f:
        all_ts = sorted(int(k) for k in f.keys())
    all_ts_set = set(all_ts)
    return all_ts, all_ts_set


def _anno_to_box_dict(ann):
    return {
        'center': list(ann['translation']),
        'size': list(ann['size']),
        'rotation': list(ann['rotation']),
        'category_name': ann['category_name'],
        'sample_token': ann['sample_token'],
    }


def build_sidecar(data_dir: str,
                  flow_h5_dir: str,
                  split_name: str = 'train',
                  mode: str = 'v1.0-trainval',
                  category_name: str = 'Car',
                  history_frames: int = 3,
                  sample_stride: int = 1,
                  output_suffix: str = ''):
    history_frames = max(0, int(history_frames))
    sample_stride = max(1, int(sample_stride))
    output_suffix = str(output_suffix).strip()
    if history_frames != 3:
        print(f'[WARN] history_frames={history_frames} is supported, but current training assumes 3.')

    split_dir = Path(flow_h5_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f'Split directory not found: {split_dir}')

    nusc = NuScenes(version=mode, dataroot=data_dir, verbose=False)
    scene_name_set = _scene_names_for_split(mode, split_name)

    category_key = _normalize_category_name(category_name)
    class_map = _tracking_to_general_class_map()
    if category_key not in class_map:
        raise ValueError(f'Unsupported category_name={category_name}. Supported: {list(class_map.keys())}')
    target_general_names = class_map[category_key]

    flow_ts_by_scene = _load_flow_index(split_dir)

    scene_token_to_name = {s['token']: s['name'] for s in nusc.scene}

    sidecar_records = []
    windows_per_scene = {}

    for instance in tqdm(nusc.instance, ncols=100, desc='Build sidecar windows'):
        category_name_full = nusc.get('category', instance['category_token'])['name']
        if category_name_full not in target_general_names:
            continue

        ann_map = {}
        ann_token = instance['first_annotation_token']
        instance_scene_name = None

        while ann_token:
            ann = nusc.get('sample_annotation', ann_token)
            sample = nusc.get('sample', ann['sample_token'])
            scene_name = scene_token_to_name[sample['scene_token']]
            ann_token = ann['next']

            if scene_name not in scene_name_set:
                continue

            # IMPORTANT:
            # OpenSceneFlow h5/index timestamps are keyed by LIDAR_TOP sample_data timestamp,
            # not by sample table timestamp. Use lidar timestamp for strict alignment.
            lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ts = int(lidar_sd['timestamp'])
            ann_map[ts] = ann
            instance_scene_name = scene_name

        if instance_scene_name is None or len(ann_map) < 5:
            continue

        if instance_scene_name not in flow_ts_by_scene:
            continue

        all_ts, all_ts_set = _load_scene_timestamps(split_dir, instance_scene_name)
        if all_ts is None:
            continue

        # Build windows on annotation keyframe chain instead of all h5 timestamps.
        # h5 contains dense sweep timestamps, while annotations are sparse keyframes.
        # Therefore window must be sampled from sorted annotation timestamps.
        flow_ts_set = flow_ts_by_scene[instance_scene_name]
        ann_ts_sorted = sorted(ann_map.keys())
        if len(ann_ts_sorted) < history_frames + 2:
            continue

        instance_records = []
        for idx in range(history_frames, len(ann_ts_sorted) - 1):
            ts0 = ann_ts_sorted[idx]
            if ts0 not in flow_ts_set:
                continue

            ts_window = ann_ts_sorted[idx - history_frames: idx + 2]
            if len(ts_window) != history_frames + 2:
                continue

            # Guard against h5/raw mismatch.
            if not all(ts in all_ts_set for ts in ts_window):
                continue

            boxes_world = [_anno_to_box_dict(ann_map[ts]) for ts in ts_window]
            instance_records.append({
                'scene_id': instance_scene_name,
                'instance_token': instance['token'],
                'timestamps': {
                    'pch3': ts_window[0],
                    'pch2': ts_window[1],
                    'pch1': ts_window[2],
                    'pc0': ts_window[3],
                    'pc1': ts_window[4],
                },
                'boxes_world': {
                    'pch3': boxes_world[0],
                    'pch2': boxes_world[1],
                    'pch1': boxes_world[2],
                    'pc0': boxes_world[3],
                    'pc1': boxes_world[4],
                },
                'category_name': category_name,
            })

        if sample_stride > 1:
            instance_records = instance_records[::sample_stride]

        sidecar_records.extend(instance_records)
        if len(instance_records) > 0:
            windows_per_scene[instance_scene_name] = (
                windows_per_scene.get(instance_scene_name, 0) + len(instance_records)
            )

    sidecar_name = f'joint_seq5_{category_key}.pkl'
    if output_suffix:
        sidecar_name = f'joint_seq5_{category_key}_{output_suffix}.pkl'
    sidecar_path = split_dir / sidecar_name
    with open(sidecar_path, 'wb') as f:
        pickle.dump(sidecar_records, f)

    print(f'[INFO] Saved sidecar: {sidecar_path}')
    print(f'[INFO] Total windows: {len(sidecar_records)}')
    print(f'[INFO] Scenes with windows: {len(windows_per_scene)}')


def parse_args():
    parser = argparse.ArgumentParser(description='Build 5-frame sidecar index for joint tracking+flow training.')
    parser.add_argument('--data_dir', type=str, required=True, help='NuScenes raw root dir')
    parser.add_argument('--flow_h5_dir', type=str, required=True, help='Flow h5 root dir containing train/val folders')
    parser.add_argument('--split_name', type=str, default='train', help='split name: train/val/test')
    parser.add_argument('--mode', type=str, default='v1.0-trainval', help='NuScenes version')
    parser.add_argument('--category_name', type=str, default='Car', help='tracking category')
    parser.add_argument('--history_frames', type=int, default=3, help='history frame count (expected 3)')
    parser.add_argument('--sample_stride', type=int, default=1, help='keep one every N windows per instance')
    parser.add_argument('--output_suffix', type=str, default='', help='optional suffix in output sidecar filename')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_sidecar(
        data_dir=args.data_dir,
        flow_h5_dir=args.flow_h5_dir,
        split_name=args.split_name,
        mode=args.mode,
        category_name=args.category_name,
        history_frames=args.history_frames,
        sample_stride=args.sample_stride,
        output_suffix=args.output_suffix,
    )
