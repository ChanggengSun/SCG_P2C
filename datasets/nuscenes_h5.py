import os
import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pickle
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from . import points_utils
from .data_classes import PointCloud
from .nuscenes import tracking_to_general_class
from .sampler import TrainSampler
from mmengine.registry import DATASETS


@DATASETS.register_module()
class NuScenesH5Dataset:
    def __init__(self,
                 path,
                 split,
                 h5_dir,
                 category_name="Car",
                 preloading=True,
                 preload_offset=10,
                 version='v1.0-trainval',
                 **kwargs):
        self.path = path
        self.split = split
        self.h5_dir = h5_dir
        self.category_name = category_name
        self.nusc = NuScenes(version=version, dataroot=path, verbose=False)
        self.version = version
        self.key_frame_only = True
        self.min_points = 1 if split == 'val' else False

        if not os.path.isdir(self.h5_dir):
            raise FileNotFoundError(f'H5 directory not found: {self.h5_dir}')

        self._sample_scene_cache = {}
        self._h5_timestamps_cache = {}

        self.track_instances = self.filter_instance(split, category_name.lower(), self.min_points)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        self._filter_tracklets_by_h5()

        self.preload_offset = preload_offset
        self.preloading = preloading
        if self.preloading:
            self.training_samples = self._load_data()

    def filter_instance(self, split, category_name=None, min_points=-1):
        if category_name is not None:
            general_classes = tracking_to_general_class[category_name]
        instances = []
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get('category', instance['category_token'])['name']
            if scene['name'] in scene_splits[split] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or instance_category in general_classes):
                instances.append(instance)
        return instances

    def _build_tracklet_anno(self):
        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        for instance in self.track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':
                ann_record = self.nusc.get('sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue

                track_anno.append({
                    "sample_data_lidar": sample_data_lidar,
                    "box_anno": ann_record
                })

            list_of_tracklet_anno.append(track_anno)
            list_of_tracklet_len.append(len(track_anno))
        return list_of_tracklet_anno, list_of_tracklet_len

    def _scene_name_from_anno(self, anno):
        if 'scene_name' in anno:
            return anno['scene_name']

        sample_token = anno['box_anno']['sample_token']
        if sample_token in self._sample_scene_cache:
            scene_name = self._sample_scene_cache[sample_token]
        else:
            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            scene_name = scene['name']
            self._sample_scene_cache[sample_token] = scene_name

        anno['scene_name'] = scene_name
        return scene_name

    def _timestamp_from_anno(self, anno):
        if 'timestamp' in anno:
            return anno['timestamp']
        timestamp = str(anno['sample_data_lidar']['timestamp'])
        anno['timestamp'] = timestamp
        return timestamp

    def _load_scene_timestamps(self, scene_name):
        if scene_name in self._h5_timestamps_cache:
            return self._h5_timestamps_cache[scene_name]

        h5_path = os.path.join(self.h5_dir, f'{scene_name}.h5')
        if not os.path.isfile(h5_path):
            self._h5_timestamps_cache[scene_name] = None
            return None

        try:
            with h5py.File(h5_path, 'r') as f:
                timestamps = set(f.keys())
            self._h5_timestamps_cache[scene_name] = timestamps
            return timestamps
        except OSError:
            self._h5_timestamps_cache[scene_name] = None
            return None

    def _filter_tracklets_by_h5(self):
        total_frames = 0
        h5_hit_frames = 0
        missing_file_frames = 0
        missing_timestamp_frames = 0

        filtered_tracklets = []
        filtered_lens = []

        for tracklet in self.tracklet_anno_list:
            kept = []
            for anno in tracklet:
                total_frames += 1
                scene_name = self._scene_name_from_anno(anno)
                timestamp = self._timestamp_from_anno(anno)
                scene_timestamps = self._load_scene_timestamps(scene_name)

                if scene_timestamps is None:
                    missing_file_frames += 1
                    continue

                if timestamp not in scene_timestamps:
                    missing_timestamp_frames += 1
                    continue

                kept.append(anno)
                h5_hit_frames += 1

            if len(kept) > 0:
                filtered_tracklets.append(kept)
                filtered_lens.append(len(kept))

        self.tracklet_anno_list = filtered_tracklets
        self.tracklet_len_list = filtered_lens

        print(
            f'[NuScenesH5Dataset] split={self.split}, category={self.category_name} | '
            f'total_frames={total_frames}, h5_hit_frames={h5_hit_frames}, '
            f'missing_file_frames={missing_file_frames}, '
            f'missing_timestamp_frames={missing_timestamp_frames}, '
            f'kept_tracklets={len(self.tracklet_anno_list)}'
        )

    def _load_data(self):
        print('preloading data into memory (h5)')
        h5_tag = os.path.basename(os.path.normpath(self.h5_dir))
        preload_data_path = os.path.join(
            self.path,
            f"preload_nuscenes_{self.category_name}_{self.split}_{self.version}_{self.preload_offset}_{self.min_points}_h5_{h5_tag}.dat"
        )

        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos (h5)')
            training_samples = []
            for i in range(len(self.tracklet_anno_list)):
                frames = []
                for anno in self.tracklet_anno_list[i]:
                    frame = self._get_frame_from_anno_data(anno)
                    if frame is not None:
                        frames.append(frame)
                if len(frames) > 0:
                    training_samples.append(frames)
            with open(preload_data_path, 'wb') as f:
                print(f'saving loaded data to {preload_data_path}')
                pickle.dump(training_samples, f)

        self.tracklet_len_list = [len(tracklet) for tracklet in training_samples]
        return training_samples

    def get_num_tracklets(self):
        return len(self.tracklet_anno_list) if not self.preloading else len(self.training_samples)

    def get_num_frames_total(self):
        return int(sum(self.tracklet_len_list))

    def get_num_frames_tracklet(self, tracklet_id):
        return self.tracklet_len_list[tracklet_id]

    def get_frames(self, seq_id, frame_ids):
        if self.preloading:
            frames = [self.training_samples[seq_id][f_id] for f_id in frame_ids]
        else:
            seq_annos = self.tracklet_anno_list[seq_id]
            frames = [self._get_frame_from_anno_data(seq_annos[f_id]) for f_id in frame_ids]

        return frames

    def _get_frame_from_anno_data(self, anno):
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])

        scene_name = self._scene_name_from_anno(anno)
        timestamp = self._timestamp_from_anno(anno)
        h5_path = os.path.join(self.h5_dir, f'{scene_name}.h5')

        if not os.path.isfile(h5_path):
            return None

        try:
            with h5py.File(h5_path, 'r') as f:
                if timestamp not in f:
                    return None
                group = f[timestamp]
                if 'lidar' not in group or 'pose' not in group:
                    return None
                lidar = group['lidar'][:].astype(np.float32)
                pose = group['pose'][:].astype(np.float64)
        except OSError:
            return None

        if lidar.ndim != 2 or lidar.shape[1] < 4:
            return None

        points = lidar.copy()
        points[:, :3] = points[:, :3] @ pose[:3, :3].T + pose[:3, 3]

        pc = PointCloud(points=points.T)
        if self.preload_offset > 0:
            pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)

        return {
            "pc": pc,
            "3d_bbox": bb,
            'meta': {
                'scene_name': scene_name,
                'timestamp': timestamp,
                'sample_data_lidar': anno['sample_data_lidar'],
                'box_anno': box_anno
            }
        }


@DATASETS.register_module()
class NuScenesAlignedH5TrackDataset:
    """Tracking sampler backed by joint H5 sidecar windows.

    Each sidecar record provides an aligned pc0 -> pc1 window in the same H5
    scene file.  This dataset converts that record directly into the input
    format consumed by the original single-task P2PVoxel model, so the old
    project can train on the same aligned H5 windows as the joint project
    without introducing the flow branch.
    """

    def __init__(
        self,
        path,
        split='train',
        sidecar_file=None,
        category_name='Car',
        preloading=True,
        sample_ratio=0.2,
        sample_seed=0,
        num_candidates=4,
        input_dim=4,
        cfg=None,
        use_world_coordinates=True,
        max_retry=10,
        use_projected_sweeps=False,
        num_projected_sweeps=9,
        projected_sweep_mode='past',
        max_sweep_dt=0.5,
        projected_cache_tag='h5_sweep_projected_v1',
        **kwargs,
    ):
        self.path = Path(path)
        self.split = str(split)
        self.category_name = str(category_name)
        self.preloading = bool(preloading)
        self.sample_ratio = float(sample_ratio)
        self.sample_seed = int(sample_seed)
        self.num_candidates = max(1, int(num_candidates))
        self.input_dim = int(input_dim)
        self.use_world_coordinates = bool(use_world_coordinates)
        self.max_retry = max(1, int(max_retry))
        self.use_projected_sweeps = bool(use_projected_sweeps)
        self.num_projected_sweeps = max(0, int(num_projected_sweeps))
        self.projected_sweep_mode = str(projected_sweep_mode).lower()
        self.max_sweep_dt = float(max_sweep_dt)
        self.projected_cache_tag = str(projected_cache_tag)
        if self.projected_sweep_mode not in ('past',):
            raise ValueError(f'Unsupported projected_sweep_mode: {self.projected_sweep_mode}')
        self._scene_timestamp_cache = {}

        if not (0.0 < self.sample_ratio <= 1.0):
            raise ValueError(f'sample_ratio must be in (0, 1], got {self.sample_ratio}')

        split_dir = self.path / self.split
        self.directory = split_dir if split_dir.is_dir() else self.path
        sidecar_name = sidecar_file or f'joint_seq5_{self.category_name.lower()}.pkl'
        self.sidecar_path = self.directory / sidecar_name
        if not self.sidecar_path.exists():
            raise FileNotFoundError(f'Sidecar file not found: {self.sidecar_path}')

        with open(self.sidecar_path, 'rb') as f:
            self.samples = pickle.load(f)

        orig_count = len(self.samples)
        self.sample_indices = np.arange(orig_count, dtype=np.int64)
        if self.sample_ratio < 1.0 and orig_count > 0:
            keep_count = max(1, int(round(orig_count * self.sample_ratio)))
            rng = np.random.RandomState(self.sample_seed)
            keep_indices = np.sort(rng.choice(orig_count, size=keep_count, replace=False))
            self.sample_indices = keep_indices.astype(np.int64)
            self.samples = [self.samples[int(i)] for i in keep_indices]
            print(
                f'[NuScenesAlignedH5TrackDataset] sample_ratio={self.sample_ratio:.4f}, '
                f'kept {len(self.samples)}/{orig_count} windows (seed={self.sample_seed}).'
            )
        self.sample_index_hash = hashlib.sha1(self.sample_indices.tobytes()).hexdigest()[:12]
        print(
            f'[NuScenesAlignedH5TrackDataset] sidecar_index_hash={self.sample_index_hash} '
            f'from {len(self.sample_indices)}/{orig_count} original windows.'
        )
        if self.use_projected_sweeps:
            print(
                '[NuScenesAlignedH5TrackDataset] projected_sweeps enabled: '
                f'mode={self.projected_sweep_mode}, '
                f'num_past={self.num_projected_sweeps}, '
                f'max_dt={self.max_sweep_dt:.3f}s, '
                f'cache_tag={self.projected_cache_tag}. '
                'Points and boxes are returned in each keyframe LiDAR coordinate system.'
            )

        if len(self.samples) == 0:
            raise RuntimeError(f'No aligned H5 windows in sidecar: {self.sidecar_path}')

        default_cfg = dict(
            num_candidates=self.num_candidates,
            target_thr=None,
            search_thr=5,
            point_cloud_range=[-4.8, -4.8, -2.0, 4.8, 4.8, 2.0],
            input_dim=self.input_dim,
            regular_pc=False,
            flip=True,
        )
        if cfg is not None:
            default_cfg.update(dict(cfg))
        self.track_cfg = SimpleNamespace(**default_cfg)

        self._cache = None
        if self.preloading:
            print('preloading aligned H5 tracking windows')
            self._cache = [self._load_core(i) for i in range(len(self.samples))]
            print(f'loaded {len(self._cache)} aligned windows into memory')

    def __len__(self):
        return len(self.samples) * self.num_candidates

    @staticmethod
    def _ensure_pose(pose):
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (4, 4):
            return pose
        out = np.eye(4, dtype=np.float32)
        if pose.ndim == 2:
            h = min(4, pose.shape[0])
            w = min(4, pose.shape[1])
            out[:h, :w] = pose[:h, :w]
        return out

    def _fit_input_dim(self, pc):
        pc = np.asarray(pc, dtype=np.float32)
        if pc.shape[1] >= self.input_dim:
            return pc[:, :self.input_dim]
        pad = np.zeros((pc.shape[0], self.input_dim - pc.shape[1]), dtype=np.float32)
        return np.concatenate([pc, pad], axis=1)

    def _read_frame_points_pose(self, h5_file, timestamp):
        timestamp = str(timestamp)
        if timestamp not in h5_file:
            raise KeyError(f'Timestamp {timestamp} not found in {h5_file.filename}')

        group = h5_file[timestamp]
        if 'lidar' in group:
            points = group['lidar'][:].astype(np.float32)
        elif 'pc0' in group:
            points = group['pc0'][:].astype(np.float32)
        else:
            raise KeyError(f'No lidar/pc0 dataset in {h5_file.filename}:{timestamp}')

        if 'pose' in group:
            pose = self._ensure_pose(group['pose'][:])
        elif 'pose0' in group:
            pose = self._ensure_pose(group['pose0'][:])
        else:
            pose = np.eye(4, dtype=np.float32)

        return self._fit_input_dim(points), pose

    @staticmethod
    def _apply_pose(points, pose):
        points = np.asarray(points, dtype=np.float32).copy()
        pose = np.asarray(pose, dtype=np.float32)
        points[:, :3] = points[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        return points

    def _load_frame(self, h5_file, timestamp):
        points, pose = self._read_frame_points_pose(h5_file, timestamp)
        if self.use_world_coordinates:
            points = self._apply_pose(points, pose)
        return points

    def _scene_timestamps(self, h5_file):
        filename = h5_file.filename
        if filename in self._scene_timestamp_cache:
            return self._scene_timestamp_cache[filename]

        timestamps = []
        for key in h5_file.keys():
            try:
                timestamps.append(int(key))
            except (TypeError, ValueError):
                continue
        timestamps.sort()
        self._scene_timestamp_cache[filename] = timestamps
        return timestamps

    def _select_projected_timestamps(self, h5_file, key_timestamp):
        key_str = str(key_timestamp)
        if not self.use_projected_sweeps or self.num_projected_sweeps <= 0:
            return [key_str]

        try:
            key_int = int(key_str)
        except (TypeError, ValueError):
            return [key_str]

        max_dt_us = int(round(max(0.0, self.max_sweep_dt) * 1_000_000))
        past = []
        for ts in self._scene_timestamps(h5_file):
            dt = key_int - ts
            if dt <= 0:
                continue
            if max_dt_us > 0 and dt > max_dt_us:
                continue
            past.append(ts)
        past = past[-self.num_projected_sweeps:]
        return [str(ts) for ts in past] + [key_str]

    def _load_projected_frame(self, h5_file, key_timestamp):
        key_str = str(key_timestamp)
        _, key_pose = self._read_frame_points_pose(h5_file, key_str)
        inv_key_pose = np.linalg.inv(key_pose).astype(np.float32)
        timestamps = self._select_projected_timestamps(h5_file, key_str)

        parts = []
        used = []
        for timestamp in timestamps:
            points, pose = self._read_frame_points_pose(h5_file, timestamp)
            if str(timestamp) == key_str:
                projected = points.copy()
            else:
                transform = (inv_key_pose @ pose).astype(np.float32)
                projected = self._apply_pose(points, transform)
            parts.append(projected)
            used.append(str(timestamp))

        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = np.concatenate(parts, axis=0)
        return merged.astype(np.float32, copy=False), key_pose, used

    @staticmethod
    def _dict_to_box(box_dict):
        return Box(
            box_dict['center'],
            box_dict['size'],
            Quaternion(box_dict['rotation']),
            name=box_dict.get('category_name', None),
            token=box_dict.get('token', None),
        )

    @staticmethod
    def _orthonormal_rotation(rotation):
        rotation = np.asarray(rotation, dtype=np.float64)
        u, _, vh = np.linalg.svd(rotation)
        out = u @ vh
        if np.linalg.det(out) < 0:
            u[:, -1] *= -1
            out = u @ vh
        return out

    def _dict_to_box_in_lidar(self, box_dict, lidar_pose):
        box = self._dict_to_box(box_dict)
        lidar_pose = np.asarray(lidar_pose, dtype=np.float32)
        box.translate(-lidar_pose[:3, 3])
        rotation = self._orthonormal_rotation(lidar_pose[:3, :3].T)
        box.rotate(Quaternion(matrix=rotation))
        return box

    def _load_core(self, sample_idx):
        sample = self.samples[sample_idx]
        scene_id = sample['scene_id']
        h5_path = self.directory / f'{scene_id}.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f'H5 file not found: {h5_path}')

        timestamps = sample['timestamps']
        boxes = sample['boxes_world']
        projected_sweeps = None
        with h5py.File(h5_path, 'r') as f:
            if self.use_projected_sweeps:
                prev_points, prev_pose, prev_used = self._load_projected_frame(f, timestamps['pc0'])
                this_points, this_pose, this_used = self._load_projected_frame(f, timestamps['pc1'])
                prev_box = self._dict_to_box_in_lidar(boxes['pc0'], prev_pose)
                this_box = self._dict_to_box_in_lidar(boxes['pc1'], this_pose)
                projected_sweeps = dict(pc0=prev_used, pc1=this_used)
            else:
                prev_points = self._load_frame(f, timestamps['pc0'])
                this_points = self._load_frame(f, timestamps['pc1'])
                prev_box = self._dict_to_box(boxes['pc0'])
                this_box = self._dict_to_box(boxes['pc1'])

        return dict(
            prev_frame=dict(
                pc=PointCloud(prev_points.T),
                **{'3d_bbox': prev_box},
            ),
            this_frame=dict(
                pc=PointCloud(this_points.T),
                **{'3d_bbox': this_box},
            ),
            scene_id=scene_id,
            timestamps=timestamps,
            projected_sweeps=projected_sweeps,
            instance_token=sample.get('instance_token', ''),
        )

    def _get_core(self, sample_idx):
        if self._cache is not None:
            return self._cache[sample_idx]
        return self._load_core(sample_idx)

    @staticmethod
    def _process_core(core, candidate_id, cfg):
        return TrainSampler.processing(
            dict(
                prev_frame=core['prev_frame'],
                this_frame=core['this_frame'],
                candidate_id=int(candidate_id),
            ),
            cfg,
        )

    def __getitem__(self, index):
        last_error = None
        current_index = int(index)

        for _ in range(self.max_retry):
            sample_idx = current_index // self.num_candidates
            candidate_id = current_index % self.num_candidates
            core = self._get_core(sample_idx)
            try:
                return self._process_core(core, candidate_id, self.track_cfg)
            except AssertionError as exc:
                last_error = exc
                current_index = int(np.random.randint(0, len(self)))

        fallback_cfg = SimpleNamespace(**vars(self.track_cfg))
        fallback_cfg.target_thr = None
        fallback_cfg.search_thr = None
        sample_idx = current_index // self.num_candidates
        candidate_id = current_index % self.num_candidates
        core = self._get_core(sample_idx)
        try:
            return self._process_core(core, candidate_id, fallback_cfg)
        except AssertionError:
            raise last_error
