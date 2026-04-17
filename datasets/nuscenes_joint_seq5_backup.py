import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from mmengine.registry import DATASETS
from .data_classes import PointCloud, Box
from .sampler import TrainSampler


@DATASETS.register_module()
class NuScenesJointSeq5Dataset(Dataset):
    """Joint dataset for tracking+flow with optional pure two-frame mode."""

    def __init__(
        self,
        path,
        split='train',
        sidecar_file=None,
        category_name='Car',
        remove_ground=False,
        input_dim=4,
        history_frames=3,
        track_cfg=None,
        num_candidates=1,
        require_deltaflow_fields=False,
        track_data_dir=None,
        track_version='v1.0-trainval',
        track_source='raw',
        skip_flow_loading=False,
        pair_mode='seq5',
        preloading=False,
        **kwargs,
    ):
        self.path = str(path)
        self.split = str(split)
        self.category_name = str(category_name)
        self.remove_ground = bool(remove_ground)
        self.input_dim = int(input_dim)
        self.history_frames = max(0, int(history_frames))
        self.num_candidates = max(1, int(num_candidates))
        self.require_deltaflow_fields = bool(require_deltaflow_fields)
        self.preloading = bool(preloading)

        self.pair_mode = str(pair_mode).strip().lower()
        if self.pair_mode == 'pc0_pc1':
            self.frame_keys = ['pc0', 'pc1']
            self.pair_order = [('pc0', 'pc1')]
        else:
            self.frame_keys = ['pch3', 'pch2', 'pch1', 'pc0', 'pc1']
            self.pair_order = [('pch3', 'pch2'), ('pch2', 'pch1'), ('pch1', 'pc0'), ('pc0', 'pc1')]

        self.track_source = str(track_source).strip().lower()
        if self.track_source not in {'raw', 'h5'}:
            raise ValueError(f'Unsupported track_source={track_source}, expected raw|h5')
        self.skip_flow_loading = bool(skip_flow_loading)
        self.track_data_dir = str(track_data_dir) if track_data_dir is not None else None
        self.track_version = str(track_version)
        self.nusc = None
        if self.track_source == 'raw':
            if self.track_data_dir is None:
                raise ValueError('track_data_dir is required when track_source=raw')
            self.nusc = NuScenes(version=self.track_version, dataroot=self.track_data_dir, verbose=False)

        split_dir = Path(self.path) / self.split
        self.directory = split_dir if split_dir.is_dir() else Path(self.path)

        sidecar_name = sidecar_file if sidecar_file is not None else f'joint_seq5_{self.category_name.lower()}.pkl'
        self.sidecar_path = self.directory / sidecar_name
        if not self.sidecar_path.exists():
            raise FileNotFoundError(f'Sidecar file not found: {self.sidecar_path}')

        with open(self.sidecar_path, 'rb') as f:
            self.samples = pickle.load(f)

        if len(self.samples) == 0:
            raise RuntimeError(f'No seq5 samples in sidecar: {self.sidecar_path}')

        default_track_cfg = dict(
            target_thr=None,
            search_thr=5,
            point_cloud_range=[-4.8, -4.8, -2.0, 4.8, 4.8, 2.0],
            input_dim=self.input_dim,
            regular_pc=False,
            flip=True,
        )
        if track_cfg is not None:
            default_track_cfg.update(dict(track_cfg))
        self.track_cfg = SimpleNamespace(**default_track_cfg)

        self._cache = None
        if self.preloading:
            print('preloading joint sidecar dataset into memory')
            self._cache = [self._load_sample_core(i) for i in range(len(self.samples))]
            print(f'loaded {len(self._cache)} joint samples into memory')

    def __len__(self):
        return len(self.samples) * self.num_candidates

    @staticmethod
    def _load_group_array(group, preferred_key, fallback_key=None):
        if preferred_key in group:
            return group[preferred_key][:]
        if fallback_key is not None and fallback_key in group:
            return group[fallback_key][:]
        return None

    @staticmethod
    def _ensure_pose(pose):
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (4, 4):
            return pose
        eye = np.eye(4, dtype=np.float32)
        if pose.ndim == 2:
            h = min(4, pose.shape[0])
            w = min(4, pose.shape[1])
            eye[:h, :w] = pose[:h, :w]
        return eye

    def _fit_input_dim(self, pc):
        if self.input_dim <= pc.shape[1]:
            return pc[:, :self.input_dim]
        pad = np.zeros((pc.shape[0], self.input_dim - pc.shape[1]), dtype=pc.dtype)
        return np.concatenate([pc, pad], axis=1)

    @staticmethod
    def _ensure_min_points(pc, input_dim):
        if pc.shape[0] < 1:
            return np.zeros((1, input_dim), dtype=np.float32)
        return pc

    def _dict_to_box(self, box_dict):
        center = np.asarray(box_dict['center'], dtype=np.float32)
        size = np.asarray(box_dict['size'], dtype=np.float32)
        quat = Quaternion(box_dict['rotation'])
        return Box(center=center, size=size, orientation=quat, name=box_dict.get('category_name', None))

    @staticmethod
    def _compute_pose_flow(pc_xyz, ego_motion):
        return pc_xyz @ ego_motion[:3, :3].T + ego_motion[:3, 3] - pc_xyz

    def _load_frame_from_h5(self, h5_file, ts):
        ts = str(ts)
        if ts not in h5_file:
            raise KeyError(f'Timestamp {ts} not found in {h5_file.filename}')
        group = h5_file[ts]

        pc = self._load_group_array(group, 'pc0', 'lidar')
        if pc is None:
            raise KeyError(f'No point cloud field in group {ts}')
        pc = pc.astype(np.float32)

        pose = self._load_group_array(group, 'pose0', 'pose')
        if pose is None:
            raise KeyError(f'No pose field in group {ts}')
        pose = self._ensure_pose(pose)

        ground_mask = group['ground_mask'][:].astype(bool) if 'ground_mask' in group else np.zeros(pc.shape[0], dtype=bool)
        return pc, pose, ground_mask

    def _load_track_pc_from_raw(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pcl_path = Path(self.track_data_dir) / sample_data_lidar['filename']

        pc = LidarPointCloud.from_file(str(pcl_path))

        cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        pose_record = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
        pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
        pc.translate(np.array(pose_record['translation']))

        return pc.points.T.astype(np.float32)

    def _build_track_pair(self, prev_pc_np, this_pc_np, prev_box, this_box, candidate_id):
        prev_frame = {'pc': PointCloud(prev_pc_np.T), '3d_bbox': prev_box}
        this_frame = {'pc': PointCloud(this_pc_np.T), '3d_bbox': this_box}
        pair_data = {
            'prev_frame': prev_frame,
            'this_frame': this_frame,
            'candidate_id': int(candidate_id),
        }

        try:
            return TrainSampler.processing(pair_data, self.track_cfg)
        except AssertionError as e:
            msg = str(e)
            if 'not enough search points' not in msg and 'not enough target points' not in msg:
                raise
            fallback_cfg = SimpleNamespace(**vars(self.track_cfg))
            fallback_cfg.target_thr = None
            fallback_cfg.search_thr = None
            return TrainSampler.processing(pair_data, fallback_cfg)

    def _build_tracking_points(self, box_map, flow_points):
        track_points = {}
        if self.track_source == 'h5':
            for key in self.frame_keys:
                if key not in flow_points:
                    raise KeyError(f'{key} missing in flow_points for track_source=h5')
                track_points[key] = flow_points[key].copy()
            return track_points

        for key in self.frame_keys:
            sample_token = box_map[key].get('sample_token', None)
            if sample_token is None:
                raise KeyError(f'sample_token missing in sidecar boxes_world[{key}]')
            track_points[key] = self._load_track_pc_from_raw(sample_token)

        return track_points

    @staticmethod
    def _identity_pose():
        return np.eye(4, dtype=np.float32)

    def _load_sample_core(self, sample_idx):
        sample = self.samples[sample_idx]
        scene_id = sample['scene_id']
        ts_map = sample['timestamps']
        box_map = sample['boxes_world']

        flow_points = {}
        frame_poses = {}
        frame_gm = {}
        flow_available = not self.skip_flow_loading
        h5_path = self.directory / f'{scene_id}.h5'

        if not self.skip_flow_loading:
            try:
                if not h5_path.exists():
                    raise FileNotFoundError(f'H5 file not found: {h5_path}')

                with h5py.File(h5_path, 'r') as f:
                    for key in self.frame_keys:
                        pc, pose, gm = self._load_frame_from_h5(f, ts_map[key])
                        flow_points[key] = pc
                        frame_poses[key] = pose
                        frame_gm[key] = gm

                    group_pc0 = f[str(ts_map['pc0'])]
                    has_deltaflow_fields = ('pc0' in group_pc0 and 'pc1' in group_pc0 and 'pose0' in group_pc0 and 'pose1' in group_pc0)
                    if self.require_deltaflow_fields and not has_deltaflow_fields:
                        raise RuntimeError(
                            f'{scene_id}/{ts_map["pc0"]} missing DeltaFlow fields. Please regenerate h5 with --deltaflow_format True.')

                    gt_flow = group_pc0['flow'][:].astype(np.float32)
                    flow_valid = group_pc0['flow_is_valid'][:].astype(bool)
                    flow_category = group_pc0['flow_category_indices'][:].astype(np.int64)
                    flow_instance = group_pc0['flow_instance_id'][:].astype(np.int64)
                    ego_motion = group_pc0['ego_motion'][:].astype(np.float32)
            except Exception:
                if self.track_source != 'raw':
                    raise
                flow_available = False
                gt_flow = None
                flow_valid = None
                flow_category = None
                flow_instance = None
                ego_motion = self._identity_pose()
                frame_poses = {k: self._identity_pose() for k in self.frame_keys}
                frame_gm = {k: None for k in self.frame_keys}
        else:
            flow_available = False
            gt_flow = None
            flow_valid = None
            flow_category = None
            flow_instance = None
            ego_motion = self._identity_pose()
            frame_poses = {k: self._identity_pose() for k in self.frame_keys}
            frame_gm = {k: None for k in self.frame_keys}

        track_points = self._build_tracking_points(box_map, flow_points if flow_available else {})
        for key in self.frame_keys:
            track_points[key] = self._fit_input_dim(track_points[key])
            track_points[key] = self._ensure_min_points(track_points[key], self.input_dim)

        if not flow_available:
            flow_points = {k: track_points[k].copy() for k in self.frame_keys}
            n0 = flow_points['pc0'].shape[0]
            gt_flow = np.zeros((n0, 3), dtype=np.float32)
            flow_valid = np.zeros((n0,), dtype=bool)
            flow_category = np.zeros((n0,), dtype=np.int64)
            flow_instance = np.zeros((n0,), dtype=np.int64)
            pose_flow = np.zeros((n0, 3), dtype=np.float32)
        else:
            if self.remove_ground:
                keep_mask_pc0 = ~frame_gm['pc0']
                flow_points['pc0'] = flow_points['pc0'][keep_mask_pc0]
                gt_flow = gt_flow[keep_mask_pc0]
                flow_valid = flow_valid[keep_mask_pc0]
                flow_category = flow_category[keep_mask_pc0]
                flow_instance = flow_instance[keep_mask_pc0]
                for key in self.frame_keys:
                    if key == 'pc0':
                        continue
                    keep_mask = ~frame_gm[key]
                    flow_points[key] = flow_points[key][keep_mask]

            for key in self.frame_keys:
                flow_points[key] = self._fit_input_dim(flow_points[key])
                flow_points[key] = self._ensure_min_points(flow_points[key], self.input_dim)

            if flow_points['pc0'].shape[0] != gt_flow.shape[0]:
                min_len = min(flow_points['pc0'].shape[0], gt_flow.shape[0])
                flow_points['pc0'] = flow_points['pc0'][:min_len]
                gt_flow = gt_flow[:min_len]
                flow_valid = flow_valid[:min_len]
                flow_category = flow_category[:min_len]
                flow_instance = flow_instance[:min_len]
                if min_len < 1:
                    flow_points['pc0'] = np.zeros((1, self.input_dim), dtype=np.float32)
                    gt_flow = np.zeros((1, 3), dtype=np.float32)
                    flow_valid = np.zeros((1,), dtype=bool)
                    flow_category = np.zeros((1,), dtype=np.int64)
                    flow_instance = np.zeros((1,), dtype=np.int64)

            pose_flow = self._compute_pose_flow(flow_points['pc0'][:, :3], ego_motion).astype(np.float32)

        return dict(
            flow_points=flow_points,
            frame_poses=frame_poses,
            gt_flow=gt_flow,
            flow_valid=flow_valid,
            flow_category=flow_category,
            flow_instance=flow_instance,
            ego_motion=ego_motion,
            pose_flow=pose_flow,
            flow_available=bool(flow_available),
            track_points=track_points,
        )

    def _get_sample_core(self, sample_idx):
        if self._cache is not None:
            return self._cache[sample_idx]
        return self._load_sample_core(sample_idx)

    def __getitem__(self, index):
        sample_idx = index // self.num_candidates
        candidate_id = index % self.num_candidates
        sample = self.samples[sample_idx]
        box_map = sample['boxes_world']

        core = self._get_sample_core(sample_idx)
        flow_points = core['flow_points']
        frame_poses = core['frame_poses']
        gt_flow = core['gt_flow']
        flow_valid = core['flow_valid']
        flow_category = core['flow_category']
        flow_instance = core['flow_instance']
        ego_motion = core['ego_motion']
        pose_flow = core['pose_flow']
        flow_available = core['flow_available']
        track_points = core['track_points']

        track_prev_points = []
        track_this_points = []
        track_wlh = []
        track_box_label = []
        track_theta = []

        for prev_key, this_key in self.pair_order:
            prev_box = self._dict_to_box(box_map[prev_key])
            this_box = self._dict_to_box(box_map[this_key])
            pair = self._build_track_pair(
                track_points[prev_key].copy(),
                track_points[this_key].copy(),
                prev_box,
                this_box,
                candidate_id=candidate_id,
            )
            track_prev_points.append(pair['inputs']['prev_points'])
            track_this_points.append(pair['inputs']['this_points'])
            track_wlh.append(pair['inputs']['wlh'])
            track_box_label.append(pair['data_samples']['box_label'])
            track_theta.append(pair['data_samples']['theta'])

        inputs = {
            'track_prev_points': [torch.as_tensor(x, dtype=torch.float32) for x in track_prev_points],
            'track_this_points': [torch.as_tensor(x, dtype=torch.float32) for x in track_this_points],
            'track_wlh': [torch.as_tensor(x, dtype=torch.float32) for x in track_wlh],
            'pc0': torch.as_tensor(flow_points['pc0'], dtype=torch.float32),
            'pc1': torch.as_tensor(flow_points['pc1'], dtype=torch.float32),
            'query_points': torch.as_tensor(flow_points['pc0'][:, :3], dtype=torch.float32),
            'pose_flow': torch.as_tensor(pose_flow, dtype=torch.float32),
            'pose0': torch.as_tensor(frame_poses['pc0'], dtype=torch.float32),
            'pose1': torch.as_tensor(frame_poses['pc1'], dtype=torch.float32),
            'ego_motion': torch.as_tensor(ego_motion, dtype=torch.float32),
        }
        if 'pch1' in flow_points:
            inputs['pch1'] = torch.as_tensor(flow_points['pch1'], dtype=torch.float32)
            inputs['poseh1'] = torch.as_tensor(frame_poses['pch1'], dtype=torch.float32)
        if 'pch2' in flow_points:
            inputs['pch2'] = torch.as_tensor(flow_points['pch2'], dtype=torch.float32)
            inputs['poseh2'] = torch.as_tensor(frame_poses['pch2'], dtype=torch.float32)
        if 'pch3' in flow_points:
            inputs['pch3'] = torch.as_tensor(flow_points['pch3'], dtype=torch.float32)
            inputs['poseh3'] = torch.as_tensor(frame_poses['pch3'], dtype=torch.float32)

        data_samples = {
            'track_box_label': [torch.as_tensor(x, dtype=torch.float32) for x in track_box_label],
            'track_theta': [torch.as_tensor(x, dtype=torch.float32) for x in track_theta],
            'gt_flow': torch.as_tensor(gt_flow, dtype=torch.float32),
            'flow_is_valid': torch.as_tensor(flow_valid, dtype=torch.bool),
            'flow_category_indices': torch.as_tensor(flow_category, dtype=torch.long),
            'flow_instance_id': torch.as_tensor(flow_instance, dtype=torch.long),
            'flow_available': torch.as_tensor(flow_available, dtype=torch.bool),
        }

        return {
            'inputs': inputs,
            'data_samples': data_samples,
            'task_type': 'joint_seq5',
            'scene_id': sample['scene_id'],
            'instance_token': sample['instance_token'],
            'candidate_id': int(candidate_id),
            'timestamps': sample['timestamps'],
        }
