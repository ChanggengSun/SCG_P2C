import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mmengine.registry import DATASETS


@DATASETS.register_module()
class NuScenesFlowSeq5NativeDataset(Dataset):
    """Flow-only seq5 dataset from native OpenSceneFlow processed h5 files.

    Frame order is fixed as: pch3, pch2, pch1, pc0, pc1
    pc0 comes from index_flow.pkl and must have flow supervision.
    """

    def __init__(
        self,
        path,
        split='train',
        remove_ground=False,
        input_dim=4,
        history_frames=3,
        **kwargs,
    ):
        self.path = str(path)
        self.split = str(split)
        self.remove_ground = bool(remove_ground)
        self.input_dim = int(input_dim)
        self.history_frames = max(0, int(history_frames))

        split_dir = Path(self.path) / self.split
        self.directory = split_dir if split_dir.is_dir() else Path(self.path)
        if not self.directory.exists():
            raise FileNotFoundError(f'Flow directory not found: {self.directory}')

        flow_index_path = self.directory / 'index_flow.pkl'
        total_index_path = self.directory / 'index_total.pkl'
        if not flow_index_path.exists():
            raise FileNotFoundError(f'index_flow.pkl not found: {flow_index_path}')
        if not total_index_path.exists():
            raise FileNotFoundError(f'index_total.pkl not found: {total_index_path}')

        with open(flow_index_path, 'rb') as f:
            flow_index = pickle.load(f)
        with open(total_index_path, 'rb') as f:
            total_index = pickle.load(f)

        self._scene_ts = {}
        for item in total_index:
            scene_id, ts = item[0], str(item[1])
            self._scene_ts.setdefault(scene_id, []).append(ts)
        for scene_id in self._scene_ts:
            self._scene_ts[scene_id].sort(key=lambda x: int(x))

        self.samples = self._build_seq5_samples(flow_index)
        if len(self.samples) == 0:
            raise RuntimeError(f'No seq5 flow samples found in {self.directory}')

    def _build_seq5_samples(self, flow_index):
        samples = []
        for item in flow_index:
            scene_id, ts0 = item[0], str(item[1])
            ts_list = self._scene_ts.get(scene_id, None)
            if ts_list is None:
                continue

            try:
                idx = ts_list.index(ts0)
            except ValueError:
                continue

            if idx - 3 < 0 or idx + 1 >= len(ts_list):
                continue

            samples.append(
                {
                    'scene_id': scene_id,
                    'timestamps': {
                        'pch3': ts_list[idx - 3],
                        'pch2': ts_list[idx - 2],
                        'pch1': ts_list[idx - 1],
                        'pc0': ts_list[idx],
                        'pc1': ts_list[idx + 1],
                    },
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

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

    def __getitem__(self, index):
        sample = self.samples[index]
        scene_id = sample['scene_id']
        ts_map = sample['timestamps']

        h5_path = self.directory / f'{scene_id}.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f'H5 file not found: {h5_path}')

        flow_points = {}
        frame_poses = {}
        frame_gm = {}

        with h5py.File(h5_path, 'r') as f:
            for key in ['pch3', 'pch2', 'pch1', 'pc0', 'pc1']:
                pc, pose, gm = self._load_frame_from_h5(f, ts_map[key])
                flow_points[key] = pc
                frame_poses[key] = pose
                frame_gm[key] = gm

            group_pc0 = f[str(ts_map['pc0'])]
            gt_flow = group_pc0['flow'][:].astype(np.float32)
            flow_valid = group_pc0['flow_is_valid'][:].astype(bool)
            flow_category = group_pc0['flow_category_indices'][:].astype(np.int64)
            flow_instance = group_pc0['flow_instance_id'][:].astype(np.int64)
            ego_motion = group_pc0['ego_motion'][:].astype(np.float32)

        if self.remove_ground:
            keep_mask_pc0 = ~frame_gm['pc0']
            flow_points['pc0'] = flow_points['pc0'][keep_mask_pc0]
            gt_flow = gt_flow[keep_mask_pc0]
            flow_valid = flow_valid[keep_mask_pc0]
            flow_category = flow_category[keep_mask_pc0]
            flow_instance = flow_instance[keep_mask_pc0]
            for key in ['pch3', 'pch2', 'pch1', 'pc1']:
                keep_mask = ~frame_gm[key]
                flow_points[key] = flow_points[key][keep_mask]

        for key in flow_points:
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

        inputs = {
            'pc0': torch.as_tensor(flow_points['pc0'], dtype=torch.float32),
            'pc1': torch.as_tensor(flow_points['pc1'], dtype=torch.float32),
            'pch1': torch.as_tensor(flow_points['pch1'], dtype=torch.float32),
            'pch2': torch.as_tensor(flow_points['pch2'], dtype=torch.float32),
            'pch3': torch.as_tensor(flow_points['pch3'], dtype=torch.float32),
            'query_points': torch.as_tensor(flow_points['pc0'][:, :3], dtype=torch.float32),
            'pose_flow': torch.as_tensor(pose_flow, dtype=torch.float32),
            'pose0': torch.as_tensor(frame_poses['pc0'], dtype=torch.float32),
            'pose1': torch.as_tensor(frame_poses['pc1'], dtype=torch.float32),
            'poseh1': torch.as_tensor(frame_poses['pch1'], dtype=torch.float32),
            'poseh2': torch.as_tensor(frame_poses['pch2'], dtype=torch.float32),
            'poseh3': torch.as_tensor(frame_poses['pch3'], dtype=torch.float32),
            'ego_motion': torch.as_tensor(ego_motion, dtype=torch.float32),
        }

        data_samples = {
            'gt_flow': torch.as_tensor(gt_flow, dtype=torch.float32),
            'flow_is_valid': torch.as_tensor(flow_valid, dtype=torch.bool),
            'flow_category_indices': torch.as_tensor(flow_category, dtype=torch.long),
            'flow_instance_id': torch.as_tensor(flow_instance, dtype=torch.long),
        }

        return {
            'inputs': inputs,
            'data_samples': data_samples,
            'task_type': 'flow_seq5_native',
            'scene_id': scene_id,
            'timestamps': ts_map,
        }
