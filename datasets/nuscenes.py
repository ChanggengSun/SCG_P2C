import os
from pathlib import Path

import numpy as np
import pickle
import nuscenes
import h5py
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes

from pyquaternion import Quaternion

from . import points_utils
from .data_classes import PointCloud
from mmengine.registry import DATASETS

general_to_tracking_class = {"animal": "void / ignore",
                             "human.pedestrian.personal_mobility": "void / ignore",
                             "human.pedestrian.stroller": "void / ignore",
                             "human.pedestrian.wheelchair": "void / ignore",
                             "movable_object.barrier": "void / ignore",
                             "movable_object.debris": "void / ignore",
                             "movable_object.pushable_pullable": "void / ignore",
                             "movable_object.trafficcone": "void / ignore",
                             "static_object.bicycle_rack": "void / ignore",
                             "vehicle.emergency.ambulance": "void / ignore",
                             "vehicle.emergency.police": "void / ignore",
                             "vehicle.construction": "void / ignore",
                             "vehicle.bicycle": "bicycle",
                             "vehicle.bus.bendy": "bus",
                             "vehicle.bus.rigid": "bus",
                             "vehicle.car": "car",
                             "vehicle.motorcycle": "motorcycle",
                             "human.pedestrian.adult": "pedestrian",
                             "human.pedestrian.child": "pedestrian",
                             "human.pedestrian.construction_worker": "pedestrian",
                             "human.pedestrian.police_officer": "pedestrian",
                             "vehicle.trailer": "trailer",
                             "vehicle.truck": "truck", }

tracking_to_general_class = {
    'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                      'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                      'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'car': ['vehicle.car'],
    'motorcycle': ['vehicle.motorcycle'],
    'pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                   'human.pedestrian.police_officer'],
    'trailer': ['vehicle.trailer'],
    'truck': ['vehicle.truck']}


@DATASETS.register_module()
class NuScenesDataset:
    def __init__(self,
                 path,
                 split,
                 category_name="Car",
                 preloading=True,
                 preload_offset=10,
                 version='v1.0-trainval',
                 point_source='raw',
                 h5_data_dir=None,
                 **kwargs):
        self.path = path
        self.split = split
        self.category_name = category_name
        self.nusc = NuScenes(version=version, dataroot=path, verbose=False)
        self.version = version
        self._sample_scene_cache = {}
        self._h5_timestamps_cache = {}
        self.point_source = str(point_source).strip().lower()
        if self.point_source not in ('raw', 'h5'):
            raise ValueError(f'Unsupported point_source={point_source}, expected raw|h5')
        self.h5_data_dir = str(h5_data_dir) if h5_data_dir is not None else None
        self.h5_directory = None
        if self.point_source == 'h5':
            if self.h5_data_dir is None:
                raise ValueError('h5_data_dir is required when point_source=h5')
            h5_root = Path(self.h5_data_dir)
            split_dir = h5_root / str(self.split)
            self.h5_directory = split_dir if split_dir.is_dir() else h5_root
        self.key_frame_only = True
        self.min_points = 1 if split == 'val' else False
        self.track_instances = self.filter_instance(split, category_name.lower(), self.min_points)
        self.tracklet_anno_list, self.tracklet_len_list = self._build_tracklet_anno()
        if self.point_source == 'h5':
            self._filter_tracklets_by_h5()

        self.preload_offset = preload_offset
        self.preloading = preloading
        if self.preloading:
            self.training_samples = self._load_data()

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
            return str(int(anno['timestamp']))
        timestamp = str(anno['sample_data_lidar']['timestamp'])
        anno['timestamp'] = int(timestamp)
        return timestamp

    def _load_scene_timestamps(self, scene_name):
        if scene_name in self._h5_timestamps_cache:
            return self._h5_timestamps_cache[scene_name]

        h5_path = self.h5_directory / f'{scene_name}.h5'
        if not h5_path.exists():
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
            f'[NuScenesDataset-H5] split={self.split}, category={self.category_name} | '
            f'total_frames={total_frames}, h5_hit_frames={h5_hit_frames}, '
            f'missing_file_frames={missing_file_frames}, '
            f'missing_timestamp_frames={missing_timestamp_frames}, '
            f'kept_tracklets={len(self.tracklet_anno_list)}'
        )

    def filter_instance(self, split, category_name=None, min_points=-1):
        """
        This function is used to filter the tracklets.

        split: the dataset split
        category_name:
        min_points: the minimum number of points in the first bbox
        """
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
                    (category_name is None or category_name is not None and instance_category in general_classes):
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
                scene = self.nusc.get('scene', sample['scene_token'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append({
                    "sample_data_lidar": sample_data_lidar,
                    "box_anno": ann_record,
                    "scene_name": scene['name'],
                    "timestamp": sample_data_lidar['timestamp'],
                })

            list_of_tracklet_anno.append(track_anno)
            list_of_tracklet_len.append(len(track_anno))
        return list_of_tracklet_anno, list_of_tracklet_len

    def _load_data(self):
        print('preloading data into memory')
        source_suffix = ''
        if self.point_source == 'h5':
            source_suffix = f'_h5_{self.split}'
        preload_data_path = os.path.join(self.path,
                                         f"preload_nuscenes_{self.category_name}_{self.split}_{self.version}_{self.preload_offset}_{self.min_points}{source_suffix}.dat")
        if os.path.isfile(preload_data_path):
            print(f'loading from saved file {preload_data_path}.')
            with open(preload_data_path, 'rb') as f:
                training_samples = pickle.load(f)
        else:
            print('reading from annos')
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

    @staticmethod
    def _as_4x4_pose(pose):
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (4, 4):
            return pose
        eye = np.eye(4, dtype=np.float32)
        h = min(4, pose.shape[0])
        w = min(4, pose.shape[1])
        eye[:h, :w] = pose[:h, :w]
        return eye

    def _load_pc_from_h5(self, anno):
        scene_name = anno.get('scene_name', None)
        if scene_name is None:
            raise KeyError('Missing scene_name in annotation for h5 loading.')
        ts = str(int(anno.get('timestamp', anno['sample_data_lidar']['timestamp'])))
        h5_path = self.h5_directory / f'{scene_name}.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f'H5 scene file not found: {h5_path}')

        with h5py.File(h5_path, 'r') as f:
            if ts not in f:
                raise KeyError(f'Timestamp {ts} not found in {h5_path}')
            group = f[ts]
            if 'lidar' in group:
                pc = group['lidar'][:].astype(np.float32)
            elif 'pc0' in group:
                pc = group['pc0'][:].astype(np.float32)
            else:
                raise KeyError(f'No lidar/pc0 field in {h5_path}:{ts}')

            if 'pose' in group:
                pose = group['pose'][:]
            elif 'pose0' in group:
                pose = group['pose0'][:]
            else:
                raise KeyError(f'No pose/pose0 field in {h5_path}:{ts}')

        pose = self._as_4x4_pose(pose)
        xyz = pc[:, :3]
        xyz_global = xyz @ pose[:3, :3].T + pose[:3, 3]
        if pc.shape[1] > 3:
            pc_global = np.concatenate([xyz_global, pc[:, 3:]], axis=1)
        else:
            pc_global = xyz_global

        return PointCloud(points=pc_global.T)

    def get_num_tracklets(self):
        return len(self.training_samples) if self.preloading else len(self.tracklet_anno_list)

    def get_num_frames_total(self):
        return sum(self.tracklet_len_list)

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
        sample_data_lidar = anno['sample_data_lidar']
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])
        if self.point_source == 'h5':
            try:
                pc = self._load_pc_from_h5(anno)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                return None
        else:
            pcl_path = os.path.join(self.path, sample_data_lidar['filename'])
            pc = LidarPointCloud.from_file(pcl_path)

            cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))

            poserecord = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pc.translate(np.array(poserecord['translation']))
            pc = PointCloud(points=pc.points)
        if self.preload_offset > 0:
            pc = points_utils.crop_pc_axis_aligned(pc, bb, offset=self.preload_offset)
        return {"pc": pc, "3d_bbox": bb, 'meta': anno}
