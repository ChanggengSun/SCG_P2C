import bisect
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.logging import print_log
from mmengine.registry import DATASETS, LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals


@LOOPS.register_module()
class JointSeq5AlignedDualTrainLoop(BaseLoop):
    """Dual-dataloader joint loop with key alignment.

    Primary stream: tracking dataloader (defines iteration count).
    Secondary stream: flow dataloader (matched by scene_id+timestamps+candidate_id).
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        flow_dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
        max_flow_cache_size: int = 50000,
    ) -> None:
        super().__init__(runner, dataloader)
        self.flow_dataloader = self._build_dataloader(flow_dataloader)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, f'`max_epochs` should be int, got {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False

        self.max_flow_cache_size = max(1000, int(max_flow_cache_size))
        self._flow_iter = None
        self._flow_cache = OrderedDict()
        self._flow_epoch_len = max(1, len(self.flow_dataloader))

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be None.',
                logger='current',
                level=logging.WARNING,
            )

        self.dynamic_milestones, self.dynamic_intervals = calc_dynamic_intervals(
            self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    @staticmethod
    def _to_int(x, default=-1):
        if x is None:
            return int(default)
        if torch.is_tensor(x):
            return int(x.detach().item())
        try:
            return int(x)
        except Exception:
            return int(default)

    @staticmethod
    def _as_batch_list(data_batch):
        if isinstance(data_batch, list):
            return data_batch
        if isinstance(data_batch, tuple):
            return list(data_batch)
        return [data_batch]

    @staticmethod
    def _ensure_2d_tensor(x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _build_dataloader(self, dataloader_cfg_or_obj):
        if isinstance(dataloader_cfg_or_obj, DataLoader):
            return dataloader_cfg_or_obj
        if not isinstance(dataloader_cfg_or_obj, dict):
            raise TypeError(f'Unsupported flow_dataloader type: {type(dataloader_cfg_or_obj)}')

        dataset_cfg = dataloader_cfg_or_obj.get('dataset', None)
        if dataset_cfg is None:
            raise KeyError('flow_dataloader.dataset is required.')
        dataset = DATASETS.build(dataset_cfg) if isinstance(dataset_cfg, dict) else dataset_cfg

        sampler_cfg = dataloader_cfg_or_obj.get('sampler', {}) or {}
        shuffle = bool(sampler_cfg.get('shuffle', False))
        batch_size = int(dataloader_cfg_or_obj.get('batch_size', 1))
        num_workers = int(dataloader_cfg_or_obj.get('num_workers', 0))
        collate_fn = dataloader_cfg_or_obj.get('collate_fn', None)
        pin_memory = bool(dataloader_cfg_or_obj.get('pin_memory', False))
        persistent_workers = bool(dataloader_cfg_or_obj.get('persistent_workers', False))
        prefetch_factor = dataloader_cfg_or_obj.get('prefetch_factor', None)
        drop_last = bool(dataloader_cfg_or_obj.get('drop_last', False))
        if num_workers <= 0:
            persistent_workers = False
            prefetch_factor = None

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)

        return DataLoader(**loader_kwargs)

    def _sample_key(self, sample):
        ts = sample.get('timestamps', {})
        key = (
            str(sample.get('scene_id', '')),
            self._to_int(ts.get('pch3', None)),
            self._to_int(ts.get('pch2', None)),
            self._to_int(ts.get('pch1', None)),
            self._to_int(ts.get('pc0', None)),
            self._to_int(ts.get('pc1', None)),
            self._to_int(sample.get('candidate_id', 0), default=0),
        )
        return key

    def _reset_flow_state(self):
        self._flow_iter = iter(self.flow_dataloader)
        self._flow_cache.clear()

    def _cache_flow_batch(self, flow_batch):
        flow_samples = self._as_batch_list(flow_batch)
        for sample in flow_samples:
            key = self._sample_key(sample)
            self._flow_cache[key] = sample
            self._flow_cache.move_to_end(key)
            while len(self._flow_cache) > self.max_flow_cache_size:
                self._flow_cache.popitem(last=False)

    def _ensure_flow_keys(self, needed_keys):
        missing = [k for k in needed_keys if k not in self._flow_cache]
        if len(missing) == 0:
            return

        pulls = 0
        max_pulls = self._flow_epoch_len + 1
        while len(missing) > 0 and pulls < max_pulls:
            try:
                flow_batch = next(self._flow_iter)
            except StopIteration:
                self._flow_iter = iter(self.flow_dataloader)
                pulls += 1
                continue

            self._cache_flow_batch(flow_batch)
            pulls += 1
            missing = [k for k in needed_keys if k not in self._flow_cache]

    def _build_flow_fallback(self, track_sample):
        t_inputs = track_sample['inputs']

        prev_list = t_inputs['track_prev_points']
        this_list = t_inputs['track_this_points']

        pch3 = self._ensure_2d_tensor(prev_list[0]).float()
        pch2 = self._ensure_2d_tensor(this_list[0]).float()
        pch1 = self._ensure_2d_tensor(this_list[1]).float()
        pc0 = self._ensure_2d_tensor(this_list[2]).float()
        pc1 = self._ensure_2d_tensor(this_list[3]).float()

        n0 = int(pc0.shape[0])
        query_points = pc0[:, :3] if pc0.shape[1] >= 3 else torch.zeros((n0, 3), dtype=torch.float32)

        flow_inputs = {
            'pc0': pc0,
            'pc1': pc1,
            'pch1': pch1,
            'pch2': pch2,
            'pch3': pch3,
            'query_points': query_points,
            'pose_flow': torch.zeros((n0, 3), dtype=torch.float32),
            'pose0': torch.eye(4, dtype=torch.float32),
            'pose1': torch.eye(4, dtype=torch.float32),
            'poseh1': torch.eye(4, dtype=torch.float32),
            'poseh2': torch.eye(4, dtype=torch.float32),
            'poseh3': torch.eye(4, dtype=torch.float32),
            'ego_motion': torch.eye(4, dtype=torch.float32),
        }
        flow_targets = {
            'gt_flow': torch.zeros((n0, 3), dtype=torch.float32),
            'flow_is_valid': torch.zeros((n0,), dtype=torch.bool),
            'flow_category_indices': torch.zeros((n0,), dtype=torch.long),
            'flow_instance_id': torch.zeros((n0,), dtype=torch.long),
            'flow_available': torch.tensor(False, dtype=torch.bool),
        }
        return flow_inputs, flow_targets

    def _merge_sample(self, track_sample, flow_sample):
        merged = {
            'task_type': 'joint_seq5',
            'scene_id': track_sample.get('scene_id', ''),
            'instance_token': track_sample.get('instance_token', ''),
            'candidate_id': track_sample.get('candidate_id', 0),
            'timestamps': track_sample.get('timestamps', {}),
        }
        merged_inputs = dict(track_sample['inputs'])
        merged_targets = dict(track_sample['data_samples'])

        if flow_sample is None:
            flow_inputs, flow_targets = self._build_flow_fallback(track_sample)
        else:
            flow_inputs = flow_sample.get('inputs', {})
            flow_targets = flow_sample.get('data_samples', {})
            if 'flow_available' not in flow_targets:
                flow_targets['flow_available'] = torch.tensor(True, dtype=torch.bool)

        flow_input_keys = [
            'pc0', 'pc1', 'pch1', 'pch2', 'pch3',
            'query_points', 'pose_flow', 'pose0', 'pose1',
            'poseh1', 'poseh2', 'poseh3', 'ego_motion',
        ]
        flow_target_keys = [
            'gt_flow', 'flow_is_valid', 'flow_category_indices',
            'flow_instance_id', 'flow_available',
        ]

        for k in flow_input_keys:
            if k in flow_inputs:
                merged_inputs[k] = flow_inputs[k]
        for k in flow_target_keys:
            if k in flow_targets:
                merged_targets[k] = flow_targets[k]

        merged['inputs'] = merged_inputs
        merged['data_samples'] = merged_targets
        return merged

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_train')
        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        self._reset_flow_state()

        for idx, track_batch in enumerate(self.dataloader):
            self.run_iter(idx, track_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        track_samples = self._as_batch_list(data_batch)
        needed_keys = [self._sample_key(sample) for sample in track_samples]
        self._ensure_flow_keys(needed_keys)

        merged_samples = []
        matched = 0
        for sample, key in zip(track_samples, needed_keys):
            flow_sample = self._flow_cache.pop(key, None)
            if flow_sample is not None:
                matched += 1
            merged_samples.append(self._merge_sample(sample, flow_sample))

        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=merged_samples)
        outputs = self.runner.model.train_step(merged_samples, optim_wrapper=self.runner.optim_wrapper)
        if isinstance(outputs, dict):
            ref_tensor = None
            for _, v in outputs.items():
                if torch.is_tensor(v):
                    ref_tensor = v
                    break
            if ref_tensor is None:
                ref_tensor = torch.tensor(0.0)
            outputs['aligned_flow_matches'] = ref_tensor.new_tensor(float(matched))
            outputs['aligned_track_batch'] = ref_tensor.new_tensor(float(len(track_samples)))

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=merged_samples,
            outputs=outputs,
        )
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
