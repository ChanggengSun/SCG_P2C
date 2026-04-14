import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.logging import print_log
from mmengine.registry import DATASETS, LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals


@LOOPS.register_module()
class JointSeq5IndependentDualTrainLoop(BaseLoop):
    """Independent dual-dataloader loop without key alignment.

    Primary stream: flow dataloader (defines epoch iteration length).
    Secondary stream: tracking dataloader (cycled independently).

    Epoch behavior:
    - epoch < track_start_epoch: flow-only stage.
    - epoch >= track_start_epoch: flow stage then tracking stage per iter.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        track_dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        flow_dataloader: Optional[Union[DataLoader, Dict]] = None,
        track_start_epoch: int = 11,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        primary_loader = flow_dataloader if flow_dataloader is not None else dataloader
        super().__init__(runner, primary_loader)
        self.track_dataloader = self._build_dataloader(track_dataloader)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, f'`max_epochs` should be int, got {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0

        self.track_start_epoch = max(1, int(track_start_epoch))
        self.val_begin = max(1, int(val_begin))
        self.val_interval = max(1, int(val_interval))
        self.stop_training = False

        self._flow_iter = None
        self._track_iter = None

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

    def _build_dataloader(self, dataloader_cfg_or_obj):
        if isinstance(dataloader_cfg_or_obj, DataLoader):
            return dataloader_cfg_or_obj
        if not isinstance(dataloader_cfg_or_obj, dict):
            raise TypeError(f'Unsupported dataloader type: {type(dataloader_cfg_or_obj)}')

        dataset_cfg = dataloader_cfg_or_obj.get('dataset', None)
        if dataset_cfg is None:
            raise KeyError('dataloader.dataset is required.')
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

    @staticmethod
    def _as_batch_list(data_batch):
        if isinstance(data_batch, list):
            return data_batch
        if isinstance(data_batch, tuple):
            return list(data_batch)
        return [data_batch]

    @staticmethod
    def _merge_stage_outputs(flow_outputs, track_outputs):
        merged = {}
        ref_tensor = None
        stage_updates = 0.0

        def _consume(src):
            nonlocal ref_tensor, stage_updates
            if not isinstance(src, dict):
                return
            for k, v in src.items():
                if k == 'stage_updates' and torch.is_tensor(v):
                    stage_updates += float(v.detach().item())
                    if ref_tensor is None:
                        ref_tensor = v.detach()
                    continue
                merged[k] = v
                if ref_tensor is None and torch.is_tensor(v):
                    ref_tensor = v.detach()

        _consume(flow_outputs)
        _consume(track_outputs)

        if stage_updates > 0:
            if ref_tensor is None:
                ref_tensor = torch.tensor(0.0)
            merged['stage_updates'] = ref_tensor.new_tensor(stage_updates)

        return merged

    def _reset_stream_state(self):
        self._flow_iter = iter(self.dataloader)
        self._track_iter = iter(self.track_dataloader)

    def _next_flow_batch(self):
        try:
            return next(self._flow_iter)
        except StopIteration:
            self._flow_iter = iter(self.dataloader)
            return next(self._flow_iter)

    def _next_track_batch(self):
        try:
            return next(self._track_iter)
        except StopIteration:
            self._track_iter = iter(self.track_dataloader)
            return next(self._track_iter)

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
        self._reset_stream_state()

        current_epoch = self._epoch + 1
        run_track = current_epoch >= self.track_start_epoch

        for idx in range(len(self.dataloader)):
            flow_batch = self._next_flow_batch()
            track_batch = self._next_track_batch() if run_track else None
            self.run_iter(idx, flow_batch, track_batch, current_epoch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, flow_batch, track_batch, current_epoch) -> None:
        loop_batch = dict(
            flow_batch=flow_batch,
            track_batch=track_batch,
            current_epoch=current_epoch,
        )

        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=loop_batch)

        flow_outputs = self.runner.model.train_step_flow(
            flow_batch, optim_wrapper=self.runner.optim_wrapper)

        track_outputs = None
        if track_batch is not None:
            track_outputs = self.runner.model.train_step_track(
                track_batch, optim_wrapper=self.runner.optim_wrapper)

        outputs = self._merge_stage_outputs(flow_outputs, track_outputs)

        # Helpful counters for logs.
        if isinstance(outputs, dict):
            ref_tensor = None
            for _, v in outputs.items():
                if torch.is_tensor(v):
                    ref_tensor = v
                    break
            if ref_tensor is None:
                ref_tensor = torch.tensor(0.0)
            outputs['flow_batch_size'] = ref_tensor.new_tensor(float(len(self._as_batch_list(flow_batch))))
            outputs['track_batch_size'] = ref_tensor.new_tensor(float(len(self._as_batch_list(track_batch))) if track_batch is not None else 0.0)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=loop_batch,
            outputs=outputs,
        )
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
