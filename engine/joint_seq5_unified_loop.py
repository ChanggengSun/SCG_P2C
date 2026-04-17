"""Unified single-dataloader training loop for joint tracking + flow.

Eliminates the dual-dataloader pattern where tracking and flow dataloaders
independently load overlapping point clouds from the same scenes.  Instead,
a single ``NuScenesJointSeq5Dataset`` (with ``skip_flow_loading=False`` and
``track_source='h5'``) produces **both** flow and tracking tensors per sample.

The loop dispatches the same batch to ``model.train_step_flow`` and
``model.train_step_track`` in sequence, achieving identical training semantics
as ``JointSeq5IndependentDualTrainLoop`` while halving disk I/O and eliminating
redundant NuScenes SDK overhead.

Drop-in replacement: only the config needs to change —
    ``type='JointSeq5UnifiedTrainLoop'`` with a single ``dataloader`` entry.
"""

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
class JointSeq5UnifiedTrainLoop(BaseLoop):
    """Single-dataloader loop for joint tracking + flow training.

    The dataset must produce both flow fields (``pc0``, ``pc1``, ``query_points``,
    ``pose_flow``, ``gt_flow``, …) **and** tracking fields (``track_prev_points``,
    ``track_this_points``, ``track_wlh``, ``track_box_label``, ``track_theta``).

    ``NuScenesJointSeq5Dataset`` with ``skip_flow_loading=False`` satisfies this.

    Parameters
    ----------
    runner : Runner
        The mmengine runner instance.
    dataloader : DataLoader | dict
        Unified dataloader producing both flow and tracking data.
    max_epochs : int
        Total number of training epochs.
    track_start_epoch : int
        Epoch number (1-indexed) from which tracking loss is added.
        Before this epoch, only flow loss is computed.
    val_begin : int
        Epoch number (1-indexed) from which validation starts.
    val_interval : int
        Validation frequency in epochs.
    dynamic_intervals : list of (milestone, interval) tuples, optional
        Dynamic validation interval schedule.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        track_start_epoch: int = 11,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(runner, dataloader)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be int, got {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0

        self.track_start_epoch = max(1, int(track_start_epoch))
        self.val_begin = max(1, int(val_begin))
        self.val_interval = max(1, int(val_interval))
        self.stop_training = False

        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} '
                'has no metainfo. ``dataset_meta`` in visualizer will be None.',
                logger='current',
                level=logging.WARNING,
            )

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(self.val_interval, dynamic_intervals)

    # ---- properties --------------------------------------------------------

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

    # ---- helpers -----------------------------------------------------------

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

    # ---- main loop ---------------------------------------------------------

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

        current_epoch = self._epoch + 1
        run_track = current_epoch >= self.track_start_epoch

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, run_track, current_epoch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch, run_track, current_epoch) -> None:
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)

        # Stage 1: flow
        flow_outputs = self.runner.model.train_step_flow(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        # Stage 2: tracking (same batch, no extra I/O)
        track_outputs = None
        if run_track:
            track_outputs = self.runner.model.train_step_track(
                data_batch, optim_wrapper=self.runner.optim_wrapper)

        outputs = self._merge_stage_outputs(flow_outputs, track_outputs)

        # Helpful counters for logs
        if isinstance(outputs, dict):
            ref_tensor = None
            for _, v in outputs.items():
                if torch.is_tensor(v):
                    ref_tensor = v
                    break
            if ref_tensor is None:
                ref_tensor = torch.tensor(0.0)
            bs = float(len(self._as_batch_list(data_batch)))
            outputs['batch_size'] = ref_tensor.new_tensor(bs)
            outputs['flow_batch_size'] = ref_tensor.new_tensor(bs)
            outputs['track_batch_size'] = ref_tensor.new_tensor(
                bs if run_track else 0.0)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(
            self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
