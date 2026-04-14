import bisect
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals


@LOOPS.register_module()
class JointSeq5TrainLoop(BaseLoop):
    """Epoch loop for single-loader 5-frame synchronized joint training."""

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(runner, dataloader)

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, f'`max_epochs` should be int, got {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False

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

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
