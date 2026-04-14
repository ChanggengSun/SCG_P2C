import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.metrics import FlowOfficialMetric
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import DATASETS, HOOKS


def _simple_collate(batch):
    return batch


@HOOKS.register_module()
class JointFlowEvalHook(Hook):
    """Run flow validation during joint training and save flow-best checkpoint."""

    priority = 'LOW'

    def __init__(
        self,
        flow_data_dir='C:/develop/OpenSceneFlow/data/processed',
        flow_repo_root='C:/develop/OpenSceneFlow',
        split='val',
        interval=1,
        start=1,
        batch_size=1,
        num_workers=0,
        remove_ground=False,
        input_dim=4,
        history_frames=3,
        metric_key='val/Dynamic/Mean',
        log_ssf=False,
        run_on_train_epoch=True,
    ):
        self.flow_data_dir = str(flow_data_dir)
        self.flow_repo_root = str(flow_repo_root)
        self.split = str(split)
        self.interval = max(1, int(interval))
        self.start = max(1, int(start))
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
        self.remove_ground = bool(remove_ground)
        self.input_dim = int(input_dim)
        self.history_frames = max(0, int(history_frames))
        self.metric_key = str(metric_key)
        self.log_ssf = bool(log_ssf)
        self.run_on_train_epoch = bool(run_on_train_epoch)

        self._val_loader = None
        self.best_score = None
        self.best_ckpt_path = None

    def before_train(self, runner):
        if self._val_loader is not None:
            return

        dataset_cfg = dict(
            type='NuScenesFlowSeq5NativeDataset',
            path=self.flow_data_dir,
            split=self.split,
            remove_ground=self.remove_ground,
            input_dim=self.input_dim,
            history_frames=self.history_frames,
        )
        dataset = DATASETS.build(dataset_cfg)
        self._val_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=_simple_collate,
        )

    @staticmethod
    def _ensure_sample_dict(model, data):
        if hasattr(model, '_ensure_sample_dict'):
            return model._ensure_sample_dict(data)
        if isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], dict):
                return data[0]
            raise RuntimeError(f'Unsupported batch list format for flow eval: len={len(data)}')
        if not isinstance(data, dict):
            raise TypeError(f'Invalid flow-eval data type: {type(data)}')
        return data

    @torch.no_grad()
    def _run_flow_eval(self, runner):
        self.before_train(runner)

        model = runner.model
        was_training = model.training
        model.eval()

        metric = FlowOfficialMetric(flow_repo_root=self.flow_repo_root, log_ssf=self.log_ssf)
        for data_batch in self._val_loader:
            data = model.data_preprocessor(data_batch, False)
            data = self._ensure_sample_dict(model, data)
            outputs = model(**data, mode='flow_predict')
            metric.process(data_batch, outputs)

        metrics = metric.compute_metrics(metric.results)
        if was_training:
            model.train()
        return metrics

    @staticmethod
    def _is_finite_number(x):
        return isinstance(x, (int, float)) and math.isfinite(float(x))

    def _should_run(self, epoch_display):
        if int(epoch_display) < self.start:
            return False
        return int(epoch_display) % self.interval == 0

    def _save_best_flow_ckpt(self, runner, current_score, epoch_display):
        if not math.isfinite(current_score):
            return
        if self.best_score is not None and current_score >= self.best_score:
            return

        self.best_score = float(current_score)
        flow_dir = Path(runner.work_dir) / 'best_flow'
        flow_dir.mkdir(parents=True, exist_ok=True)

        filename = f'best_flow_dynamic_mean_epoch_{int(epoch_display)}.pth'
        runner.save_checkpoint(
            out_dir=str(flow_dir),
            filename=filename,
            save_optimizer=False,
            save_param_scheduler=False,
            meta=dict(
                epoch=int(epoch_display),
                best_flow_dynamic_mean=self.best_score,
                metric_key=self.metric_key,
            ),
            by_epoch=True,
        )
        self.best_ckpt_path = flow_dir / filename
        print_log(
            f'[JointFlowEval] New best flow checkpoint: {self.best_ckpt_path} '
            f'({self.metric_key}={self.best_score:.4f})',
            logger='current',
        )

    def _evaluate_and_record(self, runner, epoch_display):
        if not self._should_run(epoch_display):
            return

        flow_metrics = self._run_flow_eval(runner)
        metric_items = []
        for key in sorted(flow_metrics.keys()):
            val = flow_metrics[key]
            if self._is_finite_number(val):
                metric_items.append(f'{key}: {float(val):.4f}')
            else:
                metric_items.append(f'{key}: {val}')

        msg = f'Epoch(flow-val) [{int(epoch_display)}]    ' + '  '.join(metric_items)
        print_log(msg, logger='current')

        runner.message_hub.update_info('joint_flow_metrics', flow_metrics)
        current_score = float(flow_metrics.get(self.metric_key, float('inf')))
        self._save_best_flow_ckpt(runner, current_score, epoch_display)

    def after_train_epoch(self, runner):
        if not self.run_on_train_epoch:
            return
        epoch_display = int(getattr(runner, 'epoch', 0)) + 1
        self._evaluate_and_record(runner, epoch_display)

    def after_val_epoch(self, runner, metrics=None):
        if self.run_on_train_epoch:
            return
        epoch_display = int(getattr(runner, 'epoch', 0))
        self._evaluate_and_record(runner, epoch_display)

    def after_train(self, runner):
        work_dir = Path(runner.work_dir)
        best_track = sorted(
            work_dir.glob('best_precision*.pth'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(best_track) > 0:
            out_dir = work_dir / 'best_tracking'
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(best_track[0]), str(out_dir / 'track_best.pth'))

        if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
            flow_alias = self.best_ckpt_path.parent / 'flow_best.pth'
            shutil.copy2(str(self.best_ckpt_path), str(flow_alias))
