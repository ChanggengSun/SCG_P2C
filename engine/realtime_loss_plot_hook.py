from collections import defaultdict
from pathlib import Path
import math

import torch
from mmengine.hooks import Hook


class RealtimeLossPlotHook(Hook):
    """Realtime loss plotting synchronized with terminal logger interval.

    - Update once per logging interval (e.g. every 50 iters when log prints [50/...]).
    - Keep all epochs' points on one continuous curve.
    - Save:
      * live snapshot
      * per-epoch snapshot (non-overwrite)
      * final all-epochs snapshot
    """

    priority = 'LOW'

    def __init__(
        self,
        iter_interval=50,
        task_name='task',
        enable_window=True,
        save_image=True,
        include_keys=None,
        exclude_keys=None,
        max_points=200000,
        use_log_processor=True,
    ):
        self.iter_interval = max(1, int(iter_interval))
        self.task_name = str(task_name)
        self.enable_window = bool(enable_window)
        self.save_image = bool(save_image)
        self.include_keys = [
            str(k).lower() for k in (include_keys or [])
            if str(k).strip() != ''
        ]
        self.exclude_keys = [
            str(k).lower() for k in (exclude_keys or [])
            if str(k).strip() != ''
        ]
        self.max_points = max(1000, int(max_points))
        self.use_log_processor = bool(use_log_processor)

        self._series = defaultdict(list)
        self._warned_no_plot = False

        self._plt = None
        self._fig = None
        self._ax = None
        self._plot_dir = None
        self._live_path = None

    @staticmethod
    def _to_float(val):
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return None
            val = val.detach().float().mean().item()
        elif hasattr(val, 'item'):
            try:
                val = float(val.item())
            except Exception:
                return None
        else:
            try:
                val = float(val)
            except Exception:
                return None
        if not math.isfinite(val):
            return None
        return val

    def _setup_plot(self, runner):
        if self._plt is not None:
            return True
        try:
            import matplotlib.pyplot as plt

            if self.enable_window:
                plt.ion()
            self._plt = plt
            self._fig, self._ax = plt.subplots(figsize=(11, 5), num=f'{self.task_name}_loss')
            self._fig.suptitle(f'{self.task_name} loss monitor')
            if self.enable_window:
                plt.show(block=False)

            if self.save_image:
                self._plot_dir = Path(runner.work_dir) / 'loss_plots' / self.task_name
                self._plot_dir.mkdir(parents=True, exist_ok=True)
                self._live_path = self._plot_dir / f'{self.task_name}_loss_live.png'
            return True
        except Exception as exc:
            if not self._warned_no_plot:
                print(f'[LossPlotHook] Disable plotting ({self.task_name}): {exc}')
                self._warned_no_plot = True
            return False

    def _collect_loss_values(self, global_iter, outputs):
        if not isinstance(outputs, dict):
            return
        for key, val in outputs.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if 'loss' not in key_lower:
                continue
            if self.include_keys and not any(token in key_lower for token in self.include_keys):
                continue
            if self.exclude_keys and any(token in key_lower for token in self.exclude_keys):
                continue
            scalar = self._to_float(val)
            if scalar is None:
                continue
            self._series[key_str].append((global_iter, scalar))
            if len(self._series[key_str]) > self.max_points:
                self._series[key_str] = self._series[key_str][-self.max_points:]

    def _collect_from_log_tag(self, global_iter, log_tag, num_digits=4):
        if not isinstance(log_tag, dict):
            return
        for key, val in log_tag.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if 'loss' not in key_lower:
                continue
            if self.include_keys and not any(token in key_lower for token in self.include_keys):
                continue
            if self.exclude_keys and any(token in key_lower for token in self.exclude_keys):
                continue
            scalar = self._to_float(val)
            if scalar is None:
                continue
            # Keep plot values aligned with terminal displayed precision.
            scalar = round(scalar, int(num_digits))
            self._series[key_str].append((global_iter, scalar))
            if len(self._series[key_str]) > self.max_points:
                self._series[key_str] = self._series[key_str][-self.max_points:]

    def _draw(self, runner, extra_title=''):
        if not self._setup_plot(runner):
            return
        if len(self._series) == 0:
            return

        self._ax.clear()
        for key in sorted(self._series.keys()):
            pairs = self._series[key]
            if len(pairs) == 0:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            self._ax.plot(xs, ys, label=key, linewidth=1.2)

        self._ax.set_xlabel('Global Iteration')
        self._ax.set_ylabel('Loss')
        title = f'{self.task_name} loss (update every logger interval={self.iter_interval} iters)'
        if extra_title:
            title = f'{title} | {extra_title}'
        self._ax.set_title(title)
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc='best')

        if self.enable_window:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(0.001)

        if self.save_image and self._live_path is not None:
            self._fig.savefig(self._live_path, dpi=140, bbox_inches='tight')

    def _save_epoch_snapshot(self, runner):
        if not self.save_image or self._plot_dir is None or self._fig is None:
            return
        epoch_idx = int(getattr(runner, 'epoch', 0)) + 1
        p = self._plot_dir / f'{self.task_name}_loss_epoch_{epoch_idx:03d}.png'
        self._fig.savefig(p, dpi=160, bbox_inches='tight')

    def _save_final_snapshot(self, runner):
        if not self.save_image or self._plot_dir is None or self._fig is None:
            return
        p = self._plot_dir / f'{self.task_name}_loss_all_epochs.png'
        self._fig.savefig(p, dpi=180, bbox_inches='tight')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        global_iter = int(getattr(runner, 'iter', 0)) + 1
        if global_iter % self.iter_interval != 0:
            return
        used_log_processor = False
        if self.use_log_processor and hasattr(runner, 'log_processor') and runner.log_processor is not None:
            try:
                tag, _ = runner.log_processor.get_log_after_iter(runner, batch_idx, mode='train')
                num_digits = int(getattr(runner.log_processor, 'num_digits', 4))
                self._collect_from_log_tag(global_iter, tag, num_digits=num_digits)
                used_log_processor = True
            except Exception:
                used_log_processor = False
        if not used_log_processor:
            self._collect_loss_values(global_iter, outputs)
        self._draw(runner, extra_title=f'iter={global_iter}')

    def after_train_epoch(self, runner):
        self._draw(runner, extra_title=f'epoch={int(getattr(runner, "epoch", 0)) + 1}')
        self._save_epoch_snapshot(runner)

    def after_train(self, runner):
        self._draw(runner, extra_title='final')
        self._save_final_snapshot(runner)
