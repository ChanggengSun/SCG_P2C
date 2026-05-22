from pathlib import Path

import torch

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class BranchEpochCheckpointHook(Hook):
    """Save per-epoch branch-specific checkpoints for flow and tracking.

    This does not alter training/optimization logic. It only exports filtered
    state_dict snapshots so each task has its own epoch-wise checkpoint file.
    """

    priority = 'LOW'

    def __init__(
        self,
        interval=1,
        flow_dir='epoch_flow',
        track_dir='epoch_tracking',
        save_flow=True,
        save_track=True,
        include_shared=True,
    ):
        self.interval = max(1, int(interval))
        self.flow_dir = str(flow_dir)
        self.track_dir = str(track_dir)
        self.save_flow = bool(save_flow)
        self.save_track = bool(save_track)
        self.include_shared = bool(include_shared)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    @staticmethod
    def _filter_state_dict(state_dict, prefixes):
        out = {}
        for k, v in state_dict.items():
            for p in prefixes:
                if k.startswith(p):
                    out[k] = v.detach().cpu() if torch.is_tensor(v) else v
                    break
        return out

    def _save_one(self, runner, out_dir_name, file_name, state_dict):
        out_dir = Path(runner.work_dir) / out_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        payload = dict(
            state_dict=state_dict,
            meta=dict(
                epoch=int(getattr(runner, 'epoch', 0)) + 1,
                iter=int(getattr(runner, 'iter', 0)),
                branch_checkpoint=True,
            ),
        )
        torch.save(payload, str(out_path))
        return out_path

    def after_train_epoch(self, runner):
        epoch_display = int(getattr(runner, 'epoch', 0)) + 1
        if epoch_display % self.interval != 0:
            return

        model = self._unwrap_model(runner.model)
        full_sd = model.state_dict()

        shared = ('backbone.',) if self.include_shared else tuple()

        if self.save_flow:
            flow_prefixes = shared + ('flow_fuse.', 'flow_head.')
            flow_sd = self._filter_state_dict(full_sd, flow_prefixes)
            if len(flow_sd) > 0:
                p = self._save_one(
                    runner,
                    self.flow_dir,
                    f'flow_epoch_{epoch_display}.pth',
                    flow_sd,
                )
                print_log(f'[BranchCkpt] Saved flow epoch checkpoint: {p}', logger='current')

        if self.save_track:
            # ``fuse`` is kept as the tracking fuser key for legacy P2P
            # compatibility.
            track_prefixes = shared + ('fuse.', 'tracking_head.')
            track_sd = self._filter_state_dict(full_sd, track_prefixes)
            if len(track_sd) > 0:
                p = self._save_one(
                    runner,
                    self.track_dir,
                    f'track_epoch_{epoch_display}.pth',
                    track_sd,
                )
                print_log(f'[BranchCkpt] Saved tracking epoch checkpoint: {p}', logger='current')
