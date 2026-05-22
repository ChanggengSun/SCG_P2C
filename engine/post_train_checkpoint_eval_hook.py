import json
import math
import re
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


def _strip_module_prefix(state_dict):
    if len(state_dict) == 0:
        return state_dict
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def _legacy_fuse_key_for(model_key):
    """Return legacy shared-fuser key for branch-specific fuser slots."""
    if model_key.startswith('flow_fuse.'):
        return 'fuse.' + model_key[len('flow_fuse.'):]
    return None


def _adapt_tensor_shape(src_tensor, dst_tensor):
    """Try adapting source tensor layout to destination layout.

    Main use case is sparse-conv kernel layout mismatch across spconv versions.
    """
    if src_tensor.shape == dst_tensor.shape:
        return src_tensor, False

    if src_tensor.ndim == 5 and dst_tensor.ndim == 5:
        cand_list = [
            src_tensor.permute(1, 2, 3, 4, 0).contiguous(),  # [in,out,k,k,k] -> [out,k,k,k,in]
            src_tensor.permute(0, 2, 3, 4, 1).contiguous(),  # [out,in,k,k,k] -> [out,k,k,k,in]
            src_tensor.permute(4, 1, 2, 3, 0).contiguous(),
            src_tensor.permute(4, 0, 1, 2, 3).contiguous(),
            src_tensor.permute(1, 0, 2, 3, 4).contiguous(),
        ]
        for cand in cand_list:
            if cand.shape == dst_tensor.shape:
                return cand, True

    return None, False


@HOOKS.register_module()
class PostTrainCheckpointEvalHook(Hook):
    """Evaluate saved checkpoints after training ends.

    Supported modes:
    - per_checkpoint: evaluate flow + tracking for each checkpoint in one pass.
    - track_then_flow: evaluate tracking over all ckpts first, then flow over all ckpts.
    - flow_then_track: evaluate flow first, then tracking.
    """

    priority = 'LOW'

    def __init__(
        self,
        flow_data_dir='C:/develop/OpenSceneFlow/data/processed',
        flow_repo_root='C:/develop/OpenSceneFlow',
        flow_split='val',
        flow_batch_size=1,
        flow_num_workers=0,
        remove_ground=False,
        input_dim=4,
        history_frames=3,
        enable_flow_eval=True,
        enable_track_eval=True,
        flow_metric_key='val/Dynamic/Mean',
        track_metric_key='precision',
        checkpoint_pattern='epoch_*.pth',
        result_filename='post_train_all_ckpt_eval.json',
        log_ssf=False,
        eval_reverse=False,
        eval_epoch_start=None,
        eval_epoch_end=None,
        task_eval_order='per_checkpoint',
        fixed_flow_checkpoint=None,
    ):
        self.flow_data_dir = str(flow_data_dir)
        self.flow_repo_root = str(flow_repo_root)
        self.flow_split = str(flow_split)
        self.flow_batch_size = max(1, int(flow_batch_size))
        self.flow_num_workers = max(0, int(flow_num_workers))
        self.remove_ground = bool(remove_ground)
        self.input_dim = int(input_dim)
        self.history_frames = max(0, int(history_frames))

        self.enable_flow_eval = bool(enable_flow_eval)
        self.enable_track_eval = bool(enable_track_eval)
        self.flow_metric_key = str(flow_metric_key)
        self.track_metric_key = str(track_metric_key)
        self.checkpoint_pattern = str(checkpoint_pattern)
        self.result_filename = str(result_filename)
        self.log_ssf = bool(log_ssf)
        self.eval_reverse = bool(eval_reverse)
        self.eval_epoch_start = None if eval_epoch_start is None else int(eval_epoch_start)
        self.eval_epoch_end = None if eval_epoch_end is None else int(eval_epoch_end)
        self.fixed_flow_checkpoint = (
            None if fixed_flow_checkpoint in (None, '') else Path(fixed_flow_checkpoint)
        )

        order = str(task_eval_order).strip().lower()
        if order not in ('per_checkpoint', 'track_then_flow', 'flow_then_track'):
            order = 'per_checkpoint'
        self.task_eval_order = order

        self._flow_val_loader = None

    @staticmethod
    def _is_finite_number(x):
        return isinstance(x, (int, float)) and math.isfinite(float(x))

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

    def before_train(self, runner):
        if not self.enable_flow_eval:
            return
        if self._flow_val_loader is not None:
            return

        dataset_cfg = dict(
            type='NuScenesFlowSeq5NativeDataset',
            path=self.flow_data_dir,
            split=self.flow_split,
            remove_ground=self.remove_ground,
            input_dim=self.input_dim,
            history_frames=self.history_frames,
        )
        dataset = DATASETS.build(dataset_cfg)
        self._flow_val_loader = DataLoader(
            dataset=dataset,
            batch_size=self.flow_batch_size,
            shuffle=False,
            num_workers=self.flow_num_workers,
            pin_memory=True,
            persistent_workers=(self.flow_num_workers > 0),
            collate_fn=_simple_collate,
        )

    @torch.no_grad()
    def _run_flow_eval(self, runner):
        if not self.enable_flow_eval:
            return {}

        self.before_train(runner)
        model = runner.model
        was_training = model.training
        model.eval()

        metric = FlowOfficialMetric(flow_repo_root=self.flow_repo_root, log_ssf=self.log_ssf)
        for data_batch in self._flow_val_loader:
            data = model.data_preprocessor(data_batch, False)
            data = self._ensure_sample_dict(model, data)
            outputs = model(**data, mode='flow_predict')
            metric.process(data_batch, outputs)

        metrics = metric.compute_metrics(metric.results)
        if was_training:
            model.train()
        return metrics

    @staticmethod
    def _collect_epoch_checkpoints(work_dir: Path, pattern: str):
        paths = list(work_dir.glob(pattern))
        parsed = []
        for p in paths:
            m = re.search(r'epoch_(\d+)\.pth$', p.name)
            if m is None:
                continue
            parsed.append((int(m.group(1)), p))
        parsed.sort(key=lambda x: x[0])
        return parsed

    def _filter_and_order_checkpoints(self, ckpts):
        if len(ckpts) == 0:
            return ckpts

        if self.eval_epoch_start is not None and self.eval_epoch_end is not None:
            lo = min(self.eval_epoch_start, self.eval_epoch_end)
            hi = max(self.eval_epoch_start, self.eval_epoch_end)
            ckpts = [(ep, p) for ep, p in ckpts if lo <= ep <= hi]
        elif self.eval_epoch_start is not None:
            ckpts = [(ep, p) for ep, p in ckpts if ep >= self.eval_epoch_start]
        elif self.eval_epoch_end is not None:
            ckpts = [(ep, p) for ep, p in ckpts if ep <= self.eval_epoch_end]

        if self.eval_reverse:
            ckpts = sorted(ckpts, key=lambda x: x[0], reverse=True)
        else:
            ckpts = sorted(ckpts, key=lambda x: x[0])
        return ckpts

    @staticmethod
    def _branch_epoch_path(work_dir: Path, branch_dir: str, prefix: str, epoch_id: int):
        p = work_dir / branch_dir / f'{prefix}_epoch_{int(epoch_id)}.pth'
        if p.exists():
            return p
        return None

    @staticmethod
    def _to_float(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().item()
        return float(x)

    @staticmethod
    def _set_runner_epoch_for_logging(runner, epoch_id: int):
        try:
            if getattr(runner, 'train_loop', None) is not None and hasattr(runner.train_loop, '_epoch'):
                runner.train_loop._epoch = int(epoch_id)
            if hasattr(runner, 'message_hub') and runner.message_hub is not None:
                runner.message_hub.update_info('epoch', int(epoch_id))
        except Exception:
            pass

    def _load_checkpoint_to_model(self, runner, ckpt_path: Path):
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        state_dict = _strip_module_prefix(ckpt.get('state_dict', ckpt))
        if isinstance(state_dict, (list, tuple)):
            # Defensive compatibility: some legacy saves may serialize as
            # iterable of (key, tensor) pairs.
            try:
                state_dict = dict(state_dict)
            except Exception:
                state_dict = {}

        # Strong trace log: print the real absolute checkpoint path and key
        # shapes before any adaptation, to pinpoint mismatch source quickly.
        abs_ckpt_path = str(ckpt_path.resolve())
        print_log(
            f'[PostEval][CKPT-TRACE] loading_ckpt={abs_ckpt_path}',
            logger='current',
        )

        model_state = runner.model.state_dict()

        def _shape_of(obj):
            if isinstance(obj, torch.Tensor):
                return tuple(obj.shape)
            if obj is None:
                return 'NOT_FOUND'
            return f'NON_TENSOR({type(obj).__name__})'

        def _unwrap_src_obj(obj):
            if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[1], torch.Tensor):
                return obj[1]
            return obj

        probe_keys = [
            'backbone.middle_encoder.conv_input.0.weight',
            'backbone.middle_encoder.conv_input.3.weight',
        ]
        for key in probe_keys:
            src_obj = _unwrap_src_obj(state_dict.get(key, None))
            dst_obj = model_state.get(key, None)
            print_log(
                f'[PostEval][CKPT-TRACE] key={key} | '
                f'ckpt_shape={_shape_of(src_obj)} | model_shape={_shape_of(dst_obj)}',
                logger='current',
            )

        mismatch_examples = []
        for k, dst in model_state.items():
            src_key = k
            if src_key not in state_dict:
                legacy_key = _legacy_fuse_key_for(k)
                if legacy_key is not None and legacy_key in state_dict:
                    src_key = legacy_key
                else:
                    continue
            src_obj = _unwrap_src_obj(state_dict[src_key])
            if not isinstance(src_obj, torch.Tensor):
                continue
            if not isinstance(dst, torch.Tensor):
                continue
            if src_obj.shape != dst.shape:
                mismatch_examples.append((k, tuple(src_obj.shape), tuple(dst.shape)))
                if len(mismatch_examples) >= 8:
                    break
        if mismatch_examples:
            pretty = '; '.join([f'{k}: ckpt{ss} vs model{ds}' for k, ss, ds in mismatch_examples])
            print_log(
                f'[PostEval][CKPT-TRACE] shape_mismatch_examples={pretty}',
                logger='current',
            )

        patched_state = {}
        converted = 0
        skipped_shape = 0
        loaded = 0

        for k, dst in model_state.items():
            src_key = k
            if src_key not in state_dict:
                legacy_key = _legacy_fuse_key_for(k)
                if legacy_key is not None and legacy_key in state_dict:
                    src_key = legacy_key
                else:
                    continue

            src_obj = state_dict[src_key]
            # Defensive compatibility: legacy values may be wrapped as
            # (name, tensor) or [name, tensor].
            src_obj = _unwrap_src_obj(src_obj)

            if not isinstance(src_obj, torch.Tensor):
                # Skip non-tensor entries for model tensor slots.
                skipped_shape += 1
                continue

            src = src_obj.detach().cpu()
            did_convert = False
            if src.shape != dst.shape:
                adapted, did_convert = _adapt_tensor_shape(src, dst)
                if adapted is None:
                    skipped_shape += 1
                    continue
            else:
                adapted = src

            # Final guard: never pass mismatched shapes into load_state_dict.
            if adapted.shape != dst.shape:
                skipped_shape += 1
                continue

            patched_state[k] = adapted.to(dtype=dst.dtype)
            loaded += 1
            if did_convert:
                converted += 1

        # Final sanitize before loading to avoid any malformed value slipping in.
        sanitized_state = {}
        for k, v in patched_state.items():
            dst = model_state.get(k, None)
            if dst is None:
                continue
            if not isinstance(v, torch.Tensor):
                skipped_shape += 1
                continue
            if v.shape != dst.shape:
                skipped_shape += 1
                continue
            sanitized_state[k] = v

        # Use direct copy instead of load_state_dict to avoid hard failure on
        # legacy sparse-conv kernel-layout mismatches during post-eval.
        copied = 0
        with torch.no_grad():
            for k, src in sanitized_state.items():
                dst = model_state.get(k, None)
                if dst is None:
                    continue
                try:
                    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
                    copied += 1
                except Exception:
                    skipped_shape += 1

        missing = max(0, len(model_state) - copied)
        print_log(
            f'[PostEval] Loaded {ckpt_path.name} with '
            f'loaded={copied}, missing={missing}, unexpected=0, '
            f'converted={converted}, skipped_shape={skipped_shape}',
            logger='current',
        )

    @staticmethod
    def _normalize_metrics(metrics: dict):
        out = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k] = float(v)
            elif torch.is_tensor(v):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = v
        return out

    def _eval_track_for_ckpt(self, runner, work_dir: Path, epoch_id: int, base_ckpt_path: Path):
        track_ckpt = self._branch_epoch_path(work_dir, 'epoch_tracking', 'track', epoch_id) or base_ckpt_path
        self._load_checkpoint_to_model(runner, track_ckpt)
        track_metrics = {}
        if getattr(runner, 'val_loop', None) is not None:
            self._set_runner_epoch_for_logging(runner, epoch_id)
            maybe_metrics = runner.val_loop.run()
            if isinstance(maybe_metrics, dict):
                track_metrics = maybe_metrics
        return self._normalize_metrics(track_metrics), track_ckpt

    def _eval_flow_for_ckpt(self, runner, work_dir: Path, epoch_id: int, base_ckpt_path: Path):
        if not self.enable_flow_eval:
            return {}, base_ckpt_path
        flow_ckpt = self.fixed_flow_checkpoint
        if flow_ckpt is None:
            flow_ckpt = self._branch_epoch_path(work_dir, 'epoch_flow', 'flow', epoch_id) or base_ckpt_path
        if not flow_ckpt.exists():
            raise FileNotFoundError(f'Fixed flow checkpoint not found: {flow_ckpt}')
        self._load_checkpoint_to_model(runner, flow_ckpt)
        flow_metrics = self._run_flow_eval(runner)
        return self._normalize_metrics(flow_metrics), flow_ckpt

    def _update_best(self, best, score, epoch_id, ckpt_path, greater_is_better=False):
        if not self._is_finite_number(score):
            return best
        score = float(score)
        if best is None:
            return (score, int(epoch_id), ckpt_path)
        if greater_is_better and score > best[0]:
            return (score, int(epoch_id), ckpt_path)
        if (not greater_is_better) and score < best[0]:
            return (score, int(epoch_id), ckpt_path)
        return best

    def after_train(self, runner):
        work_dir = Path(runner.work_dir)
        ckpts = self._collect_epoch_checkpoints(work_dir, self.checkpoint_pattern)
        ckpts = self._filter_and_order_checkpoints(ckpts)
        if len(ckpts) == 0:
            print_log('[PostEval] No epoch checkpoints found, skip post-training evaluation.', logger='current')
            return

        order = 'desc' if self.eval_reverse else 'asc'
        print_log(
            f'[PostEval] Start evaluating {len(ckpts)} checkpoints after training... '
            f'(order={order}, start={self.eval_epoch_start}, end={self.eval_epoch_end}, mode={self.task_eval_order})',
            logger='current')

        rows = {
            int(ep): dict(epoch=int(ep), checkpoint=str(path), flow_metrics={}, track_metrics={})
            for ep, path in ckpts
        }

        best_flow = None
        best_track = None
        fixed_flow_metrics = None
        fixed_flow_src = None

        if self.enable_flow_eval and self.fixed_flow_checkpoint is not None:
            fixed_flow_src = self.fixed_flow_checkpoint
            if not fixed_flow_src.exists():
                raise FileNotFoundError(f'Fixed flow checkpoint not found: {fixed_flow_src}')
            print_log(
                f'[PostEval-FlowFixed] Evaluate fixed flow checkpoint only: {fixed_flow_src}',
                logger='current',
            )
            self._load_checkpoint_to_model(runner, fixed_flow_src)
            fixed_flow_metrics = self._normalize_metrics(self._run_flow_eval(runner))
            flow_score = fixed_flow_metrics.get(self.flow_metric_key, None)
            best_flow = self._update_best(best_flow, flow_score, 0, fixed_flow_src, greater_is_better=False)
            if self._is_finite_number(flow_score):
                flow_msg = f'{self.flow_metric_key}={float(flow_score):.4f}'
            else:
                flow_msg = f'{self.flow_metric_key}=N/A'
            print_log(
                f'[PostEval-FlowFixed] {flow_msg} | ckpt={Path(fixed_flow_src).name}',
                logger='current',
            )

        if self.task_eval_order == 'per_checkpoint':
            for epoch_id, ckpt_path in ckpts:
                if fixed_flow_metrics is not None:
                    flow_metrics, flow_src = fixed_flow_metrics, fixed_flow_src
                else:
                    flow_metrics, flow_src = self._eval_flow_for_ckpt(runner, work_dir, epoch_id, ckpt_path)
                if self.enable_track_eval:
                    track_metrics, track_src = self._eval_track_for_ckpt(runner, work_dir, epoch_id, ckpt_path)
                else:
                    track_metrics, track_src = {}, None

                rows[epoch_id]['flow_metrics'] = flow_metrics
                rows[epoch_id]['track_metrics'] = track_metrics

                flow_score = flow_metrics.get(self.flow_metric_key, None)
                track_score = track_metrics.get(self.track_metric_key, None)

                if fixed_flow_metrics is None:
                    best_flow = self._update_best(best_flow, flow_score, epoch_id, flow_src, greater_is_better=False)
                if self.enable_track_eval:
                    best_track = self._update_best(best_track, track_score, epoch_id, track_src, greater_is_better=True)

                flow_msg = f'{self.flow_metric_key}=N/A'
                if self.enable_flow_eval and self._is_finite_number(flow_score):
                    flow_msg = f'{self.flow_metric_key}={float(flow_score):.4f}'
                elif not self.enable_flow_eval:
                    flow_msg = 'flow=disabled'

                track_msg = f'{self.track_metric_key}=N/A'
                if not self.enable_track_eval:
                    track_msg = 'track=disabled'
                elif self._is_finite_number(track_score):
                    track_msg = f'{self.track_metric_key}={float(track_score):.4f}'

                print_log(f'[PostEval] epoch={epoch_id} | {flow_msg} | {track_msg} | ckpt={ckpt_path.name}', logger='current')

        else:
            if self.task_eval_order == 'track_then_flow':
                phase_first = 'track'
                phase_second = 'flow'
            else:
                phase_first = 'flow'
                phase_second = 'track'

            for phase in (phase_first, phase_second):
                if phase == 'flow' and not self.enable_flow_eval:
                    print_log('[PostEval] Skip flow phase because flow eval is disabled.', logger='current')
                    continue
                if phase == 'track' and not self.enable_track_eval:
                    print_log('[PostEval] Skip track phase because track eval is disabled.', logger='current')
                    continue
                if phase == 'flow' and fixed_flow_metrics is not None:
                    for epoch_id, _ in ckpts:
                        rows[epoch_id]['flow_metrics'] = fixed_flow_metrics
                    flow_score = fixed_flow_metrics.get(self.flow_metric_key, None)
                    if self._is_finite_number(flow_score):
                        flow_msg = f'{self.flow_metric_key}={float(flow_score):.4f}'
                    else:
                        flow_msg = f'{self.flow_metric_key}=N/A'
                    print_log(
                        f'[PostEval-Flow] fixed checkpoint already evaluated once | '
                        f'{flow_msg} | ckpt={Path(fixed_flow_src).name}',
                        logger='current',
                    )
                    continue

                print_log(f'[PostEval] Phase {phase}: start {len(ckpts)} checkpoints.', logger='current')
                for epoch_id, ckpt_path in ckpts:
                    if phase == 'track':
                        track_metrics, track_src = self._eval_track_for_ckpt(runner, work_dir, epoch_id, ckpt_path)
                        rows[epoch_id]['track_metrics'] = track_metrics
                        track_score = track_metrics.get(self.track_metric_key, None)
                        best_track = self._update_best(best_track, track_score, epoch_id, track_src, greater_is_better=True)
                        if self._is_finite_number(track_score):
                            track_msg = f'{self.track_metric_key}={float(track_score):.4f}'
                        else:
                            track_msg = f'{self.track_metric_key}=N/A'
                        print_log(f'[PostEval-Track] epoch={epoch_id} | {track_msg} | ckpt={Path(track_src).name}', logger='current')
                    else:
                        flow_metrics, flow_src = self._eval_flow_for_ckpt(runner, work_dir, epoch_id, ckpt_path)
                        rows[epoch_id]['flow_metrics'] = flow_metrics
                        flow_score = flow_metrics.get(self.flow_metric_key, None)
                        best_flow = self._update_best(best_flow, flow_score, epoch_id, flow_src, greater_is_better=False)
                        if self._is_finite_number(flow_score):
                            flow_msg = f'{self.flow_metric_key}={float(flow_score):.4f}'
                        else:
                            flow_msg = f'{self.flow_metric_key}=N/A'
                        print_log(f'[PostEval-Flow] epoch={epoch_id} | {flow_msg} | ckpt={Path(flow_src).name}', logger='current')

        all_rows = [rows[ep] for ep, _ in sorted(ckpts, key=lambda x: x[0])]

        result_path = work_dir / self.result_filename
        result_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print_log(f'[PostEval] Saved full report: {result_path}', logger='current')

        if best_flow is not None:
            out_dir = work_dir / 'best_flow'
            out_dir.mkdir(parents=True, exist_ok=True)
            _, ep, src = best_flow
            shutil.copy2(str(src), str(out_dir / 'flow_best.pth'))
            print_log(
                f'[PostEval] Best flow ckpt -> {out_dir / "flow_best.pth"} '
                f'({self.flow_metric_key}={best_flow[0]:.4f}, epoch={ep})',
                logger='current',
            )

        if best_track is not None:
            out_dir = work_dir / 'best_tracking'
            out_dir.mkdir(parents=True, exist_ok=True)
            _, ep, src = best_track
            shutil.copy2(str(src), str(out_dir / 'track_best.pth'))
            print_log(
                f'[PostEval] Best tracking ckpt -> {out_dir / "track_best.pth"} '
                f'({self.track_metric_key}={best_track[0]:.4f}, epoch={ep})',
                logger='current',
            )
