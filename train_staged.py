import argparse
import json
import multiprocessing
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from datasets.metrics import FlowOfficialMetric, TrackAccuracy
from engine.realtime_loss_plot_hook import RealtimeLossPlotHook


CATEGORY_TO_TRACK_CONFIG = {
    'Car': 'configs/voxel/nuscenes/car.py',
    'Bus': 'configs/voxel/nuscenes/bus.py',
    'Pedestrian': 'configs/voxel/nuscenes/ped.py',
    'Trailer': 'configs/voxel/nuscenes/trailer.py',
    'Truck': 'configs/voxel/nuscenes/truck.py',
}


def _set_nested_if_exists(obj, path, value):
    cur = obj
    for key in path[:-1]:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        if hasattr(cur, '__getitem__'):
            try:
                cur = cur[key]
                continue
            except Exception:
                return False
        return False
    if isinstance(cur, Mapping) and path[-1] in cur:
        cur[path[-1]] = value
        return True
    if hasattr(cur, path[-1]):
        setattr(cur, path[-1], value)
        return True
    return False


def _find_best_ckpt(work_dir: Path, pattern: str):
    candidates = sorted(work_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if len(candidates) > 0 else None


def _find_latest_ckpt(work_dir: Path):
    for pattern in ['epoch_*.pth', '*.pth']:
        candidates = sorted(work_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(candidates) > 0:
            return candidates[0]
    return None


def _latest_log_file(work_dir: Path):
    logs = sorted(work_dir.rglob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if len(logs) > 0 else None


def _parse_best_track_metrics(log_path: Path):
    if log_path is None or not log_path.exists():
        return None, None

    best_precision = None
    best_success_for_precision = None
    p_pat = re.compile(r'precision:\s*([\-+0-9.eE]+)')
    s_pat = re.compile(r'success:\s*([\-+0-9.eE]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p_m = p_pat.search(line)
            s_m = s_pat.search(line)
            if p_m is None or s_m is None:
                continue
            precision = float(p_m.group(1))
            success = float(s_m.group(1))
            if best_precision is None or precision > best_precision:
                best_precision = precision
                best_success_for_precision = success

    return best_precision, best_success_for_precision


def _parse_best_flow_dynamic_mean(log_path: Path):
    if log_path is None or not log_path.exists():
        return None
    pats = [
        re.compile(r'val/Dynamic/Mean:\s*([A-Za-z0-9+\-\.eE]+)'),
        re.compile(r'Dynamic/Mean:\s*([A-Za-z0-9+\-\.eE]+)'),
    ]
    best_val = None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for pat in pats:
                m = pat.search(line)
                if m is None:
                    continue
                cur_val = float(m.group(1))
                if best_val is None or cur_val < best_val:
                    best_val = cur_val
    return best_val


def _strip_module_prefix(state_dict):
    if len(state_dict) == 0:
        return state_dict
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def _adapt_tensor_shape(src_tensor, dst_tensor):
    """Try to adapt source tensor to destination layout for known cases.

    Main use case: spconv weight layout mismatch between checkpoints, e.g.
    [in, out, kz, ky, kx] -> [out, kz, ky, kx, in].
    """
    if src_tensor.shape == dst_tensor.shape:
        return src_tensor, False

    if src_tensor.ndim == 5 and dst_tensor.ndim == 5:
        # Try common sparse-conv kernel layouts.
        cand_list = [
            src_tensor.permute(1, 2, 3, 4, 0).contiguous(),  # [in,out,k,k,k] -> [out,k,k,k,in]
            src_tensor.permute(0, 2, 3, 4, 1).contiguous(),  # [out,in,k,k,k] -> [out,k,k,k,in]
            src_tensor.permute(4, 1, 2, 3, 0).contiguous(),  # fallback variants
            src_tensor.permute(4, 0, 1, 2, 3).contiguous(),
            src_tensor.permute(1, 0, 2, 3, 4).contiguous(),
        ]
        for cand in cand_list:
            if cand.shape == dst_tensor.shape:
                return cand, True

    return None, False


def _load_backbone_fuser(track_model, flow_ckpt_path: Path):
    ckpt = torch.load(str(flow_ckpt_path), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Invalid checkpoint format: {flow_ckpt_path}')
    state_dict = _strip_module_prefix(state_dict)

    filtered = {
        k: v for k, v in state_dict.items()
        if k.startswith('backbone.') or k.startswith('fuse.')
    }
    if len(filtered) == 0:
        raise RuntimeError(f'No backbone/fuse weights found in {flow_ckpt_path}')

    named_params = dict(track_model.named_parameters())
    named_buffers = dict(track_model.named_buffers())
    converted = 0
    skipped_not_found = 0
    skipped_shape = 0
    loaded_count = 0

    for k, v in filtered.items():
        if k in named_params:
            dst = named_params[k].data
        elif k in named_buffers:
            dst = named_buffers[k].data
        else:
            skipped_not_found += 1
            continue

        src = v.detach().cpu() if isinstance(v, torch.Tensor) else v
        adapted, did_convert = _adapt_tensor_shape(src, dst)
        if adapted is None:
            skipped_shape += 1
            continue

        dst.copy_(adapted.to(dtype=dst.dtype, device=dst.device))
        loaded_count += 1
        if did_convert:
            converted += 1

    if loaded_count == 0:
        raise RuntimeError(
            f'No compatible backbone/fuse weights can be loaded from {flow_ckpt_path}. '
            f'filtered={len(filtered)}, skipped_not_found={skipped_not_found}, skipped_shape={skipped_shape}'
        )

    print(f'[Stage2] Candidate flow keys (backbone/fuse): {len(filtered)}')
    print(f'[Stage2] Loaded compatible keys: {loaded_count}')
    print(f'[Stage2] Converted layout keys: {converted}')
    print(f'[Stage2] Skipped (not found in tracking model): {skipped_not_found}')
    print(f'[Stage2] Skipped (shape incompatible): {skipped_shape}')
    return loaded_count, skipped_shape, skipped_not_found


def _archive_checkpoint(src_ckpt: Path, archive_dir: Path, dst_name: str):
    if src_ckpt is None:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / dst_name
    shutil.copy2(str(src_ckpt), str(dst))
    return dst


def _get_logger_interval(cfg, fallback=50):
    try:
        hooks = cfg.get('default_hooks', None)
        if hooks is None:
            return int(fallback)
        logger_cfg = hooks.get('logger', None)
        if logger_cfg is None:
            return int(fallback)
        interval = logger_cfg.get('interval', fallback)
        interval = int(interval)
        if interval > 0:
            return interval
    except Exception:
        pass
    return int(fallback)


def parse_args():
    parser = argparse.ArgumentParser(description='Two-stage training: Flow-only -> Tracking-only')
    parser.add_argument(
        '--category',
        default='Car',
        choices=['Car', 'Bus', 'Pedestrian', 'Trailer', 'Truck'],
        help='tracking category for stage2',
    )
    parser.add_argument(
        '--flow_config',
        default='configs/voxel/nuscenes/flow_seq5_native.py',
        help='flow-only config file path',
    )
    parser.add_argument(
        '--track_config',
        default=None,
        help='tracking config file path; if omitted use category default',
    )
    parser.add_argument(
        '--flow_data_dir',
        default='C:/develop/OpenSceneFlow/data/processed',
        help='flow native h5 root path',
    )
    parser.add_argument('--flow_batch_size', type=int, default=32, help='flow train batch size')
    parser.add_argument('--flow_val_batch_size', type=int, default=8, help='flow val batch size')
    parser.add_argument('--flow_num_workers', type=int, default=4, help='flow train dataloader workers')
    parser.add_argument('--flow_val_num_workers', type=int, default=2, help='flow val dataloader workers')
    parser.add_argument(
        '--track_data_dir',
        default='C:/Users/SunChanggeng/Desktop/NuScenes_correct',
        help='raw nuscenes path for tracking',
    )
    parser.add_argument('--flow_work_dir', default='./work_dir_flow_stage', help='flow stage work dir')
    parser.add_argument('--track_work_dir', default='./work_dir_track_stage', help='tracking stage work dir')
    parser.add_argument('--flow_resume', default=None, help='resume checkpoint for flow stage')
    parser.add_argument('--track_resume', default=None, help='resume checkpoint for tracking stage')
    parser.add_argument(
        '--skip_flow_stage',
        action='store_true',
        help='skip flow training stage and directly start tracking with --flow_init_ckpt',
    )
    parser.add_argument(
        '--flow_init_ckpt',
        default=None,
        help='flow checkpoint path used to initialize stage2 backbone/fuser when --skip_flow_stage is set',
    )
    parser.add_argument(
        '--plot_sample_interval',
        type=int,
        default=100,
        help='deprecated fallback; when not syncing from config logger interval',
    )
    parser.add_argument(
        '--plot_no_window',
        action='store_true',
        help='disable realtime plot window and only save plot images',
    )
    parser.add_argument(
        '--plot_no_save',
        action='store_true',
        help='disable saving plot images and only keep realtime window',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _build_flow_runner(cfg, args):
    plot_interval = _get_logger_interval(cfg, fallback=args.plot_sample_interval)
    custom_hooks = []
    if cfg.get('custom_hooks', None):
        custom_hooks.extend(cfg.custom_hooks)
    custom_hooks.append(
        RealtimeLossPlotHook(
            iter_interval=plot_interval,
            task_name='flow',
            enable_window=not args.plot_no_window,
            save_image=not args.plot_no_save,
        )
    )

    runner_kwargs = dict(
        model=cfg.model,
        resume=args.flow_resume,
        load_from=None,
        visualizer=cfg.visualizer,
        default_hooks=cfg.default_hooks,
        env_cfg=cfg.env_cfg,
        work_dir=args.flow_work_dir,
        train_cfg=cfg.train_cfg,
        train_dataloader=cfg.train_dataloader,
        optim_wrapper=cfg.optim_wrapper,
        launcher=args.launcher,
        custom_hooks=custom_hooks,
        cfg=dict(),
    )
    if hasattr(cfg, 'val_dataloader') and cfg.get('val_cfg', None) is not None:
        runner_kwargs.update(
            val_dataloader=cfg.val_dataloader,
            val_evaluator=Evaluator(FlowOfficialMetric()),
            val_cfg=cfg.val_cfg,
        )
    return Runner(**runner_kwargs)


def _build_track_runner(cfg, args):
    plot_interval = _get_logger_interval(cfg, fallback=args.plot_sample_interval)
    custom_hooks = []
    if cfg.get('custom_hooks', None):
        custom_hooks.extend(cfg.custom_hooks)
    custom_hooks.append(
        RealtimeLossPlotHook(
            iter_interval=plot_interval,
            task_name='tracking',
            enable_window=not args.plot_no_window,
            save_image=not args.plot_no_save,
        )
    )

    runner_kwargs = dict(
        model=cfg.model,
        resume=args.track_resume,
        load_from=None,
        visualizer=cfg.visualizer,
        default_hooks=cfg.default_hooks,
        env_cfg=cfg.env_cfg,
        work_dir=args.track_work_dir,
        train_cfg=cfg.train_cfg,
        train_dataloader=cfg.train_dataloader,
        optim_wrapper=cfg.optim_wrapper,
        launcher=args.launcher,
        custom_hooks=custom_hooks,
        cfg=dict(),
    )
    if hasattr(cfg, 'val_dataloader') and cfg.get('val_cfg', None) is not None:
        runner_kwargs.update(
            val_dataloader=cfg.val_dataloader,
            val_evaluator=Evaluator(TrackAccuracy()),
            val_cfg=cfg.val_cfg,
        )
    return Runner(**runner_kwargs)


def main():
    args = parse_args()
    run_tag = time.strftime('%Y%m%d_%H%M%S')

    flow_work_dir = Path(args.flow_work_dir)
    if args.skip_flow_stage:
        if args.flow_init_ckpt is None:
            raise RuntimeError('--skip_flow_stage requires --flow_init_ckpt')
        flow_best_ckpt = Path(args.flow_init_ckpt)
        if not flow_best_ckpt.exists():
            raise FileNotFoundError(f'--flow_init_ckpt not found: {flow_best_ckpt}')
        print(f'[Stage1] Skipped. Using provided flow checkpoint: {flow_best_ckpt}')
    else:
        # Stage1: flow-only
        flow_cfg = Config.fromfile(args.flow_config)
        _set_nested_if_exists(flow_cfg, ['flow_h5_dir'], args.flow_data_dir)
        _set_nested_if_exists(flow_cfg, ['train_dataloader', 'dataset', 'path'], args.flow_data_dir)
        _set_nested_if_exists(flow_cfg, ['val_dataloader', 'dataset', 'path'], args.flow_data_dir)
        _set_nested_if_exists(flow_cfg, ['batch_size'], args.flow_batch_size)
        _set_nested_if_exists(flow_cfg, ['val_batch_size'], args.flow_val_batch_size)
        _set_nested_if_exists(flow_cfg, ['train_dataloader', 'batch_size'], args.flow_batch_size)
        _set_nested_if_exists(flow_cfg, ['val_dataloader', 'batch_size'], args.flow_val_batch_size)
        _set_nested_if_exists(flow_cfg, ['train_dataloader', 'num_workers'], args.flow_num_workers)
        _set_nested_if_exists(flow_cfg, ['val_dataloader', 'num_workers'], args.flow_val_num_workers)
        _set_nested_if_exists(flow_cfg, ['train_dataloader', 'persistent_workers'], args.flow_num_workers > 0)
        _set_nested_if_exists(flow_cfg, ['val_dataloader', 'persistent_workers'], args.flow_val_num_workers > 0)
        _set_nested_if_exists(flow_cfg, ['train_dataloader', 'pin_memory'], True)
        _set_nested_if_exists(flow_cfg, ['val_dataloader', 'pin_memory'], True)

        flow_runner = _build_flow_runner(flow_cfg, args)
        print('[Stage1] Starting flow-only training...')
        flow_runner.train()
        flow_best_ckpt = _find_best_ckpt(flow_work_dir, 'best_val_Dynamic_Mean*.pth')
        if flow_best_ckpt is None:
            flow_best_ckpt = _find_best_ckpt(flow_work_dir, 'best_Dynamic_Mean*.pth')
        if flow_best_ckpt is None:
            flow_best_ckpt = _find_best_ckpt(flow_work_dir, 'best_flow_epe*.pth')
        if flow_best_ckpt is None:
            flow_best_ckpt = _find_latest_ckpt(flow_work_dir)
        if flow_best_ckpt is None:
            raise RuntimeError(f'[Stage1] Cannot find flow checkpoint under {flow_work_dir}')
        print(f'[Stage1] Selected flow checkpoint: {flow_best_ckpt}')

    # Stage2: tracking-only
    track_cfg_path = args.track_config if args.track_config else CATEGORY_TO_TRACK_CONFIG[args.category]
    track_cfg = Config.fromfile(track_cfg_path)
    _set_nested_if_exists(track_cfg, ['data_dir'], args.track_data_dir)
    _set_nested_if_exists(track_cfg, ['category_name'], args.category)
    _set_nested_if_exists(track_cfg, ['train_dataloader', 'dataset', 'dataset', 'path'], args.track_data_dir)
    _set_nested_if_exists(track_cfg, ['val_dataloader', 'dataset', 'dataset', 'path'], args.track_data_dir)
    _set_nested_if_exists(track_cfg, ['test_dataloader', 'dataset', 'dataset', 'path'], args.track_data_dir)
    _set_nested_if_exists(track_cfg, ['train_dataloader', 'dataset', 'dataset', 'category_name'], args.category)
    _set_nested_if_exists(track_cfg, ['val_dataloader', 'dataset', 'dataset', 'category_name'], args.category)
    _set_nested_if_exists(track_cfg, ['test_dataloader', 'dataset', 'dataset', 'category_name'], args.category)

    track_runner = _build_track_runner(track_cfg, args)
    loaded_keys, missing_keys, unexpected_keys = _load_backbone_fuser(track_runner.model, flow_best_ckpt)
    print('[Stage2] Starting tracking-only training...')
    track_runner.train()

    track_work_dir = Path(args.track_work_dir)
    track_best_ckpt = _find_best_ckpt(track_work_dir, 'best_precision*.pth')
    if track_best_ckpt is None:
        track_best_ckpt = _find_latest_ckpt(track_work_dir)

    archive_dir = track_work_dir / 'staged_best_ckpts' / run_tag
    archived_flow_best = _archive_checkpoint(flow_best_ckpt, archive_dir, 'flow_best.pth')
    archived_track_best = _archive_checkpoint(track_best_ckpt, archive_dir, 'track_best.pth')

    flow_log = _latest_log_file(flow_work_dir)
    track_log = _latest_log_file(track_work_dir)
    best_flow_dynamic_mean = _parse_best_flow_dynamic_mean(flow_log)
    best_precision, best_success = _parse_best_track_metrics(track_log)

    summary = dict(
        flow_best_ckpt=str(flow_best_ckpt) if flow_best_ckpt is not None else None,
        track_best_ckpt=str(track_best_ckpt) if track_best_ckpt is not None else None,
        archived_flow_best_ckpt=str(archived_flow_best) if archived_flow_best is not None else None,
        archived_track_best_ckpt=str(archived_track_best) if archived_track_best is not None else None,
        best_flow_dynamic_mean=best_flow_dynamic_mean,
        best_precision=best_precision,
        best_success=best_success,
        loaded_backbone_fuser_keys=loaded_keys,
        missing_keys_after_transfer=missing_keys,
        unexpected_keys_after_transfer=unexpected_keys,
    )

    track_work_dir.mkdir(parents=True, exist_ok=True)
    summary_json = track_work_dir / 'staged_summary.json'
    summary_txt = track_work_dir / 'staged_summary.txt'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')

    print('[Done] Staged training summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
