import argparse
import multiprocessing
import os
from collections.abc import Mapping
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from datasets.metrics import TrackAccuracy
from engine import JointFlowEvalHook, RealtimeLossPlotHook


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
                pass
            return False
    if isinstance(cur, Mapping) and path[-1] in cur:
        cur[path[-1]] = value
        return True
    if hasattr(cur, path[-1]):
        setattr(cur, path[-1], value)
        return True
    return False


def _get_nested_if_exists(obj, path):
    cur = obj
    for key in path:
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
                return None
        return None
    return cur


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


def _build_legacy_log_processor_cfg(cfg):
    """Tracking uses old P2P style; flow keeps single-flow style."""
    cfg_lp = cfg.get('log_processor', None)
    log_processor_cfg = dict(cfg_lp) if isinstance(cfg_lp, Mapping) else {}

    by_epoch = _get_nested_if_exists(cfg, ['train_cfg', 'by_epoch'])
    if 'by_epoch' not in log_processor_cfg:
        log_processor_cfg['by_epoch'] = bool(by_epoch) if by_epoch is not None else True

    # Keep mmengine default smoothing to align with old standalone tracking project.
    log_processor_cfg['window_size'] = 10

    # Keep flow branch in single-flow style: log current step value (raw current).
    custom_cfg = list(log_processor_cfg.get('custom_cfg', []))
    has_flow_current = False
    for item in custom_cfg:
        if not isinstance(item, Mapping):
            continue
        if item.get('data_src') == 'flow_loss' and item.get('log_name', 'flow_loss') == 'flow_loss':
            item['method_name'] = 'current'
            item.pop('window_size', None)
            has_flow_current = True
            break
    if not has_flow_current:
        custom_cfg.append(dict(data_src='flow_loss', method_name='current'))
    log_processor_cfg['custom_cfg'] = custom_cfg
    return log_processor_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Joint synchronized training for tracking + flow')
    parser.add_argument('--load_from', default=None, help='path to pretrained model checkpoint')
    parser.add_argument(
        '--config',
        default='configs/voxel/nuscenes/joint_track_flow_seq5_sync.py',
        help='joint train config file path')
    parser.add_argument('--resume', default=None, help='path to checkpoint for resuming')
    parser.add_argument(
        '--category',
        default=None,
        choices=['Car', 'Bus', 'Pedestrian', 'Trailer', 'Truck'],
        help='tracking category for joint training; overrides config category_name')
    parser.add_argument(
        '--flow_data_dir',
        default='C:/develop/OpenSceneFlow/data/processed',
        help='flow processed h5 root path')
    parser.add_argument(
        '--track_data_dir',
        default='C:/Users/SunChanggeng/Desktop/NuScenes_correct',
        help='raw nuScenes path for tracking')
    parser.add_argument(
        '--track_batch_size',
        type=int,
        default=None,
        help='optional override for tracking train batch size')
    parser.add_argument(
        '--flow_batch_size',
        type=int,
        default=None,
        help='optional override for flow train batch size')
    parser.add_argument(
        '--sidecar_file',
        default=None,
        help='optional sidecar file name under flow_data_dir/<split>; default uses joint_seq5_<category>.pkl')
    parser.add_argument(
        '--disable_flow_val',
        action='store_true',
        help='disable auxiliary flow validation hook during joint training')
    parser.add_argument(
        '--flow_repo_root',
        default='C:/develop/OpenSceneFlow',
        help='OpenSceneFlow repo root used by official flow metric')
    parser.add_argument('--flow_val_batch_size', type=int, default=1, help='flow validation batch size')
    parser.add_argument('--flow_val_num_workers', type=int, default=0, help='flow validation dataloader workers')
    parser.add_argument(
        '--plot_sample_interval',
        type=int,
        default=100,
        help='deprecated fallback; when not syncing from config logger interval')
    parser.add_argument(
        '--plot_no_window',
        action='store_true',
        help='disable realtime plot window and only save images')
    parser.add_argument(
        '--plot_no_save',
        action='store_true',
        help='disable saving plot images and only keep realtime window')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the config, key=value format')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Inject data roots for synchronized joint training.
    _set_nested_if_exists(cfg, ['flow_h5_dir'], args.flow_data_dir)
    _set_nested_if_exists(cfg, ['raw_data_dir'], args.track_data_dir)

    # Tracking stream dataset roots
    _set_nested_if_exists(cfg, ['track_train_dataloader', 'dataset', 'path'], args.flow_data_dir)
    _set_nested_if_exists(cfg, ['track_train_dataloader', 'dataset', 'track_data_dir'], args.track_data_dir)
    _set_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'path'], args.flow_data_dir)
    _set_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'track_data_dir'], args.track_data_dir)

    # Flow stream dataset roots
    _set_nested_if_exists(cfg, ['flow_train_dataloader', 'dataset', 'path'], args.flow_data_dir)
    _set_nested_if_exists(cfg, ['train_cfg', 'flow_dataloader', 'dataset', 'path'], args.flow_data_dir)

    # Tracking val roots
    _set_nested_if_exists(cfg, ['val_dataloader', 'dataset', 'dataset', 'path'], args.track_data_dir)

    if args.track_batch_size is not None:
        _set_nested_if_exists(cfg, ['track_batch_size'], int(args.track_batch_size))
        _set_nested_if_exists(cfg, ['track_train_dataloader', 'batch_size'], int(args.track_batch_size))
        _set_nested_if_exists(cfg, ['train_dataloader', 'batch_size'], int(args.track_batch_size))

    if args.flow_batch_size is not None:
        _set_nested_if_exists(cfg, ['flow_batch_size'], int(args.flow_batch_size))
        _set_nested_if_exists(cfg, ['flow_train_dataloader', 'batch_size'], int(args.flow_batch_size))
        _set_nested_if_exists(cfg, ['train_cfg', 'flow_dataloader', 'batch_size'], int(args.flow_batch_size))

    if args.category is not None:
        updated = False
        updated |= _set_nested_if_exists(cfg, ['category_name'], args.category)
        updated |= _set_nested_if_exists(
            cfg, ['track_train_dataloader', 'dataset', 'category_name'], args.category)
        updated |= _set_nested_if_exists(
            cfg, ['flow_train_dataloader', 'dataset', 'category_name'], args.category)
        updated |= _set_nested_if_exists(
            cfg, ['train_cfg', 'flow_dataloader', 'dataset', 'category_name'], args.category)
        updated |= _set_nested_if_exists(
            cfg, ['train_dataloader', 'dataset', 'category_name'], args.category)
        updated |= _set_nested_if_exists(
            cfg, ['val_dataloader', 'dataset', 'dataset', 'category_name'], args.category)
        if not updated:
            raise RuntimeError('Cannot find category_name path in config for --category override.')

    effective_category = (
        args.category
        or _get_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'category_name'])
        or _get_nested_if_exists(cfg, ['category_name'])
    )
    if effective_category is None:
        effective_category = 'Car'
    sidecar_file = args.sidecar_file if args.sidecar_file else f'joint_seq5_{str(effective_category).lower()}.pkl'
    _set_nested_if_exists(cfg, ['track_train_dataloader', 'dataset', 'sidecar_file'], sidecar_file)
    _set_nested_if_exists(cfg, ['flow_train_dataloader', 'dataset', 'sidecar_file'], sidecar_file)
    _set_nested_if_exists(cfg, ['train_cfg', 'flow_dataloader', 'dataset', 'sidecar_file'], sidecar_file)
    _set_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'sidecar_file'], sidecar_file)

    logger_interval = _get_logger_interval(cfg, fallback=args.plot_sample_interval)
    log_processor_cfg = _build_legacy_log_processor_cfg(cfg)
    custom_hooks = []
    if cfg.get('custom_hooks', None):
        custom_hooks.extend(cfg.custom_hooks)

    custom_hooks.append(
        RealtimeLossPlotHook(
            iter_interval=logger_interval,
            task_name='joint_tracking',
            enable_window=not args.plot_no_window,
            save_image=not args.plot_no_save,
            include_keys=['tracking_loss', 'track_pair'],
        )
    )
    custom_hooks.append(
        RealtimeLossPlotHook(
            iter_interval=logger_interval,
            task_name='joint_flow',
            enable_window=not args.plot_no_window,
            save_image=not args.plot_no_save,
            include_keys=['flow_loss'],
        )
    )

    if not args.disable_flow_val:
        input_dim = _get_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'input_dim'])
        history_frames = _get_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'history_frames'])
        remove_ground = _get_nested_if_exists(cfg, ['train_dataloader', 'dataset', 'remove_ground'])
        val_interval = _get_nested_if_exists(cfg, ['train_cfg', 'val_interval']) or 1

        custom_hooks.append(
            JointFlowEvalHook(
                flow_data_dir=args.flow_data_dir,
                flow_repo_root=args.flow_repo_root,
                split='val',
                interval=int(val_interval),
                start=1,
                batch_size=args.flow_val_batch_size,
                num_workers=args.flow_val_num_workers,
                remove_ground=bool(remove_ground) if remove_ground is not None else False,
                input_dim=int(input_dim) if input_dim is not None else 4,
                history_frames=int(history_frames) if history_frames is not None else 3,
            )
        )

    # If dual dataloaders are configured, make sure train_cfg receives stream cfg.
    if _get_nested_if_exists(cfg, ['train_cfg', 'flow_dataloader']) is None:
        flow_loader_cfg = _get_nested_if_exists(cfg, ['flow_train_dataloader'])
        if flow_loader_cfg is not None and _get_nested_if_exists(cfg, ['train_cfg']) is not None:
            cfg.train_cfg['flow_dataloader'] = flow_loader_cfg

    if _get_nested_if_exists(cfg, ['train_cfg', 'track_dataloader']) is None:
        track_loader_cfg = _get_nested_if_exists(cfg, ['track_train_dataloader'])
        if track_loader_cfg is not None and _get_nested_if_exists(cfg, ['train_cfg']) is not None:
            cfg.train_cfg['track_dataloader'] = track_loader_cfg

    loop_type = _get_nested_if_exists(cfg, ['train_cfg', 'type'])
    if str(loop_type) == 'JointSeq5IndependentDualTrainLoop':
        train_loader_cfg = _get_nested_if_exists(cfg, ['flow_train_dataloader'])
    else:
        train_loader_cfg = _get_nested_if_exists(cfg, ['track_train_dataloader'])
    if train_loader_cfg is None:
        train_loader_cfg = cfg.train_dataloader

    runner_kwargs = dict(
        model=cfg.model,
        resume=args.resume,
        load_from=args.load_from,
        visualizer=cfg.visualizer,
        default_hooks=cfg.default_hooks,
        env_cfg=cfg.env_cfg,
        work_dir='./work_dir_joint',
        train_cfg=cfg.train_cfg,
        train_dataloader=train_loader_cfg,
        optim_wrapper=cfg.optim_wrapper,
        launcher=args.launcher,
        log_processor=log_processor_cfg,
        custom_hooks=custom_hooks,
        cfg=dict(),
    )

    if hasattr(cfg, 'val_dataloader') and cfg.get('val_cfg', None) is not None:
        metric = TrackAccuracy()
        evaluator = Evaluator(metric)
        runner_kwargs.update(
            val_dataloader=cfg.val_dataloader,
            val_evaluator=evaluator,
            val_cfg=cfg.val_cfg,
        )

    runner = Runner(**runner_kwargs)
    runner.train()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


