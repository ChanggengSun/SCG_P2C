import argparse
import multiprocessing
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from datasets.metrics import TrackAccuracy


# ========== 将 collate 函数移到主文件中 ==========
def simple_collate_fn(batch):
    """可序列化的简单 collate 函数"""
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--load_from', default=None, help='path to pretrained model checkpoint')
    parser.add_argument('--config', default='configs/voxel/kitti/car.py', help='train config file path')
    parser.add_argument('--resume', default=None, help='train config file path')
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
    cfg = Config.fromfile(args.config)

    metric = TrackAccuracy()
    evaluator = Evaluator(metric)

    runner = Runner(
        model=cfg.model,
        resume=args.resume,
        load_from=args.load_from,   # 新增这一行
        visualizer=cfg.visualizer,
        default_hooks=cfg.default_hooks,
        env_cfg=cfg.env_cfg,
        work_dir='./work_dir',
        train_cfg=cfg.train_cfg,
        train_dataloader=cfg.train_dataloader,
        val_dataloader=cfg.val_dataloader,
        val_evaluator=evaluator,
        val_cfg=cfg.val_cfg,
        optim_wrapper=cfg.optim_wrapper,
        launcher=args.launcher,
        cfg=dict()
    )

    runner.train()


if __name__ == '__main__':
    # Windows 多进程必需：防止递归创建进程
    multiprocessing.freeze_support()
    main()
