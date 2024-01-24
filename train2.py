import argparse
import copy


import os
import os.path as osp
from sys import argv
import time
import warnings

import wandb
import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
            '--launcher',
            choices=['none', 'pytorch', 'slurm', 'mpi'],
            default='none',
            help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)
    

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args




def main(): 
    wandb.login()
    seed = set_random_seed(0, deterministic=False)

    args = parse_args()

    cfg = Config.fromfile(f"/media/quadro/NVME/Mehrab/Current_Experiment/config.py")
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)    

    setup_multi_processes(cfg)

    init_dist(args.launcher, **cfg.dist_params)
    cfg.gpu_ids = [2,3]


    val = True
    distributed = True
    # Build dataset

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]


    
    # Build the detector
    model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))



    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=val
        )



if __name__ == '__main__':
    main()







