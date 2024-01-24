import os
import os.path as osp
from sys import argv

import mmcv
import wandb
from mmcv import Config
from mmcv.runner import init_dist,load_checkpoint, get_dist_info
from mmdet.apis import set_random_seed, train_detector

# Let's take a look at the dataset image
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import subprocess
import random

from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)




wandb.login()
cfg = Config.fromfile(f"/media/quadro/NVME/Mehrab/Current_Experiment/config.py")
# Initialize distributed training environment
init_dist('pytorch', **cfg.dist_params)

set_random_seed(cfg.seed, deterministic=False)

cfg = replace_cfg_vals(cfg)
update_data_root(cfg)    

setup_multi_processes(cfg)

    
val = True
# Build dataset
datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val_loss)]


# Build the detector
# Build the detector
model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))




# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES



# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# Train the model with distributed training
train_detector(model, datasets, cfg, distributed=True, validate=True)



