import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from shutil import copy2

arg_conf = 'doppelgangers/configs/training_configs/doppelgangers_classifier_noflip.yaml'

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# parse config file

with open(arg_conf, 'r') as f:
    config = yaml.safe_load(f)
cfg = dict2namespace(config)

#  Create log_name
cfg_file_name = os.path.splitext(os.path.basename(arg_conf))[0]
run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
# Currently save dir and log_dir are the same
cfg.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
cfg.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
cfg.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
# os.makedirs(config.log_dir + '/config')
# copy2(arg_conf, config.log_dir + '/config')

data_lib = importlib.import_module(cfg.data.type)
loaders = data_lib.get_data_loaders(cfg.data)

train_loader = loaders['train_loader']
test_loader = loaders['test_loader']

import kornia as K
from kornia.utils import tensor_to_image
from kornia.io import load_image
import matplotlib.pyplot as plt

train0 = train_loader.dataset[0]


plt.imshow(tensor_to_image(train0['image'][4:7]))
plt.show()

plt.imshow(tensor_to_image(train0['image'][7:9]))
plt.show()