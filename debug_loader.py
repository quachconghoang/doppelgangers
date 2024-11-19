#%%
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

import cv2 as cv
from pathlib import Path

arg_conf = 'doppelgangers/configs/training_configs/doppelgangers_classifier_aliked_noflip.yaml'

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

# train_loader = loaders['train_loader']
test_loader = loaders['test_loader']
train_loader = loaders['train_loader']

#%%
from kornia.utils import tensor_to_image
import numpy as np
# get random sample from train_loader
rng = np.random.default_rng()
# get 100 samples
idx = rng.choice(len(test_loader.dataset), 100, replace=False)

for id in idx:
    db = test_loader.dataset[id]
    inp = db['image']
    img1_warp = tensor_to_image(inp[4:7])
    img0_warp = tensor_to_image(inp[7:10])

    warp_preview = np.ascontiguousarray(np.concatenate([img0_warp, img1_warp], axis=1) * 255, dtype=np.uint8)
    #creat tmp folder
    Path('./logs').mkdir(parents=True, exist_ok=True)
    cv.imwrite(f'./logs/{id}_non_warp.jpg', warp_preview)

#%%
# import kornia as K
# from kornia.utils import tensor_to_image
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# test_pairs = np.load('./data/doppelgangers_dataset/doppelgangers/pairs_metadata/test_pairs.npy', allow_pickle=True)
# logs = np.load('./val_logs/doppelgangers_classifier_noflip_val_2024-Oct-06-10-40-46/test_doppelgangers_list.npy', allow_pickle=True).item()
# gt_list = logs['gt']
# pred_list = logs['pred']
# prob_list = logs['prob']
# # precision = np.sum(gt_list == pred_list) / len(gt_list) # Precision = 0.88, AP = 0.95???
# # ap = compute_ap(gt_list, prob_list)
# failed = logs['pred'] != logs['gt']
# failed_idx = np.where(failed)[0]
#
# def get_resized_wh(w, h, resize=None):
#     if resize is not None:  # resize the longer edge
#         scale = resize / max(h, w)
#         w_new, h_new = int(round(w*scale)), int(round(h*scale))
#     else:
#         w_new, h_new = w, h
#     return w_new, h_new
#
# img_size = 1024
# img_dir = Path('./data/doppelgangers_dataset/doppelgangers/images/test_set')
# font = cv.FONT_HERSHEY_SIMPLEX
#
# for pair_ID in failed_idx:
#     print(pair_ID)
#     pair = test_pairs[pair_ID]
#     db = test_loader.dataset[pair_ID]
#     inp = db['image']
#     img1_warp = tensor_to_image(inp[4:7])
#     img0_warp = tensor_to_image(inp[7:10])
#     warp_preview = np.ascontiguousarray(np.concatenate([img0_warp, img1_warp], axis=1)*255, dtype=np.uint8)
#     cv.putText(warp_preview, 'ID: %d' % pair_ID, (10, 50), font, 1, (0, 0, 255), 2, cv.LINE_AA)
#     cv.putText(warp_preview, 'Warp 0-to-1', (10, 100), font, 1, (0, 0, 255), 2, cv.LINE_AA)
#     cv.imwrite(f'./val_logs/pairs/{pair_ID}_warp.jpg', warp_preview)
#
#     valid_match = db['valid_match']
#     img0 = cv.imread(str(img_dir / pair[0]), cv.IMREAD_COLOR)
#     img1 = cv.imread(str(img_dir / pair[1]), cv.IMREAD_COLOR)
#     w, h = img0.shape[1], img0.shape[0]
#     w_new, h_new = get_resized_wh(w, h, img_size)
#     img0 = cv.resize(img0, (w_new, h_new))
#
#     w1, h1 = img1.shape[1], img1.shape[0]
#     w_new = round((w1/h1) * h_new)
#     img1 = cv.resize(img1, (w_new, h_new))
#     # Visualize failed pairs
#     img_viz = np.concatenate([img0, img1], axis=1)
#     cv.putText(img_viz, 'ID: %d' % pair_ID, (10, 50), font, 1, (0, 0, 255), 2, cv.LINE_AA)
#     cv.putText(img_viz, 'GT: %d' % gt_list[pair_ID], (10, 100), font, 1, (0, 0, 255), 2, cv.LINE_AA)
#     if not valid_match:
#         cv.putText(img_viz, 'Invalid match', (10, 150), font, 1, (0, 0, 255), 2, cv.LINE_AA)
#     cv.imwrite(f'./val_logs/pairs/{pair_ID}.jpg', img_viz)
#     sub_folder = 'false'
#     if gt_list[pair_ID] == 1:
#         sub_folder = 'true'
#     cv.imwrite(f'./val_logs/pairs/{sub_folder}/{pair_ID}_0.jpg', img0)
#     cv.imwrite(f'./val_logs/pairs/{sub_folder}/{pair_ID}_1.jpg', img1)