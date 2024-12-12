#%%
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import tqdm

import torch
import cv2 as cv

### CORE PRCOCESSING
from lightglue import LightGlue, SuperPoint, ALIKED, match_pair
from lightglue.utils import rbd, numpy_image_to_torch,resize_image
from lightglue import viz2d
from doppelgangers.models.dinov2 import Dinov2FeatureExtractor
from doppelgangers.utils.dataset import imread_rgb
from config import CONFIG

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

db_path = CONFIG.db_path
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)

#%% check metadata is available
if(not Path(db_path/'graph_metadata'/'train_pairs_noflip_img_list.npy').exists()):
    print("metadata not found")
    # re-index pairs
    train_pairs = train_pairs[:, :2]
    img_list = np.unique(train_pairs.flatten())

    train_pairs_with_index = []
    for pair in train_pairs:
        id0 = np.where(img_list == pair[0])[0][0]
        id1 = np.where(img_list == pair[1])[0][0]
        train_pairs_with_index.append([id0, id1])
    train_pairs_with_index = np.array(train_pairs_with_index)

    np.save(db_path / 'graph_metadata' / 'train_pairs_noflip_img_list.npy', img_list)
    np.save(db_path / 'graph_metadata' / 'train_pairs_noflip_index.npy', train_pairs_with_index)
else:
    print("metadata found")
    img_list = np.load(db_path / 'graph_metadata' / 'train_pairs_noflip_img_list.npy', allow_pickle=True)
    train_pairs_with_index = np.load(db_path / 'graph_metadata' / 'train_pairs_noflip_index.npy', allow_pickle=True)


#%%
dm = Dinov2FeatureExtractor(half_precision=False, device='cuda')
img0 = cv.cvtColor(imread_rgb(db_path / 'images' / 'train_set_noflip' / img_list[0]), cv.COLOR_BGR2RGB)
img1 = cv.cvtColor(imread_rgb(db_path / 'images' / 'train_set_noflip' / img_list[1]), cv.COLOR_BGR2RGB)
image_tensor, grid_size, resize_scale = dm.prepare_image(img0)
with torch.inference_mode():
    image_batch = image_tensor.unsqueeze(0).to(device)
    desc, cls_token = dm.model.get_intermediate_layers(image_batch, n=1, return_class_token=True, reshape=True)[0]

results = { "grid_size": grid_size,
            "resize_scale": resize_scale,
            "descriptors": desc.flatten(-2).transpose(-2, -1).to('cpu').numpy().astype(np.float32)}

dinov2_output = './logs/feats-dinov2-14.h5'

import h5py
with h5py.File(dinov2_output, "a", libver="latest") as f:
    grp = f.create_group(str(0))
    for k, v in results.items():
        grp.create_dataset(k, data=v)


with h5py.File(dinov2_output, "r", libver="latest") as f:
    print(f.keys())
    print(f['0'].keys())
    print(f['0']['global_descriptor'].shape)
    print(f['0']['descriptors'].shape)

