#%%
import numpy as np
from PIL import Image
import pathlib as path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import tqdm

import torch
import torchvision.transforms as transforms
import cv2 as cv

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

### CORE PRCOCESSING
from lightglue import LightGlue, SuperPoint, ALIKED, match_pair
from lightglue.utils import rbd, numpy_image_to_torch,resize_image
from lightglue import viz2d
from doppelgangers.models.dinov2 import Dinov2FeatureExtractor
from doppelgangers.utils.dataset import imread_rgb

import tqdm
import h5py
from config import CONFIG

db_path = CONFIG.db_path
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)
img_list = np.load(db_path/'pairs_metadata'/'train_pairs_noflip_img_list.npy', allow_pickle=True)
feats_output = './logs/feats-aliked-n16.h5'
dinov2_output = './logs/feats-dinov2-b14-@518.h5'

#%%
@torch.no_grad()
def extract_features(img_list: list, output_path: str):
    ### SuperPoint+LightGlue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.2, nms_radius=2).eval().to(
        device)  # load the extractor

    with h5py.File(output_path, "a", libver="latest") as f:
        for id, img_path in enumerate(tqdm.tqdm(img_list)):
            img = imread_rgb(db_path / 'images' / 'train_set_noflip' / img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            image = numpy_image_to_torch(img)
            image = image.to(device)
            ft = rbd(extractor.extract(image))
            pred = {k: v.cpu().numpy() for k, v in ft.items()}
            grp = f.create_group(str(id))
            for k, v in pred.items():
                grp.create_dataset(k, data=v)

# extract_features(img_list, feats_output)

#%% extract features from dinov2
@torch.no_grad()
def extract_dinov2_features(img_list: list, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    dm = Dinov2FeatureExtractor(half_precision=False, device=device)
    dm.set_smaller_edge_size(518)
    with h5py.File(output_path, "a", libver="latest") as f:
        for id, img_path in enumerate(tqdm.tqdm(img_list)):
            img = imread_rgb(db_path / 'images' / 'train_set_noflip' / img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            image_tensor, grid_size, resize_scale = dm.prepare_image(img)
            with torch.inference_mode():
                image_batch = image_tensor.unsqueeze(0).to(device)
                desc, cls_token = dm.model.get_intermediate_layers(image_batch, n=1, return_class_token=True, reshape=True)[0]
            results = {"grid_size": grid_size, "resize_scale": resize_scale,
                       "descriptors": desc.flatten(-2).transpose(-2, -1).to('cpu').numpy().astype(np.float32)}
            grp = f.create_group(str(id))
            for k, v in results.items():
                grp.create_dataset(k, data=v)

extract_dinov2_features(img_list, dinov2_output)