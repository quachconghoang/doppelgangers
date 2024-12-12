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

torch.set_grad_enabled(False)
db_path = path.Path('./data/doppelgangers_dataset/doppelgangers/')
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)
test_pairs = np.load(db_path/'pairs_metadata'/'test_pairs.npy', allow_pickle=True)

### SuperPoint+LightGlue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = ALIKED(max_num_keypoints=4096, detection_threshold=0.2, nms_radius=2).eval().to(device)  # load the extractor
matcher = LightGlue(features='aliked',depth_confidence=-1, width_confidence=-1).eval().to(device)

def load_doppleganger_img(pair, dataset_path, db_type, size=None):
    img_root = dataset_path / 'images' / db_type
    img0 = imread_rgb(img_root / pair[0])
    img1 = imread_rgb(img_root / pair[1])
    if size is not None:
        img0 = resize_image(img0, size=size)[0]
        img1 = resize_image(img1, size=size)[0]
    img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    image0 = numpy_image_to_torch(img0)
    image1 = numpy_image_to_torch(img1)
    return image0, image1

def draw_matches(ft0, ft1, matches, img0, img1):
    _ft0, _ft1, matches01 = [rbd(x) for x in [ft0, ft1, matches]]  # remove batch dimension
    kpts0, kpts1, matches, scores = _ft0["keypoints"], _ft1["keypoints"], matches01["matches"],  matches01["scores"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    # torch tensor empty zero 1x3x1024x1024
    prv_img0 = torch.zeros(3, 1024, 1024)
    prv_img1 = torch.zeros(3, 1024, 1024)
    prv_img0[:, 0:img0.shape[1], 0:img0.shape[2]] = img0
    prv_img1[:, 0:img1.shape[1], 0:img1.shape[2]] = img1

    viz2d.plot_images([prv_img0, prv_img1])
    # viz2d.plot_keypoints([kpts0, kpts1], colors="blue", ps=3)
    viz2d.plot_matches(m_kpts0, m_kpts1, lw=0.2, ps=0 )

    scores = scores.cpu().numpy()
    kpc0, kpc1 = viz2d.cm_RdGn(scores), viz2d.cm_RdGn(scores)
    viz2d.plot_keypoints([m_kpts0, m_kpts1], colors=[kpc0, kpc1], ps=5)
    plt.show()

def save_matches(ft0, ft1, matches, img0, img1, save_path, preview_path = None):
    _ft0, _ft1, matches01 = [rbd(x) for x in [ft0, ft1, matches]]  # remove batch dimension
    kpts0, kpts1, matches, scores = _ft0["keypoints"], _ft1["keypoints"], matches01["matches"],  matches01["scores"]

    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    scores = scores.cpu().numpy()
    kp0 = m_kpts0.cpu().numpy()
    kp1 = m_kpts1.cpu().numpy()
    ### save to numpy pickle overwriting
    np.save(save_path, {'kpt0': kp0, 'kpt1': kp1, 'conf': scores})

    # random T/F for 1% True
    can_draw = np.random.choice([True, False], p=[0.01, 0.99])
    if preview_path is not None and can_draw:
        # print('saving preview')
        prv_img0 = torch.zeros(3, 1024, 1024)
        prv_img1 = torch.zeros(3, 1024, 1024)
        prv_img0[:, 0:img0.shape[1], 0:img0.shape[2]] = img0
        prv_img1[:, 0:img1.shape[1], 0:img1.shape[2]] = img1
        viz2d.plot_images([prv_img0, prv_img1])
        # viz2d.plot_keypoints([kpts0, kpts1], colors="blue", ps=3)
        viz2d.plot_matches(m_kpts0, m_kpts1, lw=0.2, ps=0)
        kpc0, kpc1 = viz2d.cm_RdGn(scores), viz2d.cm_RdGn(scores)
        viz2d.plot_keypoints([m_kpts0, m_kpts1], colors=[kpc0, kpc1], ps=5)
        viz2d.save_plot(preview_path)
        plt.close()

#%%
# db_type = 'train_set_noflip' # 'train_set_noflip', 'test_set'
# pair_ID = 0
# pair = train_pairs[pair_ID]
# torch.set_grad_enabled(False)
# image0, image1 = load_doppleganger_img(pair, db_path, db_type, size=1024)
#
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
# matches01 = matcher({'image0': feats0, 'image1': feats1})
# save_path = db_path/'aliked_1024_lg_matches'/db_type/f'{pair_ID}.npy'
# preview_path = db_path/'aliked_1024_lg_matches_preview'/db_type/f'{pair_ID}.png'
# draw_matches(feats0, feats1, matches01, img0=image0, img1=image1)
# save_matches(feats0, feats1, matches01, image0, image1, save_path, preview_path)

#%%
db_type = 'train_set_noflip'
if not (db_path/'aliked_1024_lg_matches'/db_type).exists():
    (db_path/'aliked_1024_lg_matches'/db_type).mkdir(parents=True)
if not (db_path/'aliked_1024_lg_matches_preview'/db_type).exists():
    (db_path/'aliked_1024_lg_matches_preview'/db_type).mkdir(parents=True)
for pair_ID in tqdm.tqdm(range(train_pairs.shape[0])):
    # print('processing pair:', pair_ID)
    pair = train_pairs[pair_ID]
    img_root = db_path / 'images' / db_type
    image0, image1 = load_doppleganger_img(pair, db_path, db_type, size=1024)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    # save matches
    save_path = db_path/'aliked_1024_lg_matches'/db_type/f'{pair_ID}.npy'
    preview_path = db_path/'aliked_1024_lg_matches_preview'/db_type/f'{pair_ID}.png'
    save_matches(feats0, feats1, matches01, image0, image1, save_path, preview_path)

#%%
db_type = 'test_set'
if not (db_path/'aliked_1024_lg_matches'/db_type).exists():
    (db_path/'aliked_1024_lg_matches'/db_type).mkdir(parents=True)
if not (db_path/'aliked_1024_lg_matches_preview'/db_type).exists():
    (db_path/'aliked_1024_lg_matches_preview'/db_type).mkdir(parents=True)

for pair_ID in tqdm.tqdm(range(test_pairs.shape[0])):
    # print('processing pair:', pair_ID)
    pair = test_pairs[pair_ID]
    img_root = db_path / 'images' / db_type
    image0, image1 = load_doppleganger_img(pair, db_path, db_type, size=1024)
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    save_path = db_path/'aliked_1024_lg_matches'/db_type/f'{pair_ID}.npy'
    preview_path = db_path/'aliked_1024_lg_matches_preview'/db_type/f'{pair_ID}.png'
    save_matches(feats0, feats1, matches01, image0, image1, save_path, preview_path)


# load loftr matches
# pair_matches = np.load('./data/doppelgangers_dataset/doppelgangers/loftr_matches/train_set_noflip/0.npy', allow_pickle=True).item()

# Load the DINOv2 model
dm = Dinov2FeatureExtractor(half_precision=False, device='cuda')
ft_dino_0, grid_0 = dm.extract_features(image0)
# ft_dino_1, grid_1 = dm.extract_features(image1)
# vis1 = dm.get_embedding_visualization(ft_dino_0, grid_0)
# vis2 = dm.get_embedding_visualization(ft_dino_1, grid_1)