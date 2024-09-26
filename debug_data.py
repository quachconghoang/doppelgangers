import numpy as np
import pathlib as path
import kornia as K
from kornia.utils import tensor_to_image
from kornia.io import load_image
import cv2 as cv
import matplotlib.pyplot as plt
import torch

db_path = path.Path('./data/doppelgangers_dataset/doppelgangers/')
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)

image_paths = db_path/'images'/'train_set_noflip'
match_paths = db_path/'loftr_matches'/'train_set_noflip'

positive_pair = train_pairs[train_pairs[:, 2] == 1]
# negative_pair = train_pairs[train_pairs[:, 2] == 0]

c_ID = 1001
pair = train_pairs[c_ID]
matches = np.load(match_paths/f'{c_ID}.npy', allow_pickle=True).item()

keypoints0 = np.array(matches['kpt0'])
keypoints1 = np.array(matches['kpt1'])
conf = matches['conf']
F, mask = cv.findFundamentalMat(keypoints0[conf > 0.8], keypoints1[conf > 0.8], cv.FM_RANSAC, 3, 0.99)
mtc = np.array(np.ones((keypoints0.shape[0], 2)) * np.arange(keypoints0.shape[0]).reshape(-1, 1)).astype(int)[conf > 0.8][
    mask.ravel() == 1]

from doppelgangers.utils.dataset import read_loftr_matches

image = read_loftr_matches(image_paths/pair[0], image_paths/pair[1], 640, 8, True,
                           keypoints0, keypoints1, mtc, warp=True,
                           conf=conf)
img0 = tensor_to_image(image[0])
img1 = tensor_to_image(image[1])
img2 = tensor_to_image(image[2])
img3 = tensor_to_image(image[3])
img4 = tensor_to_image(image[4:7])
img5 = tensor_to_image(image[7:10])

# valid_source = (((train_pairs[:, 0] == pair[0]) | (train_pairs[:, 1] == pair[0])) &
#                 (train_pairs[:, 2] == 0) & (train_pairs[:, 3] > 100))
# valid_source[c_ID] = False
# check0 = np.where(valid_source)
#
# valid_target = (((train_pairs[:, 0] == pair[1]) | (train_pairs[:, 1] == pair[1])) &
#                 (train_pairs[:, 2] == 0) & (train_pairs[:, 3] > 100))
# valid_target[c_ID] = False
# check1 = np.where(valid_target)

img0 = load_image(image_paths/pair[0], K.io.ImageLoadType.RGB32)
img1 = load_image(image_paths/pair[1], K.io.ImageLoadType.RGB32)

# draw the images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(tensor_to_image(img0))
ax[1].imshow(tensor_to_image(img1))
plt.show()

# for id in check0:
#     p = train_pairs[id]
#     img0 = load_image(image_paths / p[0], K.io.ImageLoadType.RGB32)
#     img1 = load_image(image_paths / p[1], K.io.ImageLoadType.RGB32)
#     # draw the images
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].imshow(tensor_to_image(img0))
#     ax[1].imshow(tensor_to_image(img1))
#     plt.show()