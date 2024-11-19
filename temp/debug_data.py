#%%
import numpy as np
import pathlib as path
import kornia as K
from kornia.utils import tensor_to_image
from kornia.io import load_image
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pandas as pd


db_path = path.Path('./data/doppelgangers_dataset/doppelgangers/')
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)

#%%
image_paths = db_path/'images'/'train_set_noflip'
match_paths = db_path/'loftr_matches'/'train_set_noflip'

positive_pair = train_pairs[train_pairs[:, 2] == 1][:, :2]
pos_img_list = np.unique(np.concatenate([positive_pair[:, 0],positive_pair[:, 1]]))
negative_pair = train_pairs[train_pairs[:, 2] == 0][:, :2]
neg_img_list = np.unique(np.concatenate([negative_pair[:, 0],negative_pair[:, 1]]))

# find img not in positive pairs
toxic_img_list = np.setdiff1d(neg_img_list, pos_img_list)
# save to csv
toxic_img_list = np.unique(toxic_img_list)
df = pd.DataFrame(toxic_img_list, columns=['image'])
df.to_csv(db_path/'pairs_metadata'/'toxic_img_list.csv', index=False)

# merge positive and negative pairs
img_list = np.concatenate([train_pairs[:, 0],train_pairs[:, 1]])
img_list = np.unique(img_list)


#%%
# get location and direction
tag_list = []
for img_path in img_list:
    path = Path(img_path)
    # get dirs
    tag = path.parts[0]+'_'+path.parts[1]
    tag_list.append(tag)
    # get

# unique tags
unique_tags = np.unique(tag_list)

# export train_pairs to csv
df = pd.DataFrame(train_pairs[:,:3], columns=['image0', 'image1', 'label'])
df.to_csv(db_path/'pairs_metadata'/'train_pairs_noflip.csv', index=False)



#%%
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

img0 = load_image(image_paths/pair[0], K.io.ImageLoadType.RGB32)
img1 = load_image(image_paths/pair[1], K.io.ImageLoadType.RGB32)

# draw the images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(tensor_to_image(img0))
ax[1].imshow(tensor_to_image(img1))
plt.show()