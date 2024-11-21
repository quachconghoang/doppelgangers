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

from doppelgangers.utils.dataset import imread_rgb
import tqdm

import os
import numpy as np

#%%
db_path = path.Path('./data/doppelgangers_dataset/doppelgangers/')
test_pairs = np.load(db_path/'pairs_metadata'/'test_pairs.npy', allow_pickle=True)
train_pairs = np.load(db_path/'pairs_metadata'/'train_pairs_noflip.npy', allow_pickle=True)

#%%
# get all locations
locations = []
for pair in train_pairs:
    locations.append(pair[0].split('/')[0])

# count unique locations
unique_locations, loc_counts = np.unique(locations, return_counts=True)
unique_locations = np.concatenate([unique_locations.reshape(-1, 1), loc_counts.reshape(-1, 1)], axis=1)


#%%
#
valid_locations = ['Arc_de_Triomphe_du_Carrousel_by_angle',
                   '%C3%89glise_de_la_Madeleine_2',
                   'Ch%C3%A2teau_de_Cheverny',
                   'Cour_Napol%C3%A9on',
                   'Da_Lat_Station',
                   'Entrance_to_Aleppo_citadel_(view_from',
                   'Liberty_Square_(Taipei)_2',
                   'Mainz_Cathedral',
                   'the_Ch%C3%A2teau_de_Chambord',
                   'the_Ch%C3%A2teau_de_Sceaux',
                   'the_Grands_Guichets_du_Louvre',
                   'Torre_de_Bel%C3%A9m']
valid_ids = []
for i, pair in enumerate(train_pairs):
    if pair[0].split('/')[0] in valid_locations:
        valid_ids.append(i)

#%%
# create dir
os.makedirs(db_path/'loftr_matches_easy'/'train_set_noflip', exist_ok=True)
for id,old_id in enumerate(valid_ids):
    old_npy_path = db_path/'loftr_matches'/'train_set_noflip'/f'{old_id}.npy'
    new_npy_path = db_path/'loftr_matches_easy'/'train_set_noflip'/f'{id}.npy'
    print(old_npy_path,'--->', new_npy_path)
    # copy file
    os.system(f'cp {old_npy_path} {new_npy_path}')

# new train_pairs
train_pairs_easy = train_pairs[valid_ids]
# save new train_pairs
np.save(db_path/'pairs_metadata'/'easy_train_pairs_noflip.npy', train_pairs_easy)

#%%
valid_locations = ['Arc_de_Triomphe_de_l%27%C3%89toile_by_angle',
                        'Brandenburg_Gate',
                        'Exterior_of_Charlottenburg_Palace',
                        'Washington_Square_Arch'
                        ]
valid_ids = []
for i, pair in enumerate(test_pairs):
    if pair[0].split('/')[0] in valid_locations:
        valid_ids.append(i)

#%%
os.makedirs(db_path/'loftr_matches_easy'/'test_set', exist_ok=True)
for id,old_id in enumerate(valid_ids):
    old_npy_path = db_path/'loftr_matches'/'test_set'/f'{old_id}.npy'
    new_npy_path = db_path/'loftr_matches_easy'/'test_set'/f'{id}.npy'
    print(old_npy_path,'--->', new_npy_path)
    # copy file
    os.system(f'cp {old_npy_path} {new_npy_path}')

test_pairs_easy = test_pairs[valid_ids]
np.save(db_path/'pairs_metadata'/'easy_test_pairs.npy', test_pairs_easy)
