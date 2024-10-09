import numpy as np
from trainers.utils.utils import compute_ap
from pathlib import Path
import cv2 as cv


test_pairs = np.load('./data/doppelgangers_dataset/doppelgangers/pairs_metadata/test_pairs.npy', allow_pickle=True)
logs = np.load('./val_logs/doppelgangers_classifier_noflip_val_2024-Oct-06-10-40-46/test_doppelgangers_list.npy', allow_pickle=True).item()
gt_list = logs['gt']
pred_list = logs['pred']
prob_list = logs['prob']
# compute precision
precision = np.sum(gt_list == pred_list) / len(gt_list) # Precision = 0.88, AP = 0.95???
# ap = compute_ap(gt_list, prob_list)


failed = logs['pred'] != logs['gt']
# get idx of failed predictions
failed_idx = np.where(failed)[0]
# failed_pairs = test_pairs[failed_idx]

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new

img_size = 1024
img_dir = Path('./data/doppelgangers_dataset/doppelgangers/images/test_set')

for pair_ID in failed_idx:
    print(pair_ID)
    pair = test_pairs[pair_ID]
    img0 = cv.imread(str(img_dir / pair[0]), cv.IMREAD_COLOR)
    img1 = cv.imread(str(img_dir / pair[1]), cv.IMREAD_COLOR)
    w, h = img0.shape[1], img0.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    img0 = cv.resize(img0, (w_new, h_new))

    w1, h1 = img1.shape[1], img1.shape[0]
    w_new = round((w1/h1) * h_new)
    img1 = cv.resize(img1, (w_new, h_new))
    # Visualize failed pairs
    img_viz = np.concatenate([img0, img1], axis=1)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img_viz, 'ID: %d' % pair_ID, (10, 50), font, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(img_viz, 'GT: %d' % gt_list[pair_ID], (10, 100), font, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imwrite(f'./val_logs/{pair_ID}.jpg', img_viz)