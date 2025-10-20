"""
30/09/25
Based on Pawan PNAS 2018, code for fitting elipses in conv1 weights, and computing RF size accordingly.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.spatial.distance import cdist
import os


model_blur = '0'
ker_size = 22  # size of conv1 kernel (original is 7)


# ------------------------------
# 2. Ellipse fitting via PCA
# ------------------------------
def fit_ellipse(mask):
    """Fit ellipse via PCA on contour points."""
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if contour.shape[0] < 5:
        return None

    y, x = contour[:, 0], contour[:, 1]
    coords = np.column_stack((x, y))

    # PCA
    cov = np.cov(coords.T)
    evals, evecs = np.linalg.eig(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    center = coords.mean(axis=0)
    axes = 2 * np.sqrt(evals)   # lengths of axes
    angle = np.degrees(np.arctan2(*evecs[:,0][::-1]))

    return (center, axes, angle, coords)


def extreme_points(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) < 2:
        return None
    dists = cdist(coords, coords)
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    return coords[i], coords[j]


# ------------------------------
# 3. Analysis function
# ------------------------------
def analyze_rf(kernel, thresh_factor=0.3):
    k = kernel / np.max(np.abs(kernel))
    pos_mask = (k > thresh_factor).astype(np.uint8)
    neg_mask = (k < -thresh_factor).astype(np.uint8)

    # Morphological cleanup
    pos_mask = morphology.binary_opening(pos_mask)
    neg_mask = morphology.binary_opening(neg_mask)

    ellipses = {}
    rf_metric = None

    for label, mask in zip(["pos", "neg"], [pos_mask, neg_mask]):
        e = fit_ellipse(mask)
        if e is not None:
            ellipses[label] = e

    if "pos" in ellipses and "neg" in ellipses:
        # extreme points
        pos_pts = extreme_points(pos_mask)
        neg_pts = extreme_points(neg_mask)
        if pos_pts is not None and neg_pts is not None:
            mid_neg = np.mean(neg_pts, axis=0)
            p1, p2 = pos_pts
            v = p2 - p1
            v = v / np.linalg.norm(v)
            proj = p1 + v * np.dot(mid_neg - p1, v)
            rf_metric = np.linalg.norm(mid_neg - proj)

    return rf_metric, ellipses


def filter_bw(weights, thresh=.4):
    w_filt = []
    for w in weights:
        k = 0
        t = 0

        w_min, w_max = w.min(), w.max()
        w_norm = (w - w_min) / (w_max - w_min)

        for x in range(ker_size):
            for y in range(ker_size):
                cur_w = w_norm[:, y, x]
                if any(abs(np.append(np.diff(cur_w), cur_w[0] - cur_w[-1])) > thresh):
                    k += 1
                t += 1
        if k == 0:
            w_filt.append(w)

    return torch.stack(w_filt).numpy()


def main():
    # ------------------------------
    # 1. Load trained model
    # ------------------------------
    checkpoint_dir = '/home/projects/bagon/ilanaveh/code/examples_imagenet/imagenet/out/New'
    model_name = f'train_resnet_blur{model_blur}_ker{ker_size}' if (ker_size != 7) else f'train_resnet_blur{model_blur}'
    cp_pth = os.path.join(checkpoint_dir, model_name, 'model_best.pth.tar')
    model = models.resnet101()  # initialize base architecture

    if ker_size != 7:
        ori_conv1_ker_size = model.conv1.weight.shape[-1]
        print("=> Changing conv1 kernel size from: {}, to: {}".format(ori_conv1_ker_size, ker_size))
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(ker_size, ker_size),
                                stride=(2, 2), padding=int((ker_size - 1) / 2), bias=False)

    model = nn.DataParallel(model)
    checkpoint = torch.load(cp_pth, map_location="cpu")

    model.load_state_dict(checkpoint['state_dict'])  # load trained weights
    model = model.module
    model.eval()

    weights = model.conv1.weight.data.cpu()  # [64, 3, ker_size, ker_size] ker_size=7 for original model.
    weights = filter_bw(weights)
    rf_maps = weights.mean(axis=1)  # [64, ker_size, ker_size]

    # ------------------------------
    # 2. Compute metrics
    # ------------------------------
    metrics, ellipses_all = [], []
    for rf in rf_maps:
        m, e = analyze_rf(rf)
        metrics.append(m)
        ellipses_all.append(e)

    # ------------------------------
    # 3. Keep only kernels where both ellipses were successfully fitted
    # ------------------------------
    valid_idx = [i for i, e in enumerate(ellipses_all) if "pos" in e and "neg" in e]

    print(f"{len(valid_idx)} of {len(rf_maps)} kernels had valid ellipse fits.")

    # Option A: simply visualize all valid ones
    selected = valid_idx

    # Option B: rank the valid ones by std if you still want top-N
    topN = 15
    stds = rf_maps.reshape(len(rf_maps), -1).std(axis=1)
    selected = sorted(valid_idx, key=lambda i: stds[i], reverse=True)[:topN]

    # ------------------------------
    # 4. Plot kernels with ellipses
    # ------------------------------
    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    plt.figure(figsize=(cols*3, rows*3))

    for i, idx in enumerate(selected):
        rf = rf_maps[idx]
        m = metrics[idx]
        ellipses = ellipses_all[idx]

        plt.subplot(rows, cols, i+1)
        plt.imshow(rf, cmap="bwr", vmin=-np.max(np.abs(rf)), vmax=np.max(np.abs(rf)))

        if "pos" in ellipses:
            center, axes, angle, _ = ellipses["pos"]
            ellipse = plt.matplotlib.patches.Ellipse(center, *axes, angle=angle,
                                                     edgecolor="green", facecolor="none", lw=1)
            plt.gca().add_patch(ellipse)
        if "neg" in ellipses:
            center, axes, angle, _ = ellipses["neg"]
            ellipse = plt.matplotlib.patches.Ellipse(center, *axes, angle=angle,
                                                     edgecolor="red", facecolor="none", lw=1)
            plt.gca().add_patch(ellipse)

        title = f"K{idx} RF={m:.2f}" if m else f"K{idx} no fit"
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.suptitle(f"Top-{topN} kernels with ellipse fits", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()