"""
30/9/25
Visualize weights of CNN models trained with different blur-levels.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import os

save_fig = True

model_blur = '0'
ker_size = 22  # size of conv1 kernel; None for original (7)
# ------------------------------
# 1. Load trained model
# ------------------------------
checkpoint_dir = '/home/projects/bagon/ilanaveh/code/examples_imagenet/imagenet/out/New'
model_name = f'train_resnet_blur{model_blur}_ker{ker_size}' if ker_size else f'train_resnet_blur{model_blur}'
cp_pth = os.path.join(checkpoint_dir, model_name, 'model_best.pth.tar')
model = models.resnet101()  # initialize base architecture

if ker_size is not None:
    ori_conv1_ker_size = model.conv1.weight.shape[-1]
    print("=> Changing conv1 kernel size from: {}, to: {}".format(ori_conv1_ker_size, ker_size))
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(ker_size, ker_size),
                            stride=(2, 2), padding=int((ker_size - 1) / 2), bias=False)

model = nn.DataParallel(model)
checkpoint = torch.load(cp_pth, map_location="cpu")

model.load_state_dict(checkpoint['state_dict'])  # load trained weights
model = model.module
model.eval()

# ------------------------------
# 2. Extract the first convolutional layer
# ------------------------------
first_layer = model.conv1  # first conv layer in ResNet101
weights = first_layer.weight.data.clone()

print("First layer weight shape:", weights.shape)
# Shape: [64, 3, 7, 7] → (num_filters, in_channels, kernel_h, kernel_w)

# ------------------------------
# 3. Visualize filters
# ------------------------------
num_filters = weights.shape[0]
num_cols = 8
num_rows = num_filters // num_cols

f = plt.figure(figsize=(num_cols, num_rows))

for i in range(num_filters):
    # Normalize filter for visualization
    w = weights[i]
    w_min, w_max = w.min(), w.max()
    w = (w - w_min) / (w_max - w_min)  # scale to [0,1]
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(w.permute(1, 2, 0).cpu())  # [C,H,W] → [H,W,C]
    plt.axis("off")

plt.suptitle(f"{model_name} First Layer Filters", fontsize=16)

if save_fig:
    f.savefig(os.path.join('from_weight_vis', model_name))
plt.show()
