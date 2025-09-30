"""
30/9/25
Visualize weights of CNN models trained with different blur-levels.
"""

import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import os

model_blur = '0'
# ------------------------------
# 1. Load trained model
# ------------------------------
checkpoint_dir = '/home/projects/bagon/ilanaveh/code/examples_imagenet/imagenet/out/New'
cp_pth = os.path.join(checkpoint_dir, f'train_resnet_blur{model_blur}', 'model_best.pth.tar')
model = models.resnet101()  # initialize base architecture
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

plt.figure(figsize=(num_cols, num_rows))

for i in range(num_filters):
    # Normalize filter for visualization
    w = weights[i]
    w_min, w_max = w.min(), w.max()
    # w = (w - w_min) / (w_max - w_min)  # scale to [0,1]

    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(w.permute(1, 2, 0).cpu())  # [C,H,W] → [H,W,C]
    plt.axis("off")

plt.suptitle("ResNet-101 First Layer Filters", fontsize=16)
plt.show()
