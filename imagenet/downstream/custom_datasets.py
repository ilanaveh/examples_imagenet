from torchvision.datasets.folder import ImageFolder, default_loader
import os
import numpy as np
from collections import Counter


class AffectnetDataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, des_classes=[], balance_clss=False, debug=False):
        self.des_classes = des_classes  # desired classes (list of integeres, between 0-10)

        # Number of images in each AffectNet category (in train_set):
        self.count_class_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3750, 8: 33088,
                                 9: 82415, 10: 419799}
        self.balance_clss = balance_clss  # whether to balance number of images from each class.
        self.class_to_idx = {c: i for (i, c) in enumerate(des_classes)}

        # if debugging, build small dataset:
        self.limit_dataset_size = 100 if debug else None

        super().__init__(root, loader=default_loader, transform=transform, target_transform=target_transform)

    def make_dataset(self, directory, class_to_idx=None, extensions=None, is_valid_file=None, allow_empty=False):
        """
        Override the make_dataset method, since Affectnet is not organized in sub-folders corresponding to classes.
        Original method (used by ImageFolder): /usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py
        """
        # Copy the first part from original make_dataset method:
        directory = os.path.expanduser(directory)
        images_path = os.path.join(directory, 'images')
        ann_path = os.path.join(directory, 'annotations')

        # If balance classes, get number of images required for each class:
        if self.balance_clss:
            cnt_ims_lst = [self.count_class_dict[c] for c in self.des_classes]
            des_ims_each_clss = np.min(cnt_ims_lst)

        # Initialize dictionary for keeping track of how many images from each class:
        if self.balance_clss or self.limit_dataset_size:
            track_num_ims_each_clss = {c: 0 for c in self.des_classes}
            target_num = self.limit_dataset_size if self.limit_dataset_size else des_ims_each_clss

        # Initialize list of image paths and labels
        images = []

        # Walk through images and annotations
        print_done = 0
        for (i, img) in enumerate(os.listdir(images_path)):
            im_id = img.split('.')[0]  # remove '.jpg'
            if img.lower().endswith(extensions):
                ann = int(np.load(os.path.join(ann_path, (im_id + '_exp.npy'))))
                if ann in self.des_classes:
                    if (not self.balance_clss and not self.limit_dataset_size) \
                            or (track_num_ims_each_clss[ann] < target_num):
                        path = os.path.join(images_path, img)
                        images.append((path, self.class_to_idx[ann]))
                        if self.balance_clss or self.limit_dataset_size:
                            track_num_ims_each_clss[ann] += 1
                            if np.all([v >= target_num for v in track_num_ims_each_clss.values()]):
                                for ann, v in track_num_ims_each_clss.items():
                                    print(f"Number of images from class {ann}: {v}")
                                    print_done = 1
                                break
        if not print_done:
            cnt_anns = Counter([im[1] for im in images])
            for ann, v in sorted(cnt_anns.items()):
                print(f"Number of images from class {ann}: {v}")

        return images
