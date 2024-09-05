import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from Utils import iou_width_height


def load_annotations(annot_path):
    boxes = []
    with open(annot_path) as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Use normalized bounding box coordinates
            boxes.append([x_center, y_center, width, height, class_id])

    return boxes


class YoloDataset(Dataset):
    def __init__(self,
                 images_dir,
                 annotations_dir,
                 anchors,
                 c=1,
                 s=None,
                 transform=None):

        if s is None:
            self.s = [13, 26, 52]
        else:
            self.s = s
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.numAnchors = self.anchors.shape[0]
        self.numAnchorsPerScale = self.numAnchors // 3
        self.c = c
        self.ignoreIoUThreshold = 0.5

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        annot_path = os.path.join(self.annotations_dir, img_file.replace('.jpg', '.txt'))

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        # Load and normalize annotations (already normalized)
        boxes = load_annotations(annot_path)

        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        targets = [torch.zeros(self.numAnchors // 3, S, S, 6) for S in self.s]
        # Convert list of boxes to tensor
        for box in boxes:
            iouAnchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchorIndices = iouAnchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchorIndices:
                scale_idx = anchor_idx // self.numAnchorsPerScale
                anchor_on_scale = anchor_idx % self.numAnchorsPerScale
                S = self.s[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iouAnchors[anchor_idx] > self.ignoreIoUThreshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
