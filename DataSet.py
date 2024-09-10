import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from Utils import iou_width_height
import xml.etree.ElementTree as ET

def load_annotations(annot_path):
    boxes = []
    tree = ET.parse(annot_path)
    root = tree.getroot()
    imWidth = int(root.find('size').find('width').text)
    imHeight = int(root.find('size').find('height').text)
    for objectElem in root.findall('object'):
        className = objectElem.find('name').text  # Get the class name
        classId = 1 if className=="LP" else 0
        bndbox = objectElem.find('bndbox')

        # Get bounding box coordinates
        xmin = int(bndbox.find('xmin').text)/imWidth
        ymin = int(bndbox.find('ymin').text)/imHeight
        xmax = int(bndbox.find('xmax').text)/imWidth
        ymax = int(bndbox.find('ymax').text)/imHeight

        width = xmax - xmin
        height = ymax - ymin
        xCenter = xmin + width / 2
        yCenter = ymin + height / 2

        boxes.append([xCenter, yCenter, width, height, classId])

        # Use normalized bounding box coordinates
            # boxes.append([x_center, y_center, width, height, class_id])

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
        annot_path = os.path.join(self.annotations_dir, img_file.replace('.jpg', '.xml'))

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



# Testing
# imageSize = 416
# scale = 1.1
# anchors = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ]
#
# s = [imageSize // 32, imageSize // 16, imageSize // 8] # 52 , 26 , 13
# numWorkers = 1
# batchSize = 4
# dropLast = False
# pinMemory = True
#
# trainDataset = YoloDataset(
#     'DataSet/train',
#     'DataSet/train',
#     s=s,
#     anchors=anchors,
#     transform=None,
# ).__getitem__(0)
# trainLoader = DataLoader(trainDataset, shuffle=True, num_workers=numWorkers, batch_size=batchSize,
#                          drop_last=dropLast, pin_memory=pinMemory)