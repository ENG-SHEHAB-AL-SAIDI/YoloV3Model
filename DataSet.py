import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class YOLODataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        annot_path = os.path.join(self.annotations_dir, img_file.replace('.jpg', '.txt'))

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Load and normalize annotations (already normalized)
        boxes = self.load_annotations(annot_path)

        # Apply transformations
        if self.transform:
            image, boxes = self.transform(image, boxes)


        # Convert list of boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        bbox_targets = boxes[:, :4]  # Bounding box coordinates
        class_targets = boxes[:, 4].long()  # Class labels

        # Objectiveness targets: Assuming 1 for simplicity, adjust based on your actual requirements
        obj_targets = torch.ones(len(boxes), dtype=torch.float32)

        return image, (bbox_targets, obj_targets, class_targets)

    def load_annotations(self, annot_path):
        boxes = []
        with open(annot_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Use normalized bounding box coordinates
                boxes.append([x_center, y_center, width, height, class_id])

        return boxes
