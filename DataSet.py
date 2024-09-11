import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from Utils import iou_width_height, loadAnnotations

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YoloDataset(Dataset):
    def __init__(self,
                 imagesDir,
                 annotationsDir,
                 anchors,
                 annotationsFormat = '.txt',
                 c=1,
                 s=None,
                 transform=None):

        if s is None:
            self.s = [13, 26, 52]
        else:
            self.s = s
        self.images_dir = imagesDir
        self.annotationsDir = annotationsDir
        self.annotationsFormat = annotationsFormat
        self.transform = transform
        self.imageFiles = [f for f in os.listdir(imagesDir) if f.endswith('.jpg')]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.numAnchors = self.anchors.shape[0]
        self.numAnchorsPerScale = self.numAnchors // 3
        self.c = c
        self.ignoreIoUThreshold = 0.5

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, idx):
        img_file = self.imageFiles[idx]
        img_path = os.path.join(self.images_dir, img_file)
        annot_path = os.path.join(self.annotationsDir, img_file.replace('.jpg',self.annotationsFormat))

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        # Load and normalize annotations (already normalized)
        boxes = loadAnnotations(annot_path, annotFormat=self.annotationsFormat)


        # Apply transformations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]


        targets = [torch.zeros(self.numAnchors // self.numAnchorsPerScale, S, S, 6) for S in self.s]
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


#
# def main():
#     # Testing
#     imageSize = 416
#     scale = 1.1
#     anchors = [
#         [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#         [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#         [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ]
#
#     s = [imageSize // 32, imageSize // 16, imageSize // 8]  # 52 , 26 , 13
#     numWorkers = 1
#     batchSize = 2
#     dropLast = False
#     pinMemory = True
#
#     train_transforms = A.Compose(
#         [
#             A.LongestMaxSize(max_size=int(imageSize * scale)),
#             A.PadIfNeeded(
#                 min_height=int(imageSize * scale),
#                 min_width=int(imageSize * scale),
#                 border_mode=cv2.BORDER_CONSTANT,
#                 value=[0, 255, 0]
#             ),
#             A.RandomCrop(width=imageSize, height=imageSize),
#             A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
#             A.OneOf(
#                 [
#                     A.ShiftScaleRotate(
#                         rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
#                     ),
#                     A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT, ),
#                 ],
#                 p=1.0,
#             ),
#             A.HorizontalFlip(p=0.5),
#             A.Blur(p=0.1),
#             A.CLAHE(p=0.1),
#             A.Posterize(p=0.1),
#             A.ToGray(p=0.1),
#             A.ChannelShuffle(p=0.05),
#             A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
#             ToTensorV2(),
#         ],
#         bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
#     )
#
#
#
#
#     trainDataset = YoloDataset(
#         'DataSet/train',
#         'DataSet/train',
#         annotationsFormat='.xml',
#         s=s,
#         anchors=anchors,
#         transform=None,
#     )
#     # trainLoader = DataLoader(trainDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
#     #                          drop_last=dropLast, pin_memory=pinMemory)
#     # indices = list(range(len(trainDataset)))
#     # subData = SubsetRandomSampler(indices[:5])
#     # trainLoader = DataLoader(trainDataset, batch_size=batchSize, sampler=subData)
#     #
#     # anchor = torch.tensor([*anchors[2]]) * 52
#     # for x, y in trainLoader:
#
#     # Load image
#     image = np.array(Image.open("DataSet/test/0.jpg").convert('RGB'))
#     # Load and normalize annotations (already normalized)
#     boxes = load_annotations("DataSet/test/0.xml", format='xml')
#     print(image.shape)
#     print(boxes[:,:4])
#     plt.figure()
#     plt.imshow(image)
#     plt.title('Sample RGB Image from NumPy Array')
#     plt.show()
#     aug = train_transforms(image=image, bboxes=boxes)
#
#     print(aug["image"].shape)
#     plt.figure()
#     plt.imshow(aug["image"])
#     plt.title('Sample RGB Image from NumPy Array')
#     plt.show()
#     print("-----------------------------")
#
#
#
#     # bbox = cells_to_bboxes(trainDataset[1][0],anchor,13,is_preds= False)
#
#
# if __name__ == "__main__":
#     main()
