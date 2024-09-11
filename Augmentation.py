import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageFile
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from matplotlib.pyplot import figure
from Utils import loadAnnotations, plotImage, saveAnnotations

ImageFile.LOAD_TRUNCATED_IMAGES = True
imageSize = 416
scale = 1.1

augTransforms = A.Compose(
        [
            # Ensure crop size is valid
            A.SmallestMaxSize(max_size=imageSize),
            A.CenterCrop(height=imageSize, width=imageSize),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.RandomSizedCrop(min_max_height=(imageSize//2, imageSize), height=imageSize, width=imageSize, p=0.3),  # Randomly crop part of the image
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),  # Add contrast and brightness variations
                A.RandomGamma(p=0.3)  # Randomly adjust gamma
            ], p=0.3),
            A.OneOf([
                A.CoarseDropout(p=0.3),  # Randomly drop parts of the image
                A.GridDistortion(p=0.3)  # Distort the grid of the image
            ], p=0.3)
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], check_each_transform=True),
    )



def augmentAndSaveData(imagesDir, annotationsDir, imagesOutputDir=None, annotationsOutputDir=None, numAug = 5, annotationsFormat=".txt", transforms=augTransforms):

    if imagesOutputDir is None:
        imagesOutputDir = imagesDir
    elif not os.path.exists(imagesOutputDir):
        os.makedirs(imagesOutputDir, exist_ok=True)

    if annotationsOutputDir is None:
        annotationsOutputDir = annotationsDir
    elif not os.path.exists(annotationsOutputDir):
        os.makedirs(annotationsOutputDir, exist_ok=True)


    imageFiles = [f for f in os.listdir(imagesDir) if f.endswith('.jpg')]

    for imgFile in imageFiles:
        imgPath = os.path.join(imagesDir, imgFile)
        annotPath = os.path.join(annotationsDir, imgFile.replace('.jpg', annotationsFormat))

        # Load image
        image = np.array(Image.open(imgPath).convert('RGB'))
        # Load and normalize annotations
        boxes = loadAnnotations(annotPath, annotFormat=annotationsFormat)

        # Apply transformations
        for i in range(numAug):
            augmentations = transforms(image=image, bboxes=boxes)
            cv2.imwrite(os.path.join(imagesOutputDir, imgFile.replace('.jpg','')+f'-with-aug-{i}.jpg'), augmentations["image"],)
            saveAnnotations(annotationsOutputDir,imgFile.replace('.jpg','')+f'-with-aug-{i}', augmentations["bboxes"], augmentations["image"].shape[1],augmentations["image"].shape[0],annotationsFormat=annotationsFormat)


#   Test
# augmentAndSaveData(imagesDir="DataSet/train", annotationsDir="DataSet/train", imagesOutputDir="DataSet/augTrain", annotationsOutputDir="DataSet/augTrain", annotationsFormat=".xml")