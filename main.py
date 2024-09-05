from torch.optim import Adam
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from DataSet import YoloDataset
from YoloLoss import YoloLoss
from YoloModel import YOLOv3


def main():
    imageSize = 416
    scale = 1.1

    ###########################################################################
    #                           Define data transforms                        #
    ###########################################################################
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(imageSize * scale)),
            A.PadIfNeeded(
                min_height=int(imageSize * scale),
                min_width=int(imageSize * scale),
                border_mode=cv2.BORDER_CONSTANT,
                value=[0, 255, 0]
            ),
            A.RandomCrop(width=imageSize, height=imageSize),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT, ),
                ],
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
    )

    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=imageSize),
            A.PadIfNeeded(
                min_height=imageSize, min_width=imageSize, border_mode=cv2.BORDER_CONSTANT, value=[0, 255, 0]
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    ###########################################################################
    #                            Loading Data                                 #
    ###########################################################################
    numWorkers = 4
    batchSize = 4
    dropLast = False
    pinMemory = True
    train_dataset = YoloDataset(
        'DataSet/train',
        'DataSet/train',
        anchors=[
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ],
        transform=train_transforms,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=numWorkers, batch_size=batchSize,
                              drop_last=dropLast, pin_memory=pinMemory)

    test_dataset = YoloDataset(
        'DataSet/test',
        'DataSet/test',
        s=[imageSize // 32, imageSize // 16, imageSize // 8],
        anchors=[
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ],
        transform=test_transforms,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
                             drop_last=dropLast, pin_memory=pinMemory)

    ###########################################################################
    #                            Model Setup                                  #
    ###########################################################################

    # Initialize model, loss function, and optimizer
    model = YOLOv3(numClasses=1, numAnchors=3)
    criterion = YoloLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    ###########################################################################
    #                            Model training                               #
    ###########################################################################
    # Training loop
    numEpochs = 1
    for epoch in range(numEpochs):
        model.train()
        runningLoss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            # loss.backward()
            # optimizer.step()
            runningLoss += loss.item()

        print(f'Epoch {epoch+1}/{numEpochs}, Loss: {runningLoss/len(train_loader)}')

    # # Test model
    # model.eval()
    # with torch.no_grad():
    #     for images, targets in test_loader:
    #         outputs = model(images)
    #         # Post-processing would be done here, such as applying NMS
    #         # Example: print output shapes
    #         for out in outputs:
    #             print(out.shape)


if __name__ == "__main__":
    main()
