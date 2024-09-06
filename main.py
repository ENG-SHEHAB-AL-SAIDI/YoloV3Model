import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from DataSet import YoloDataset
from Utils import loadModelState, check_class_accuracy, get_evaluation_bboxes, mean_average_precision
from YoloLoss import YoloLoss
from YoloModel import YOLOv3
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    batchSize = 32
    dropLast = False
    pinMemory = True

    trainDataset = YoloDataset(
        'DataSet/train',
        'DataSet/train',
        anchors=[
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ],
        transform=test_transforms,
    )
    trainLoader = DataLoader(trainDataset, shuffle=True, num_workers=numWorkers, batch_size=batchSize,
                             drop_last=dropLast, pin_memory=pinMemory)

    # testDataset = YoloDataset(
    #     'DataSet/test',
    #     'DataSet/test',
    #     s=[imageSize // 32, imageSize // 16, imageSize // 8],
    #     anchors=[
    #         [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    #         [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    #         [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ],
    #     transform=test_transforms,
    # )
    # testLoader = DataLoader(testDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
    #                          drop_last=dropLast, pin_memory=pinMemory)

    ###########################################################################
    #                            Model Setup                                  #
    ###########################################################################

    # Initialize model, loss function, and optimizer
    lr = 1e-5,
    model = YOLOv3(numClasses=1, numAnchors=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    lossFunc = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    # loadModelState("ModelStatus/modelState.pth.tar",
    #                model=model, optimizer=optimizer, lr=lr, device=device)

    ###########################################################################
    #                            Model training                               #
    ###########################################################################
    # Training loop
    scaledAnchors = (
            torch.tensor([
                [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ])
            * torch.tensor([imageSize // 32, imageSize // 16, imageSize // 8])
            .unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    numEpochs = 10
    for epoch in range(numEpochs):
        model.train()
        loop = tqdm(trainLoader, total=len(trainLoader), leave=True)
        losses = []
        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device),)
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = (
                        lossFunc(out[0], y0, scaledAnchors[0])
                        + lossFunc(out[1], y1, scaledAnchors[1])
                        + lossFunc(out[2], y2, scaledAnchors[2])
                )
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

        # if epoch > 0 and epoch % 3 == 0:
        #     check_class_accuracy(model, trainLoader, threshold=0.05)
        #     pred_boxes, true_boxes = get_evaluation_bboxes(
        #         trainLoader,
        #         model,
        #         iou_threshold=0.45,
        #         anchors=[
        #             [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        #             [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        #             [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        #         ],
        #         threshold=0.05,
        #     )
        #     mapval = mean_average_precision(
        #         pred_boxes,
        #         true_boxes,
        #         iou_threshold=0.5,
        #         box_format="midpoint",
        #         num_classes=1,
        #     )
        #     print(f"MAP: {mapval.item()}")
        #     model.train()




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
