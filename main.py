import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from datetime import datetime
from DataSet import YoloDataset
from Utils import loadModelState, check_class_accuracy, get_evaluation_bboxes, mean_average_precision, \
    plot_couple_examples, saveModelState
from YoloLoss import YoloLoss
from YoloModel import YOLOv3
from tqdm import tqdm
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch.cuda.empty_cache()

annotationsFormat = '.txt'
imageSize = 416
scale = 1.1
anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], ]
s = [imageSize // 32, imageSize // 16, imageSize // 8]  # 52 , 26 , 13

###########################################################################
#                           Apply Augmentation                            #
###########################################################################
# print("Apply Data Augmentation ....")
# augmentAndSaveData(imagesDir="DataSet/train", annotationsDir="DataSet/train", imagesOutputDir="DataSet/train-aug",
#                    annotationsOutputDir="DataSet/train-aug", annotationsFormat=annotationsFormat,
#                    numAug=3)
# generateNoObjectSamples("DataSet/train", "DataSet/train", "DataSet/train-noObj", "DataSet/train-noObj", 1000)

###########################################################################
#                           Define data transforms                        #
###########################################################################

transforms = A.Compose(
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
    annotationsFormat=annotationsFormat,
    s=s,
    anchors=anchors,
    transform=transforms,
)
trainLoader = DataLoader(trainDataset, shuffle=True, num_workers=numWorkers, batch_size=batchSize,
                         drop_last=dropLast, pin_memory=pinMemory)


validationDataset = YoloDataset(
    'DataSet/test',
    'DataSet/test',
    annotationsFormat=annotationsFormat,
    s=s,
    anchors=anchors,
    transform=transforms,
)
validationLoader = DataLoader(validationDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
                        drop_last=dropLast, pin_memory=pinMemory)


testDataset = YoloDataset(
    'DataSet/test',
    'DataSet/test',
    annotationsFormat=annotationsFormat,
    s=s,
    anchors=anchors,
    transform=transforms,
)
testLoader = DataLoader(testDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
                        drop_last=dropLast, pin_memory=pinMemory)


###########################################################################
#                            Model Setup                                  #
###########################################################################
def main():
    # Initialize model, loss function, and optimizer
    learnRate = 1e-5
    model = YOLOv3(numClasses=1, numAnchors=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnRate, weight_decay=1e-4)
    lossFunc = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    scaledAnchors = (torch.tensor(anchors) * torch.tensor(s).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)).to(device)
    loadModelState("ModelStatus/",
                   model=model, optimizer=optimizer, lr=learnRate, device=device, loadLastModelState=True)
    ###########################################################################
    #                            Model training                               #
    ###########################################################################
    numEpochs = 10
    # indices = list(range(len(trainDataset)))
    # subData = SubsetRandomSampler(indices[:10])
    # trainLoader = DataLoader(trainDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
    #                             drop_last=dropLast, pin_memory=pinMemory, sampler=subData)
    #
    # testLoader = DataLoader(testDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
    #                            drop_last=dropLast, pin_memory=pinMemory, sampler=subData)

    for epoch in range(numEpochs):
        model.train()
        loop = tqdm(trainLoader, leave=True, desc=Fore.LIGHTWHITE_EX + f'Epoch {epoch + 1}/{numEpochs}')
        losses = []
        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device),)

            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = (
                            lossFunc(out[0], y0, scaledAnchors[0])
                            + lossFunc(out[1], y1, scaledAnchors[1])
                            + lossFunc(out[2], y2, scaledAnchors[2])
                    )
            else:
                out = model(x)
                loss = (
                        lossFunc(out[0], y0, scaledAnchors[0])
                        + lossFunc(out[1], y1, scaledAnchors[1])
                        + lossFunc(out[2], y2, scaledAnchors[2])
                )

            losses.append(loss.item())
            optimizer.zero_grad()
            if device == "cuda":
                scaler.scale(loss).backward()  # Use scaler only with CUDA
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

        ###########################################################################
        #                            Model Validation                             #
        ###########################################################################
        if epoch > 0 and epoch % 5 == 0:
            saveModelState(model, optimizer, f"ModelStatus/checkpoint{datetime.now().strftime('%I-%M-%S-%p')}.pth.tar")
            print("Evaluate Model")
            model.eval()
            with torch.no_grad():
                acc = check_class_accuracy(model, validationLoader, threshold=0.05, device=device)
                print(f"Class accuracy is: {acc[0]:2f}%", end=' | ')
                print(f"No obj accuracy is: {acc[1]:2f}%", end=' | ')
                print(f"Obj accuracy is: {acc[2]:2f}%")

                pred_boxes, true_boxes = get_evaluation_bboxes(
                    validationLoader,
                    model,
                    iou_threshold=0.45,
                    anchors=anchors,
                    threshold=0.5,
                    device=device
                )

                print("cal map")
                mapval = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=0.5,
                    box_format="midpoint",
                    num_classes=1,

                )
                print(f"MAP: {mapval.item()}")

            ###########################################################################
            #                            Model result show                            #
            ###########################################################################

            indices = list(range(len(trainDataset)))
            subData = SubsetRandomSampler(indices[:10])
            subTestLoader = DataLoader(testDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
                                   drop_last=dropLast, pin_memory=pinMemory, sampler=subData)

            subTestLoader = DataLoader(testDataset, shuffle=False, num_workers=numWorkers, batch_size=batchSize,
                                drop_last=dropLast, pin_memory=pinMemory,sampler=subData)

            plot_couple_examples(model=model, loader=subTestLoader, iou_thresh=0.45, anchors=anchors, thresh=0.5, )

    ###########################################################################
    #                            Model Validation                             #
    ###########################################################################

    model.eval()
    with torch.no_grad():
        acc = check_class_accuracy(model, testLoader, threshold=0.05, device=device)
        print(f"Class accuracy is: {acc[0]:2f}%", end=' | ')
        print(f"No obj accuracy is: {acc[1]:2f}%", end=' | ')
        print(f"Obj accuracy is: {acc[2]:2f}%")

        pred_boxes, true_boxes = get_evaluation_bboxes(
            testLoader,
            model,
            iou_threshold=0.45,
            anchors=anchors,
            threshold=0.5,
            device=device
        )
        print("cal map")
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=1,
        )
        print(f"MAP: {mapval.item()}")

if __name__ == "__main__":
    main()
