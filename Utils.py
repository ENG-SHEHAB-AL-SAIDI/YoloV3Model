from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from collections import Counter

from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as eT


def loadAnnotations(annotPath, annotFormat=".txt"):
    boxes = []
    if annotFormat == ".xml":
        tree = eT.parse(annotPath)
        root = tree.getroot()
        imWidth = int(root.find('size').find('width').text)
        imHeight = int(root.find('size').find('height').text)
        for objectElem in root.findall('object'):
            className = objectElem.find('name').text  # Get the class name
            classId = 0 if className == 'LP' else -1
            bndBox = objectElem.find('bndbox')

            # Get bounding box coordinates
            xmin = float(bndBox.find('xmin').text) / imWidth
            ymin = float(bndBox.find('ymin').text) / imHeight
            xmax = float(bndBox.find('xmax').text) / imWidth
            ymax = float(bndBox.find('ymax').text) / imHeight

            width = xmax - xmin
            height = ymax - ymin
            xCenter = xmin + width / 2
            yCenter = ymin + height / 2
            boxes.append([xCenter, yCenter, width, height, classId])

        # Use normalized bounding box coordinates
        # boxes.append([x_center, y_center, width, height, class_id])

    elif annotFormat == ".txt":
        with open(annotPath) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Use normalized bounding box coordinates
                boxes.append([x_center, y_center, width, height, class_id])

    return np.array(boxes)


def saveAnnotations(outputPath, imageName, boxes, imageWidth, imageHeight, annotationsFormat='.txt'):
    if annotationsFormat == '.txt':
        annotPath = os.path.join(outputPath, f"{imageName}.txt")
        with open(annotPath, 'w') as f:
            for box in boxes:
                # Ensure each box is a list/tuple with class label first followed by bounding box coordinates
                classLabel = int(box[0])  # Assuming the class is the first element in the box
                bboxCoords = box[1:]  # The rest are the bounding box coordinates
                # Write the class and the bounding box coordinates to the file
                f.write(f"{classLabel} " + " ".join(map(str, bboxCoords)) + "\n")

    elif annotationsFormat == '.xml':
        annotation = eT.Element("annotation")

        # Create folder, filename, and path elements
        folder = eT.SubElement(annotation, "folder").text = outputPath
        filename = eT.SubElement(annotation, "filename").text = f"{imageName}.jpg"
        path = eT.SubElement(annotation, "path").text = os.path.join(outputPath, f"{imageName}.jpg")

        # Create size element
        size = eT.SubElement(annotation, "size")
        eT.SubElement(size, "width").text = str(imageWidth)
        eT.SubElement(size, "height").text = str(imageHeight)
        eT.SubElement(size, "depth").text = "3"

        # Create object elements for each bounding box
        for box in boxes:
            x_center, y_center, width, height, class_id = box
            xmin = int((x_center - width / 2) * imageWidth)
            ymin = int((y_center - height / 2) * imageHeight)
            xmax = int((x_center + width / 2) * imageWidth)
            ymax = int((y_center + height / 2) * imageHeight)

            obj = eT.SubElement(annotation, "object")
            eT.SubElement(obj, "name").text = str(class_id)
            eT.SubElement(obj, "pose").text = "Unspecified"
            eT.SubElement(obj, "truncated").text = "0"
            eT.SubElement(obj, "difficult").text = "0"
            bbox = eT.SubElement(obj, "bndbox")
            eT.SubElement(bbox, "xmin").text = str(xmin)
            eT.SubElement(bbox, "ymin").text = str(ymin)
            eT.SubElement(bbox, "xmax").text = str(xmax)
            eT.SubElement(bbox, "ymax").text = str(ymax)

        # Save XML file
        xmlStr = minidom.parseString(eT.tostring(annotation)).toprettyxml(indent="   ")
        annotPath = os.path.join(outputPath, f"{imageName}.xml")
        with open(annotPath, 'w') as f:
            f.write(xmlStr)


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-8)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_evaluation_bboxes(
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format="midpoint",
        device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)
        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def check_class_accuracy(model, loader, threshold, device):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(device)
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noObj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noObj] == y[i][..., 0][noObj])
            tot_noobj += torch.sum(noObj)

    print(f"correct_class is: {correct_class}")
    print(f"tot_class_preds is: {tot_class_preds}")
    print(f"No obj  is: {correct_noobj}")
    print(f"tot_noobj  is: {tot_noobj}")
    print(f"Obj  is: {correct_obj}")
    print(f"tot_obj  is: {tot_obj}")

    model.train()
    return [(correct_class / (tot_class_preds + 1e-16)) * 100, (correct_noobj / (tot_noobj + 1e-16)) * 100,
            (correct_obj / (tot_obj + 1e-16)) * 100]


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def saveModelState(model, optimizer, filePath="my_checkpoint.pth.tar", CreateLastModelState=True):
    if filePath == "": return
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    print(f"=> Saving ModelState to {filePath}")
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, filePath)
    if CreateLastModelState:
        lastModelStatePath = os.path.join(os.path.dirname(filePath), f"lastModelState.txt")
        with open(lastModelStatePath, 'w') as f:
            f.write(f"{filePath}")
    print("saving model state successful")


def loadModelState(modelStateFilePath, model, optimizer, lr, device, loadLastModelState=False):
    if loadLastModelState and os.path.isdir(modelStateFilePath):
        if not os.path.exists(os.path.dirname(modelStateFilePath)):
            print(f"{modelStateFilePath} doesn't exist")
            return
        lastModelStatePath = os.path.join(modelStateFilePath, "lastModelState.txt")
        if not os.path.exists(os.path.exists(lastModelStatePath)):
            print(f"{lastModelStatePath} doesn't exist")
            return
        with open(lastModelStatePath) as f:
            for line in f:
                modelStateFilePath = line
    elif not loadLastModelState and os.path.isdir(modelStateFilePath):
        print("you provide Directory path with loadLastModelState=False ")
        print("please provide modelState.pth.tar or set loadLastModelState=True")
        return
    elif loadLastModelState and not os.path.isdir(modelStateFilePath):
        print("you set loadLastModelState=True ")
        print("please provide Directory path with loadLastModelState and  modelState.pth.tar")
        return

    if not os.path.exists(modelStateFilePath):
        print(f"{modelStateFilePath} doesn't exist")
        return
    elif modelStateFilePath == "":
        return

    print(f"=> Loading {modelStateFilePath}")
    state = torch.load(modelStateFilePath, map_location=device)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Loading model state successful")


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plotImage(image, boxes, savePath="", trueBboxes=None, index=0, isPred=True):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = ["LP"]
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    if isPred:

        for box in boxes:
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"

            class_pred = box[0]
            obj_score = box[1]
            box = box[2:]
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2

            upper_right_x = upper_left_x + box[2]
            upper_right_y = upper_left_y

            if trueBboxes:
                iou = intersection_over_union(torch.tensor(box), torch.tensor(trueBboxes[0][3:]))
                true_upper_left_x = trueBboxes[0][3] - trueBboxes[0][5] / 2
                true_upper_left_y = trueBboxes[0][4] - trueBboxes[0][6] / 2

                trueRect = patches.Rectangle(
                    (true_upper_left_x * width, true_upper_left_y * height),
                    trueBboxes[0][5] * width,
                    trueBboxes[0][6] * height,
                    linewidth=2,
                    edgecolor='green',
                    facecolor="none",
                )
                ax.add_patch(trueRect)
                plt.text(
                    upper_right_x * width,
                    (upper_right_y+box[3]) * height,
                    s=f"{iou.item():.4f}",
                    color="white",
                    verticalalignment="top",
                    bbox={"color": colors[int(class_pred)], "pad": 0},
                )

            predRect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=2,
                edgecolor=colors[int(class_pred)],
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(predRect)

            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )
            plt.text(
                upper_right_x * width,
                upper_right_y * height,
                s=f"{obj_score:.4f}",
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )
    else:
        for box in boxes:
            assert len(box) == 5, "box should contain x, y, width, height, class"
            class_pred = box[4]
            box = box[:4]
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=2,
                edgecolor=colors[int(class_pred)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )

    if savePath != "":
        plt.savefig(os.path.join(savePath, f'figure_{index}.png'))
    plt.show()


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors, device, savePath=""):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        temp_true_bboxes = cells_to_bboxes(y[2], anchor, S=S, is_preds=False)
        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        true_bboxes = []
        for box in temp_true_bboxes[i]:
            if box[1] > 0.5:
                true_bboxes.append([0] + box)

        plotImage(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes, trueBboxes=true_bboxes, savePath=savePath, index=i)


def predictImageBbox(model, imagePath, transformer, anchors, iou_thresh, thresh, device, index=20):
    image = np.array(Image.open(imagePath).convert('RGB'))
    trans = transformer(image=image)
    image = trans["image"]
    image = np.reshape(image, (1,) + image.shape)
    model.eval()
    image = image.to(device)
    output = model(image)
    bboxes = [[] for _ in range(image.shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(
            output[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    print(bboxes[0])
    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
    )
    plotImage(image[0].permute(1, 2, 0).detach().cpu(), nms_boxes, savePath="./", index=index)
    return nms_boxes
