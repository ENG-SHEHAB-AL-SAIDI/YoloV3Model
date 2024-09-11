import torch
import numpy as np

def extractBoundingBoxes(output, anchors, confidenceThreshold=0.5, nmsThreshold=0.4, numClasses=80, inputDim=416):
    # Assume output is of shape (batch_size, grid_size, grid_size, num_anchors * (5 + num_classes))
    batchSize, gridSize, _, numAttrs = output.shape
    numAnchors = len(anchors)  # Number of anchor boxes

    # Reshape output to (batch_size, grid_size * grid_size * num_anchors, 5 + num_classes)
    output = output.view(batchSize, gridSize * gridSize * numAnchors, 5 + numClasses)

    # Extract components
    bx = torch.sigmoid(output[..., 0])  # x center
    by = torch.sigmoid(output[..., 1])  # y center
    bw = output[..., 2]  # width
    bh = output[..., 3]  # height
    confidence = torch.sigmoid(output[..., 4])  # objectness score
    class_probs = torch.sigmoid(output[..., 5:])  # class probabilities

    # Convert grid coordinates to actual image coordinates
    gridX, gridY = torch.meshgrid(torch.arange(gridSize), torch.arange(gridSize))
    gridX = gridX.view(-1, 1).to(output.device)
    gridY = gridY.view(-1, 1).to(output.device)

    bx = (bx + gridX) / gridSize * inputDim
    by = (by + gridY) / gridSize * inputDim

    # Adjust width and height
    bw = torch.exp(bw) * torch.tensor(anchors)[:, 0].view(1, -1) / inputDim
    bh = torch.exp(bh) * torch.tensor(anchors)[:, 1].view(1, -1) / inputDim

    # Combine everything into final bounding boxes
    boxes = torch.cat((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2), dim=-1)

    # Filter boxes by confidence threshold
    mask = confidence > confidenceThreshold
    filtered_boxes = boxes[mask]
    filtered_confidences = confidence[mask]
    filtered_class_probs = class_probs[mask]

    # Apply Non-Maximum Suppression (NMS)
    final_boxes = []
    for class_index in range(numClasses):
        class_scores = filtered_confidences * filtered_class_probs[..., class_index]
        keep = nms(filtered_boxes, class_scores, nmsThreshold)
        final_boxes.extend(filtered_boxes[keep])

    return final_boxes

def nms(boxes, scores, iouThreshold):
    # Implement Non-Maximum Suppression (NMS)
    keep = []
    indices = scores.argsort(descending=True)
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        ious = iou(boxes[current], boxes[indices[1:]])
        indices = indices[1:][ious < iouThreshold]
    return keep

def iou(box1, boxes):
    # Compute Intersection over Union (IoU)
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])
    interArea = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxAreas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unionArea = box1Area + boxAreas - interArea
    return interArea / unionArea
