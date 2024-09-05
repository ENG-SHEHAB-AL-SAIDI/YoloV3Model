import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.ops import nms


def postProcess(preds, numClasses, confThreshold=0.5, nmsThreshold=0.4):
    all_boxes = []
    for pred in preds:
        pred = pred.sigmoid()  # Apply sigmoid to get probabilities

        # Reshape predictions: (B, A*(5+C), H, W) -> (B, A, 5+C, H, W)
        pred = pred.view(pred.size(0), -1, pred.size(2), pred.size(3))

        # Extract components
        obj_conf = pred[:, :, 4, :, :]  # Objectness confidence
        bbox_pred = pred[:, :, :4, :, :]  # Bounding boxes
        class_pred = pred[:, :, 5:, :, :]  # Class probabilities

        # Apply threshold
        mask = obj_conf > confThreshold
        bbox_pred = bbox_pred[mask]
        class_pred = class_pred[mask]
        obj_conf = obj_conf[mask]

        # Convert bbox predictions to [x_center, y_center, width, height]
        boxes = bbox_pred.new_zeros(bbox_pred.size())
        boxes[..., 0] = bbox_pred[..., 0]  # x_center
        boxes[..., 1] = bbox_pred[..., 1]  # y_center
        boxes[..., 2] = bbox_pred[..., 2]  # width
        boxes[..., 3] = bbox_pred[..., 3]  # height

        # Apply NMS (Non-Maximum Suppression)
        final_boxes = []
        final_scores = []
        final_classes = []

        for i in range(len(boxes)):
            scores = obj_conf[i] * class_pred[i]
            for cls in range(numClasses):
                scores_cls = scores[cls]
                indices = scores_cls > confThreshold
                boxes_cls = boxes[i][:, indices]
                scores_cls = scores_cls[indices]
                final_boxes_cls = []
                final_scores_cls = []
                keep = nms(boxes_cls, scores_cls, nmsThreshold)
                final_boxes_cls.append(boxes_cls[keep])
                final_scores_cls.append(scores_cls[keep])
                final_classes.extend([cls] * len(keep))

                if len(final_boxes_cls) > 0:
                    final_boxes.append(torch.cat(final_boxes_cls, dim=0))
                    final_scores.append(torch.cat(final_scores_cls, dim=0))

        all_boxes.append((torch.cat(final_boxes, dim=0), torch.cat(final_scores, dim=0), final_classes))

    return all_boxes

