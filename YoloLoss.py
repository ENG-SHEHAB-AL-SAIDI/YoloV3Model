import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, numAnchors, numClasses, gridSize, ignoreThreshold=0.5):
        super(YOLOLoss, self).__init__()
        self.numAnchors = numAnchors
        self.numClasses = numClasses
        self.gridSize = gridSize
        self.ignoreThreshold = ignoreThreshold

    def forward(self, preds, targets):
        totalLoss = 0
        batchSize = preds[0].size(0)

        for pred in preds:
            # Reshape the prediction to separate anchor boxes
            pred = pred.view(batchSize, self.numAnchors, -1, self.gridSize,
                             self.gridSize)  # B, A, (5 + num_classes), H, W

            # Extract bounding box predictions
            bbox_pred = pred[:, :, :4, :, :]  # Bounding boxes: [center_x, center_y, width, height]
            obj_pred = pred[:, :, 4:5, :, :]  # Objectness score
            class_pred = pred[:, :, 5:, :, :]  # Class probabilities

            # Initialize targets
            bbox_targets = torch.zeros_like(bbox_pred)
            obj_targets = torch.zeros_like(obj_pred)
            class_targets = torch.zeros(batchSize, self.numAnchors, self.gridSize, self.gridSize, dtype=torch.long,
                                        device=pred.device)

            # Assign targets to corresponding grid cells
            for b in range(batchSize):
                for t in range(len(targets[b][0])):
                    if len(targets[b][0].shape) == 2:  # Check if targets[b][0] is 2D
                        gx, gy = targets[b][0][t, :2] * self.gridSize  # Ground truth center in grid space
                        gw, gh = targets[b][0][t, 2:]  # Width and height (already normalized)
                        gi, gj = int(gx), int(gy)  # Grid cell indices

                        # Set the target box for the responsible anchor
                        bbox_targets[b, :, 0, gj, gi] = gx - gi  # x offset within the cell
                        bbox_targets[b, :, 1, gj, gi] = gy - gj  # y offset within the cell
                        bbox_targets[b, :, 2, gj, gi] = gw  # width
                        bbox_targets[b, :, 3, gj, gi] = gh  # height

                        # Set objectness target
                        obj_targets[b, :, 0, gj, gi] = 1  # Positive objectness for this anchor

                        # Set class target
                        class_targets[b, :, gj, gi] = targets[b][2][t]
                    else:
                        raise ValueError(
                            "Expected target tensor to have 2 dimensions, but got {}.".format(targets[b][0].shape))

            # Compute losses
            bbox_loss = F.mse_loss(bbox_pred, bbox_targets, reduction='sum')
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_targets, reduction='sum')
            class_loss = F.cross_entropy(class_pred.view(-1, self.numClasses), class_targets.view(-1), reduction='sum')

            totalLoss += bbox_loss + obj_loss + class_loss

        return totalLoss / batchSize
