import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compute_precision_recall(predictions, targets, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_boxes, pred_scores, pred_classes in predictions:
        gt_boxes, gt_classes = targets
        matched_gt = set()

        for i, (pred_box, pred_score, pred_class) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_class == pred_class and j not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

def compute_ap(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    true_labels = []
    pred_labels = []
    scores = []

    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        true_labels.append(gt_class)
        matched = False

        for pred_box, pred_score, pred_class in zip(pred_boxes, pred_scores, pred_classes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_threshold and pred_class == gt_class:
                pred_labels.append(gt_class)
                scores.append(pred_score)
                matched = True
                break

        if not matched:
            pred_labels.append(-1)  # -1 for no match

    precision, recall, _ = precision_recall_curve(true_labels, scores)
    ap = average_precision_score(true_labels, scores)

    return ap


def compute_mAP(predictions, targets, numClasses, iou_threshold=0.5):
    aps = []
    for cls in range(numClasses):
        cls_preds = [p for p in predictions if p[2] == cls]
        cls_targets = [t for t in targets if t[1] == cls]

        pred_boxes = [p[0] for p in cls_preds]
        pred_scores = [p[1] for p in cls_preds]
        pred_classes = [p[2] for p in cls_preds]

        gt_boxes = [t[0] for t in cls_targets]
        gt_classes = [t[1] for t in cls_targets]

        ap = compute_ap(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold)
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP
