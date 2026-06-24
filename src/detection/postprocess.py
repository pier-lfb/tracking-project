import numpy as np


def filter_detections(detections, allowed_classes, min_box_side=0, max_detections=None):
    dets = [
        d for d in detections
        if d.class_name in allowed_classes
        and (d.xyxy[2] - d.xyxy[0]) >= min_box_side
        and (d.xyxy[3] - d.xyxy[1]) >= min_box_side
    ]
    dets.sort(key=lambda d: d.score, reverse=True)
    if max_detections is not None:
        dets = dets[:max_detections]
    return dets


def cross_class_nms(detections, iou_thresh=0.7):
    if len(detections) <= 1:
        return detections

    boxes = np.array([d.xyxy for d in detections], dtype=np.float64)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    keep = []
    suppressed = np.zeros(len(detections), dtype=bool)
    for i in range(len(detections)):
        if suppressed[i]:
            continue
        keep.append(detections[i])
        x1 = np.maximum(boxes[i, 0], boxes[i + 1:, 0])
        y1 = np.maximum(boxes[i, 1], boxes[i + 1:, 1])
        x2 = np.minimum(boxes[i, 2], boxes[i + 1:, 2])
        y2 = np.minimum(boxes[i, 3], boxes[i + 1:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[i + 1:] - inter, 1e-9)
        suppressed[i + 1:] |= iou > iou_thresh
    return keep
