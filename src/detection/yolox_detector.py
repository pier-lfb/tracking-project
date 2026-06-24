from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.detection.basedet import BaseDetector, Detection


class YoloxDetector(BaseDetector):
    def __init__(self, model_path, class_names, input_size=640, conf_threshold=0.35, iou_threshold=0.45, p6=False, providers=None):
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.p6 = p6
        self.num_features = 5 + len(class_names)
        self._grids = None
        self._strides = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        input_tensor, scale = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], scale, frame.shape[:2])

    def _preprocess(self, frame):
        original_h, original_w = frame.shape[:2]
        scale = min(self.input_size / original_w, self.input_size / original_h)

        resized_w = int(original_w * scale)
        resized_h = int(original_h * scale)
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized

        image = canvas.transpose(2, 0, 1).astype(np.float32)
        return np.ascontiguousarray(image[np.newaxis, ...]), scale

    def _postprocess(self, output, scale, original_shape):
        predictions = np.squeeze(output)

        if predictions.shape[0] == self.num_features:
            predictions = predictions.T
        elif predictions.shape[1] != self.num_features:
            raise ValueError(
                f"Sortie ONNX {output.shape} incompatible avec "
                f"{len(self.class_names)} classes (attendu {self.num_features} features)."
            )

        predictions = self._decode(predictions)

        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(len(class_scores)), class_ids]

        valid = scores >= self.conf_threshold
        boxes_xywh = predictions[:, :4][valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy /= scale

        original_h, original_w = original_shape
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_h - 1)

        keep = self._nms_per_class(boxes_xyxy, scores, class_ids)
        return [
            Detection(
                xyxy=boxes_xyxy[i].astype(np.float32),
                score=float(scores[i]),
                class_id=int(class_ids[i]),
                class_name=self.class_names[int(class_ids[i])],
            )
            for i in keep
        ]

    def _decode(self, predictions):
        if self._grids is None:
            strides = [8, 16, 32, 64] if self.p6 else [8, 16, 32]
            grids, expanded = [], []
            for stride in strides:
                size = self.input_size // stride
                xv, yv = np.meshgrid(np.arange(size), np.arange(size))
                grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
                expanded.append(np.full((size * size, 1), stride))
            self._grids = np.concatenate(grids).astype(np.float32)
            self._strides = np.concatenate(expanded).astype(np.float32)

        predictions[:, :2] = (predictions[:, :2] + self._grids) * self._strides
        predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * self._strides
        return predictions

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh):
        xy, half_wh = boxes_xywh[:, :2], boxes_xywh[:, 2:4] / 2
        return np.concatenate([xy - half_wh, xy + half_wh], axis=1)

    def _nms_per_class(self, boxes, scores, class_ids):
        if len(boxes) == 0:
            return []
        offset = (boxes.max() + 1) * class_ids[:, None]
        return self._nms(boxes + offset, scores)

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
            order = rest[iou <= self.iou_threshold]
        return keep