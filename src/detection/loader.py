import yaml
from pathlib import Path

from src.detection.yolox_detector import YoloxDetector
from src.detection.yolov11_detector import YoloV11Detector

_DETECTORS = {
    "yolox": YoloxDetector,
    "yolov11": YoloV11Detector,
}


def load_detector_config(config_path, root):
    """Charge un YAML de cas d'usage -> (detector, class_names)."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    cls = _DETECTORS[m["type"]]
    detector = cls(
        model_path=str(Path(root) / m["path"]),
        class_names=cfg["classes"],
        input_size=m.get("input_size", 640),
        conf_threshold=m.get("conf_threshold", 0.25),
        iou_threshold=m.get("iou_threshold", 0.45),
    )
    return detector, cfg["classes"]
