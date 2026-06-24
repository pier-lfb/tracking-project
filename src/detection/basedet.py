from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray
    score: float
    class_id: int
    class_name: str


class BaseDetector:
    def detect(self, frame):
        raise NotImplementedError