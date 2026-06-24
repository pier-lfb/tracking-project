# src/api/usecases.py
import json
import time
from pathlib import Path

import cv2
import yaml
import numpy as np

from src.detection.loader import load_detector_config
from src.detection.postprocess import filter_detections, cross_class_nms
from src.tracking.bytetrack import ByteTracker

ROOT = Path(__file__).resolve().parents[2]

_NOOP = lambda *a, **k: None


class BaseUseCase:
    display_name = "base"

    def __init__(self, video_path):
        self.video_path = str(video_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video introuvable : {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        self._fps_smooth = None

    def _smooth_fps(self, dt):
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps_smooth = inst if self._fps_smooth is None \
            else 0.9 * self._fps_smooth + 0.1 * inst
        return self._fps_smooth

    def process(self, frame, frame_id):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
# RETAIL
# ----------------------------------------------------------------------------------------------------------------------

class RetailUseCase(BaseUseCase):
    display_name = "Retail Use Case"

    def __init__(self, video_path):
        super().__init__(video_path)
        from src.retail.zone_monitor import ZoneMonitor
        from src.retail.visualizer import RetailVisualizer

        cfg_path = ROOT / "configs" / "retail_V2.yaml"
        self.detector, self.class_names = load_detector_config(cfg_path, ROOT)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.tracker = ByteTracker(**cfg["tracker"])
        mon_cfg = dict(cfg["monitor"])
        zone_file = ROOT / mon_cfg.pop("zone_file")
        self.monitor = ZoneMonitor(zone_file=zone_file, fps=self.fps, **mon_cfg)
        self.viz = RetailVisualizer(self.monitor)
        self.viz._panel = _NOOP

    def _cat(self, t):
        cid = int(t.category)
        return self.class_names[cid] if 0 <= cid < len(self.class_names) else "?"

    def process(self, frame, frame_id):
        t0 = time.perf_counter()
        frame = cv2.resize(frame, (1280, 720))

        detections = [d for d in self.detector.detect(frame)
                      if d.class_name == "person"]
        tracks = self.tracker.update(detections)

        persons = []
        for t in tracks:
            if self._cat(t) != "person":
                continue
            x1, y1, x2, y2 = t.tlbr.astype(int)
            persons.append({"tid": t.track_id, "box": (x1, y1, x2, y2),
                            "point": ((x1 + x2) // 2, y2)})

        statuses = self.monitor.update(persons)
        fps = self._smooth_fps(time.perf_counter() - t0)
        self.viz.draw(frame, persons, statuses, fps)

        top = self.monitor.top_dwell(5)
        in_zone_ids = {s.track_id for s in statuses.values() if s.in_zone}

        monitor = {
            "title": "RETAIL MONITOR",
            "metrics": [
                {"label": "Personnes", "value": len(persons)},
                {"label": "En zone", "value": self.monitor.count_in_zone(statuses)},
                {"label": "FPS", "value": round(fps, 1)},
            ],
            "table": {
                "title": "Top 5 Dwell Time",
                "rows": [{"id": f"ID {r['id']}",
                          "value": f"{r['seconds']:.1f} s",
                          "hot": r["id"] in in_zone_ids}
                         for r in top],
            },
        }
        stats = {"total_persons": len(persons),
                 "in_zone": self.monitor.count_in_zone(statuses),
                 "fps": round(fps, 1),
                 "top_dwell": top}
        return frame, monitor, stats


# ----------------------------------------------------------------------------------------------------------------------
# LUGGAGE
# ----------------------------------------------------------------------------------------------------------------------

class LuggageUseCase(BaseUseCase):
    display_name = "Luggage Use Case"

    def __init__(self, video_path):
        super().__init__(video_path)
        from src.luggage.luggage_monitor import LuggageMonitor, BagState
        from src.luggage.visualizer import Visualizer

        self.BagState = BagState
        self.PERSON = {"person"}
        self.LUGGAGE = {"backpack", "bag", "suitcase"}
        self.VALID = self.PERSON | self.LUGGAGE

        cfg_path = ROOT / "configs" / "luggage.yaml"
        self.detector, self.class_names = load_detector_config(cfg_path, ROOT)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(ROOT / "configs" / "homography_luggage.json", "r",
                  encoding="utf-8") as f:
            H = json.load(f)["homography"]

        self.tracker = ByteTracker(**cfg["tracker"])
        self.monitor = LuggageMonitor(homography=H, fps=self.fps, **cfg["monitor"])
        self.viz = Visualizer(self.monitor)
        self.viz._panel = _NOOP
        self.viz._alert_banner = _NOOP

    def _cat(self, t):
        cid = int(t.category)
        return self.class_names[cid] if 0 <= cid < len(self.class_names) else "?"

    def process(self, frame, frame_id):
        t0 = time.perf_counter()
        frame = cv2.resize(frame, (1280, 720))

        detections = [d for d in self.detector.detect(frame)
                      if d.class_name in self.VALID]
        tracks = self.tracker.update(detections)

        bags, persons = [], []
        for t in tracks:
            name = self._cat(t)
            if name not in self.VALID:
                continue
            x1, y1, x2, y2 = t.tlbr.astype(int)
            item = {"tid": t.track_id, "category": name,
                    "box": (x1, y1, x2, y2), "point": ((x1 + x2) // 2, y2)}
            (persons if name in self.PERSON else bags).append(item)

        statuses = self.monitor.update(bags, persons)
        fps = self._smooth_fps(time.perf_counter() - t0)
        self.viz.draw(frame, bags, persons, statuses, fps)

        abandoned = [s.track_id for s in statuses.values()
                     if s.state == self.BagState.ABANDONED]

        monitor = {
            "title": "LUGGAGE MONITOR",
            "metrics": [
                {"label": "Personnes", "value": len(persons)},
                {"label": "Bagages", "value": len(bags)},
                {"label": "Alertes", "value": len(abandoned), "alert": bool(abandoned)},
                {"label": "FPS", "value": round(fps, 1)},
            ],
            "alert": (f"BAGAGE ABANDONNE : ID {', '.join(map(str, abandoned))}"
                      if abandoned else None),
        }
        stats = {"persons": len(persons), "bags": len(bags),
                 "abandoned": abandoned, "fps": round(fps, 1)}
        return frame, monitor, stats


# ----------------------------------------------------------------------------------------------------------------------
# LUGGAGE
# ----------------------------------------------------------------------------------------------------------------------

class TrafficUseCase(BaseUseCase):
    display_name = "Trafic Use Case"

    def __init__(self, video_path):
        super().__init__(video_path)
        from src.trafic.speed_estimator import SpeedEstimator
        from src.trafic.vehicle_counter import VehicleCounter
        from src.trafic.visualizer import TraficVisualizer

        self.VEHICLES = {"Bus", "Car", "Motorcycle", "Truck"}
        cfg_path = ROOT / "configs" / "traffic.yaml"
        self.detector, self.class_names = load_detector_config(cfg_path, ROOT)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(ROOT / "configs" / "homography_traffic.json", "r",
                  encoding="utf-8") as f:
            H = np.array(json.load(f)["homography"], dtype=np.float64)

        self.tracker = ByteTracker(**cfg["tracker"])
        self.speed = SpeedEstimator(homography=H, fps=self.fps,
                                    **cfg["speed_estimator"])
        self.counter = VehicleCounter(**cfg["counter"])
        self.viz = TraficVisualizer(
            speed_limit=cfg["display"]["speed_limit"],
            trail_length=cfg["display"]["trail_length"])
        self.speed_limit = cfg["display"]["speed_limit"]
        self.viz.draw_dashboard = _NOOP

    def _cat(self, t):
        cid = int(t.category)
        return self.class_names[cid] if 0 <= cid < len(self.class_names) else "?"

    def process(self, frame, frame_id):
        t0 = time.perf_counter()
        frame = cv2.resize(frame, (1280, 720))

        detections = cross_class_nms(
            filter_detections(self.detector.detect(frame),
                              self.VEHICLES, min_box_side=12))
        tracks = self.tracker.update(detections)

        live = 0
        for t in tracks:
            if self._cat(t) not in self.VEHICLES:
                continue
            live += 1
            x1, y1, x2, y2 = t.tlbr.astype(int)
            point = ((x1 + x2) // 2, y2)
            _, _, spd = self.speed.update(t.track_id, *point, frame_id=frame_id)
            crossing = self.counter.update(t.track_id, point, frame_id=frame_id)
            if crossing:
                self.viz.notify_count(frame_id, crossing)
            self.viz.draw_track(frame, t.track_id, (x1, y1, x2, y2),
                                point, spd, frame_id)

        self.viz.draw_count_gate(frame, self.counter, frame_id)
        fps = self._smooth_fps(time.perf_counter() - t0)

        if frame_id % 30 == 0:
            self.speed.prune(frame_id)
            self.counter.prune(frame_id)
            self.viz.prune(frame_id)

        counts = self.counter.get_counts()
        monitor = {
            "title": "TRAFFIC MONITOR",
            "metrics": [
                {"label": "Live", "value": live},
                {"label": "Total", "value": counts["total"]},
                {"label": "Limite", "value": f"{self.speed_limit:.0f}"},
                {"label": "FPS", "value": round(fps, 1)},
            ],
            "table": {
                "title": "Comptage",
                "rows": [
                    {"id": "Gauche", "value": counts["left"]},
                    {"id": "Droite", "value": counts["right"]},
                ],
            },
        }
        stats = {"vehicles": live, "count": counts, "fps": round(fps, 1)}
        return frame, monitor, stats


REGISTRY = {
    "retail": RetailUseCase,
    "luggage": LuggageUseCase,
    "traffic": TrafficUseCase,
}
