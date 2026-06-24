# tests/test_retail.py
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import yaml

from src.detection.loader import load_detector_config
from src.retail.visualizer import RetailVisualizer
from src.retail.zone_monitor import ZoneMonitor
from src.tracking.botsort import BotSortTracker


PERSON_CLASSES = {"person"}

DEFAULT_VIDEO_PATH = "data/shop_763.mp4"
DEFAULT_CONFIG_PATH = "configs/retail_V2.yaml"
DEFAULT_DISPLAY_SIZE = (1280, 720)
DEFAULT_PROFILE = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance le use case retail en local avec affichage OpenCV."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_SIZE[1])
    parser.add_argument("--no-profile", action="store_true")
    return parser.parse_args()


def category_name(track, class_names):
    cid = int(track.category)
    return class_names[cid] if 0 <= cid < len(class_names) else "unknown"


def print_profile(frame_id, stage_times):
    total = max(sum(stage_times.values()), 1e-9)
    parts = "  ".join(
        f"{k.split('_', 1)[1]} {1000 * v / 100:.1f}ms ({100 * v / total:.0f}%)"
        for k, v in sorted(stage_times.items())
    )
    print(f"[frame {frame_id}] {parts}")
    stage_times.clear()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    video_path = root / args.video
    config_path = root / args.config
    display_size = (args.width, args.height)
    profile = DEFAULT_PROFILE and not args.no_profile

    detector, class_names = load_detector_config(config_path, root)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tracker = BotSortTracker(**cfg["tracker"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    mon_cfg = dict(cfg["monitor"])
    zone_file = root / mon_cfg.pop("zone_file")

    monitor = ZoneMonitor(zone_file=zone_file, fps=video_fps, **mon_cfg)
    viz = RetailVisualizer(monitor)

    frame_id = 0
    fps_smooth = None
    paused = False
    stage_times = defaultdict(float)

    while True:
        if not paused:
            t_frame = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, display_size)

            t0 = time.perf_counter()
            detections = [
                d for d in detector.detect(frame)
                if d.class_name in PERSON_CLASSES
            ]
            stage_times["1_detect"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            tracks = tracker.update(detections)
            stage_times["2_track"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            persons = []

            for t in tracks:
                if category_name(t, class_names) not in PERSON_CLASSES:
                    continue

                x1, y1, x2, y2 = t.tlbr.astype(int)

                persons.append({
                    "tid": t.track_id,
                    "box": (x1, y1, x2, y2),
                    "point": ((x1 + x2) // 2, y2),
                })

            statuses = monitor.update(persons)
            stage_times["3_logic"] += time.perf_counter() - t0

            t0 = time.perf_counter()

            dt = time.perf_counter() - t_frame
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (
                inst_fps
                if fps_smooth is None
                else 0.9 * fps_smooth + 0.1 * inst_fps
            )

            viz.draw(frame, persons, statuses, fps_smooth)
            stage_times["4_draw"] += time.perf_counter() - t0

            if profile and frame_id > 0 and frame_id % 100 == 0:
                print_profile(frame_id, stage_times)

            cv2.imshow("retail_monitor", frame)
            frame_id += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
