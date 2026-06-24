# tests/test_trafic.py
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.detection.loader import load_detector_config
from src.detection.postprocess import cross_class_nms, filter_detections
from src.tracking.bytetrack import ByteTracker
from src.trafic.speed_estimator import SpeedEstimator
from src.trafic.vehicle_counter import VehicleCounter
from src.trafic.visualizer import TraficVisualizer


VEHICLE_CLASSES = {"Bus", "Car", "Motorcycle", "Truck"}

DEFAULT_VIDEO_PATH = "data/traffic.mp4"
DEFAULT_CONFIG_PATH = "configs/traffic.yaml"
DEFAULT_HOMOGRAPHY_PATH = "configs/homography_traffic.json"
DEFAULT_DISPLAY_SIZE = (1280, 720)
DEFAULT_PROFILE = True
DEFAULT_OUTPUT_VIDEO = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance le use case trafic en local avec affichage OpenCV."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--homography", default=DEFAULT_HOMOGRAPHY_PATH)
    parser.add_argument("--output-video", default=DEFAULT_OUTPUT_VIDEO)
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
    homography_path = root / args.homography
    display_size = (args.width, args.height)
    profile = DEFAULT_PROFILE and not args.no_profile

    detector, class_names = load_detector_config(config_path, root)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(homography_path, "r", encoding="utf-8") as f:
        homography = np.array(json.load(f)["homography"], dtype=np.float64)

    tracker = ByteTracker(**cfg["tracker"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    speed_estimator = SpeedEstimator(
        homography=homography,
        fps=video_fps,
        **cfg["speed_estimator"],
    )
    vehicle_counter = VehicleCounter(**cfg["counter"])

    viz = TraficVisualizer(
        speed_limit=cfg["display"]["speed_limit"],
        trail_length=cfg["display"]["trail_length"],
    )

    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output_video,
            fourcc,
            video_fps,
            display_size,
        )

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
            detections = detector.detect(frame)
            stage_times["1_detect"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            detections = cross_class_nms(
                filter_detections(
                    detections,
                    VEHICLE_CLASSES,
                    min_box_side=12,
                )
            )
            stage_times["2_postproc"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            tracks = tracker.update(detections)
            stage_times["3_track"] += time.perf_counter() - t0

            t0 = time.perf_counter()

            for t in tracks:
                if category_name(t, class_names) not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = t.tlbr.astype(int)
                tid = t.track_id
                point = ((x1 + x2) // 2, y2)

                _, _, speed_kmh = speed_estimator.update(
                    tid,
                    *point,
                    frame_id=frame_id,
                )

                crossing = vehicle_counter.update(
                    tid,
                    point,
                    frame_id=frame_id,
                )

                if crossing:
                    viz.notify_count(frame_id, crossing)

                viz.draw_track(
                    frame,
                    tid,
                    (x1, y1, x2, y2),
                    point,
                    speed_kmh,
                    frame_id,
                )

            stage_times["4_logic"] += time.perf_counter() - t0

            t0 = time.perf_counter()

            viz.draw_count_gate(frame, vehicle_counter, frame_id)

            dt = time.perf_counter() - t_frame
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (
                inst_fps
                if fps_smooth is None
                else 0.9 * fps_smooth + 0.1 * inst_fps
            )

            viz.draw_dashboard(
                frame,
                fps_smooth,
                len(tracks),
                vehicle_counter.get_counts(),
            )
            stage_times["5_draw"] += time.perf_counter() - t0

            if frame_id % 30 == 0:
                speed_estimator.prune(frame_id)
                vehicle_counter.prune(frame_id)
                viz.prune(frame_id)

            if profile and frame_id > 0 and frame_id % 100 == 0:
                print_profile(frame_id, stage_times)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("traffic", frame)
            frame_id += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            paused = not paused

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
