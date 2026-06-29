# tests/test_soccer.py
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch  # noqa: F401 - helps load CUDA DLLs before ONNX Runtime on Windows.
import yaml

from src.detection.loader import load_detector_config
from src.soccer.possession import PossessionTracker
from src.soccer.team_assigner import TeamAssigner
from src.soccer.team_tracker import TeamTracker
from src.soccer.visualizer import FootballVisualizer
from src.tracking.botsort import BotSortTracker

PLAYER_CLASSES = {"Player"}
GOALKEEPER_CLASSES = {"Goalkeeper"}
BALL_CLASSES = {"Ball"}

DEFAULT_VIDEO_PATH = "data/soccer.mp4"
DEFAULT_CONFIG_PATH = "configs/soccer.yaml"
DEFAULT_DISPLAY_SIZE = (1280, 720)
DEFAULT_PROFILE = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance le use case football en local avec affichage OpenCV."
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
        f"{key.split('_', 1)[1]} {1000 * value / 100:.1f}ms ({100 * value / total:.0f}%)"
        for key, value in sorted(stage_times.items())
    )
    print(f"[frame {frame_id}] {parts}")
    stage_times.clear()


def build_team_assigner(cfg):
    team_cfg = cfg["team"]
    return TeamAssigner(
        min_samples=team_cfg.get("min_samples", 150),
        max_samples=team_cfg.get("max_samples", 400),
        num_teams=team_cfg.get("num_teams", 2),
        min_margin=team_cfg.get("min_margin", 12.0),
    )


def build_team_tracker(cfg):
    team_cfg = cfg["team"]
    return TeamTracker(
        min_votes=team_cfg.get("min_votes", 5),
        stale_frames=team_cfg.get("stale_frames", 300),
        num_teams=team_cfg.get("num_teams", 2),
        vote_window=team_cfg.get("vote_window", 30),
    )


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
    team_assigner = build_team_assigner(cfg)
    team_tracker = build_team_tracker(cfg)
    possession = PossessionTracker(**cfg["possession"])
    viz = FootballVisualizer(class_names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

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

            ball_detections = [d for d in detections if d.class_name in BALL_CLASSES]

            t0 = time.perf_counter()
            tracks = tracker.update(detections, curr_img=frame)
            stage_times["2_track"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            player_tracks = [
                t for t in tracks
                if category_name(t, class_names) in PLAYER_CLASSES
            ]
            gk_tracks = [
                t for t in tracks
                if category_name(t, class_names) in GOALKEEPER_CLASSES
            ]

            if not team_assigner.is_fitted:
                team_assigner.collect(frame, [t.tlbr for t in player_tracks])
            else:
                team_tracker.update_players(player_tracks, frame, team_assigner, frame_id)
                team_tracker.update_goalkeepers(gk_tracks, player_tracks, frame_id)

            holder_id, holder_team = possession.update(
                player_tracks + gk_tracks,
                ball_detections,
                team_tracker,
                frame_id,
            )
            stats = possession.get_possession_stats()
            stage_times["3_logic"] += time.perf_counter() - t0

            if frame_id % 60 == 0:
                team_tracker.prune(frame_id)

            t0 = time.perf_counter()
            dt = time.perf_counter() - t_frame
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = (
                inst_fps
                if fps_smooth is None
                else 0.9 * fps_smooth + 0.1 * inst_fps
            )

            viz.draw(
                frame=frame,
                tracks=tracks,
                ball_detections=ball_detections,
                team_tracker=team_tracker,
                holder_id=holder_id,
                holder_team=holder_team,
                stats=stats,
                fps=fps_smooth,
            )
            stage_times["4_draw"] += time.perf_counter() - t0

            if profile and frame_id > 0 and frame_id % 100 == 0:
                print_profile(frame_id, stage_times)

            cv2.imshow("football_monitor", frame)
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
