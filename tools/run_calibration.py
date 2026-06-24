# tools/run_homography_calibration.py
import argparse
from pathlib import Path

import cv2

from tools.homography_calibrator import HomographyCalibrator


DEFAULT_VIDEO_PATH = "../data/shop_763.mp4"
DEFAULT_OUTPUT_PATH = "../configs/homography_retail.json"
DEFAULT_FRAME_INDEX = 0
DEFAULT_DISPLAY_SIZE = (1280, 720)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibre une homographie à partir de 4 points image et 4 points réels."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_DISPLAY_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_DISPLAY_SIZE[1])
    return parser.parse_args()


def load_frame(video_path: Path, frame_index: int, display_size: tuple[int, int]):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Impossible de lire la vidéo : {video_path}")

    return cv2.resize(frame, display_size)


def main():
    args = parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    display_size = (args.width, args.height)

    frame = load_frame(video_path, args.frame_index, display_size)

    calibrator = HomographyCalibrator()
    calibrator.pick_points(frame)

    points_world = calibrator.enter_world_coords()
    H = calibrator.compute_homography(points_world)

    calibrator.save(H, points_world, output_path)

    print(f"\nMatrice H :\n{H}")


if __name__ == "__main__":
    main()