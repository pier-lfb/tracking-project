# tools/zone_drawer.py
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO_PATH = "../data/shop_763.mp4"
DEFAULT_OUTPUT_PATH = "../configs/zone_shop_763.json"
DEFAULT_FRAME_INDEX = 0
DEFAULT_DISPLAY_SIZE = (1280, 720)


points = []
frame = None
clone = None


def redraw():
    global clone

    clone = frame.copy()

    for i, p in enumerate(points):
        cv2.circle(clone, tuple(p), 5, (0, 255, 0), -1)
        cv2.putText(
            clone,
            str(i + 1),
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if len(points) > 1:
        cv2.polylines(clone, [np.array(points)], False, (0, 255, 0), 2)

    if len(points) > 2:
        overlay = clone.copy()
        cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, clone, 0.85, 0, clone)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        redraw()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dessine une zone polygonale au sol et la sauvegarde en JSON."
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

    ret, img = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Impossible de lire la vidéo : {video_path}")

    return cv2.resize(img, display_size)


def save_zone(output_path: Path, display_size: tuple[int, int]):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "display_size": list(display_size),
                "points": points,
            },
            f,
            indent=2,
        )


def main():
    global frame, clone, points

    args = parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    display_size = (args.width, args.height)

    points = []
    frame = load_frame(video_path, args.frame_index, display_size)
    clone = frame.copy()

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)

    print("Clique les sommets de la zone au sol.")
    print("'u' : annuler le dernier point")
    print("'r' : réinitialiser")
    print("ENTRÉE : sauvegarder")

    while True:
        cv2.imshow("zone", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(points) >= 3:
            break

        if key == ord("u") and points:
            points.pop()
            redraw()

        if key == ord("r"):
            points = []
            redraw()

    cv2.destroyAllWindows()

    save_zone(output_path, display_size)
    print(f"Zone sauvegardée : {output_path} ({len(points)} sommets)")


if __name__ == "__main__":
    main()
