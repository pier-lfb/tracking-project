# tools/gen_heatmap.py
import argparse
from pathlib import Path

import torch  # à importer avant onnxruntime pour éviter certains conflits CUDA
import cv2
import yaml
import numpy as np

from tqdm import tqdm

from src.detection.loader import load_detector_config
from src.detection.postprocess import filter_detections, cross_class_nms
from src.tracking.bytetrack import ByteTracker


ROOT = Path(__file__).resolve().parents[1]
SIZE = (1280, 720)


USECASES = {
    "retail": {
        "config": "configs/retail_V2.yaml",
        "video": "data/shop_763.mp4",
        "classes": {"person"},
        "nms": False,
    },
    "luggage": {
        "config": "configs/luggage.yaml",
        "video": "data/avss2007_luggage.mp4",
        "classes": {"person", "backpack", "bag", "suitcase"},
        "nms": False,
    },
    "traffic": {
        "config": "configs/traffic.yaml",
        "video": "data/traffic.mp4",
        "classes": {"Bus", "Car", "Motorcycle", "Truck"},
        "nms": True,
    },
}


def render_heatmap(background, accum, blur, alpha):
    """Superpose la heatmap courante sur une image de fond."""
    if accum.max() == 0:
        return background.copy()

    k = blur if blur % 2 == 1 else blur + 1

    heat = cv2.GaussianBlur(accum, (k, k), 0)
    heat = np.log1p(heat)
    heat = heat / heat.max()

    heat_u8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    mask = (heat_u8 > 8).astype(np.float32)[..., None]

    overlay = (
        background.astype(np.float32) * (1 - alpha * mask)
        + heat_color.astype(np.float32) * (alpha * mask)
    )

    return overlay.astype(np.uint8)


def category_name(track, class_names):
    cid = int(track.category)
    return class_names[cid] if 0 <= cid < len(class_names) else "unknown"


def build_heatmap(
    usecase,
    video_path=None,
    out_path=None,
    make_gif=False,
    out_gif=None,
    blur=51,
    alpha=0.6,
    max_frames=None,
    gif_frames=60,
    gif_fps=12,
    freeze_seconds=2.0,
):
    if usecase not in USECASES:
        raise ValueError(
            f"Use case inconnu : {usecase} "
            f"(disponibles : {', '.join(USECASES)})"
        )

    meta = USECASES[usecase]

    config_path = ROOT / meta["config"]
    video = Path(video_path) if video_path else ROOT / meta["video"]
    classes = meta["classes"]
    use_nms = meta["nms"]

    detector, class_names = load_detector_config(config_path, ROOT)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tracker = ByteTracker(**cfg["tracker"])

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if max_frames:
        total = min(total, max_frames) if total else max_frames

    accum = np.zeros((SIZE[1], SIZE[0]), dtype=np.float32)
    background = None
    frame_id = 0

    snapshots = []
    snap_every = max(1, total // gif_frames) if make_gif and total else 1

    pbar = tqdm(total=total or None, desc=f"heatmap {usecase}", unit="f")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, SIZE)

        if background is None:
            background = frame.copy()

        detections = [
            d for d in detector.detect(frame)
            if d.class_name in classes
        ]

        if use_nms:
            detections = cross_class_nms(
                filter_detections(detections, classes, min_box_side=12)
            )

        tracks = tracker.update(detections)

        for t in tracks:
            if category_name(t, class_names) not in classes:
                continue

            x1, y1, x2, y2 = t.tlbr.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if 0 <= cx < SIZE[0] and 0 <= cy < SIZE[1]:
                accum[cy, cx] += 1.0

        if make_gif and frame_id % snap_every == 0:
            snapshots.append((frame.copy(), accum.copy()))

        frame_id += 1
        pbar.update(1)

        if max_frames and frame_id >= max_frames:
            break

    pbar.close()
    cap.release()

    if accum.max() == 0:
        raise RuntimeError("Aucune détection accumulée : heatmap vide.")

    final = render_heatmap(background, accum, blur, alpha)

    out = Path(out_path) if out_path else ROOT / f"heatmap_{usecase}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out), final)
    print(f"Heatmap PNG générée : {out} ({frame_id} frames)")

    if make_gif:
        save_gif(
            snapshots=snapshots,
            final=final,
            out_gif=out_gif,
            usecase=usecase,
            blur=blur,
            alpha=alpha,
            gif_fps=gif_fps,
            freeze_seconds=freeze_seconds,
        )

    return out


def save_gif(
    snapshots,
    final,
    out_gif,
    usecase,
    blur,
    alpha,
    gif_fps,
    freeze_seconds,
):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("GIF : installe imageio avec `pip install imageio`.") from exc

    frames_rgb = []

    for background, accum in snapshots:
        img = render_heatmap(background, accum, blur, alpha)
        frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    frames_rgb.extend([final_rgb] * int(freeze_seconds * gif_fps))

    gif_path = Path(out_gif) if out_gif else ROOT / f"heatmap_{usecase}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(str(gif_path), frames_rgb, fps=gif_fps, loop=0)

    print(f"Heatmap GIF générée : {gif_path} ({len(frames_rgb)} frames)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Génère une heatmap de fréquentation en PNG et/ou GIF."
    )

    parser.add_argument("--usecase", default="retail", choices=list(USECASES))
    parser.add_argument("--video", default=None)
    parser.add_argument("--out", default=None, help="Chemin du PNG de sortie")
    parser.add_argument("--out-gif", default=None, help="Chemin du GIF de sortie")
    parser.add_argument("--blur", type=int, default=51)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=60,
        help="Nombre d'étapes échantillonnées dans le GIF",
    )
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--freeze-seconds",
        type=float,
        default=2.0,
        help="Durée du gel final dans le GIF",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Génère uniquement le PNG",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_heatmap(
        usecase=args.usecase,
        video_path=args.video,
        out_path=args.out,
        make_gif=not args.no_gif,
        out_gif=args.out_gif,
        blur=args.blur,
        alpha=args.alpha,
        max_frames=args.max_frames,
        gif_frames=args.gif_frames,
        gif_fps=args.gif_fps,
        freeze_seconds=args.freeze_seconds,
    )


if __name__ == "__main__":
    main()
