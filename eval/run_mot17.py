import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from eval.utils_mot17 import (
    iter_mot17_sequences,
    make_tracker,
    normalize_tracker_name,
    read_mot_detections,
    reset_track_ids,
)


def load_frame(img_dir, frame_id):
    if img_dir is None:
        return None

    for ext in ["jpg", "png", "jpeg"]:
        frame_path = img_dir / "{:06d}.{}".format(frame_id, ext)

        if frame_path.exists():
            return cv2.imread(str(frame_path))

    return None


def detection_to_object(detection):
    class DetectionObject:
        pass

    obj = DetectionObject()
    obj.xyxy = detection["xyxy"]
    obj.score = detection["score"]
    obj.class_id = detection["class_id"]

    return obj


def write_track(file, frame_id, track):
    x, y, w, h = track.tlwh
    track_id = int(track.track_id)
    score = float(getattr(track, "score", 1.0))

    file.write(
        "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
            frame_id,
            track_id,
            x,
            y,
            w,
            h,
            score,
        )
    )


def run_sequence(sequence, output_path, args):
    reset_track_ids()

    detections_by_frame = read_mot_detections(
        sequence["det_path"],
        min_score=args.min_det_score,
    )

    if not detections_by_frame:
        raise RuntimeError(
            "Aucune détection trouvée: {}".format(sequence["det_path"])
        )

    seq_length = sequence["seq_length"]

    if seq_length is None:
        seq_length = max(detections_by_frame.keys())

    frame_rate = sequence["frame_rate"]

    if frame_rate is None:
        frame_rate = args.frame_rate

    tracker_name = normalize_tracker_name(args.tracker)
    use_gmc = tracker_name == "botsort" and not args.no_gmc

    if use_gmc and sequence["img_dir"] is None:
        raise RuntimeError(
            "BoT-SORT avec GMC nécessite img1 pour {}. "
            "Utilise --no-gmc si tu n'as pas les images.".format(sequence["name"])
        )

    tracker = make_tracker(
        tracker_name,
        conf_thresh=args.conf_thresh,
        track_buffer=args.track_buffer,
        frame_rate=frame_rate,
        match_thresh=args.match_thresh,
        min_hits=args.min_hits,
        delta_t=args.delta_t,
        fuse_score=not args.no_fuse_score,
        gmc=not args.no_gmc,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        for frame_id in tqdm(range(1, seq_length + 1), desc=sequence["name"]):
            raw_detections = detections_by_frame.get(frame_id, [])
            detections = [detection_to_object(d) for d in raw_detections]

            if use_gmc:
                frame = load_frame(sequence["img_dir"], frame_id)
                tracks = tracker.update(detections, curr_img=frame)
            else:
                tracks = tracker.update(detections)

            for track in tracks:
                write_track(file, frame_id, track)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait les tracks MOT17 pour un tracker et un détecteur."
    )

    parser.add_argument("--mot-root", type=Path, required=True)
    parser.add_argument(
        "--tracker",
        required=True,
        choices=["sort", "bytetrack", "botsort", "ocsort", "bot-sort", "oc-sort"],
    )
    parser.add_argument(
        "--detector",
        required=True,
        choices=["DPM", "FRCNN", "SDP"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--delta-t", type=int, default=3)
    parser.add_argument("--min-det-score", type=float, default=-1.0)

    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--no-gmc", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    sequences = list(iter_mot17_sequences(args.mot_root, args.detector))

    if not sequences:
        raise RuntimeError(
            "Aucune séquence trouvée pour detector={} dans {}".format(
                args.detector,
                args.mot_root,
            )
        )

    print(
        "[MOT17] Extraction | tracker={} | detector={} | sequences={}".format(
            args.tracker,
            args.detector,
            len(sequences),
        )
    )

    for sequence in sequences:
        output_path = args.output_dir / "{}.txt".format(sequence["name"])
        run_sequence(sequence, output_path, args)

    print("[MOT17] Tracks écrits dans {}".format(args.output_dir))


if __name__ == "__main__":
    main()
