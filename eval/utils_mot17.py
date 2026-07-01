import csv
import configparser
from pathlib import Path

import numpy as np

from src.tracking.basetrack import BaseTrack
from src.tracking.sort import SortTracker
from src.tracking.bytetrack import ByteTracker
from src.tracking.botsort import BotSortTracker
from src.tracking.ocsort import OcSortTracker
from src.tracking.c_biou import CBIoUTracker


DETECTORS = ["DPM", "FRCNN", "SDP"]
TRACKERS = ["sort", "bytetrack", "botsort", "ocsort", "cbiou"]


def normalize_tracker_name(name):
    tracker = str(name).lower()

    if tracker not in TRACKERS:
        raise ValueError(
            "Tracker inconnu: {}. Choix possibles: {}".format(
                name,
                ", ".join(TRACKERS),
            )
        )

    return tracker


def normalize_detector_name(name):
    detector = str(name).upper()

    if detector not in DETECTORS:
        raise ValueError(
            "Détecteur inconnu: {}. Choix possibles: {}".format(
                name,
                ", ".join(DETECTORS),
            )
        )

    return detector


def sequence_detector(sequence_name):
    detector = sequence_name.split("-")[-1].upper()

    if detector not in DETECTORS:
        return None

    return detector


def is_mot17_sequence(sequence_name):
    parts = sequence_name.split("-")

    if len(parts) != 3:
        return False

    if parts[0] != "MOT17":
        return False

    if sequence_detector(sequence_name) is None:
        return False

    return True


def read_seqinfo(sequence_dir):
    seqinfo_path = sequence_dir / "seqinfo.ini"

    info = {
        "seq_length": None,
        "frame_rate": None,
    }

    if not seqinfo_path.exists():
        return info

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)

    if "Sequence" not in parser:
        return info

    section = parser["Sequence"]

    info["seq_length"] = section.getint("seqLength", fallback=None)
    info["frame_rate"] = section.getint("frameRate", fallback=None)

    return info


def iter_mot17_sequences(mot_root, detector):
    mot_root = Path(mot_root)
    detector = normalize_detector_name(detector)

    if not mot_root.exists():
        raise FileNotFoundError("Dossier MOT17 introuvable: {}".format(mot_root))

    for sequence_dir in sorted(mot_root.iterdir()):
        if not sequence_dir.is_dir():
            continue

        sequence_name = sequence_dir.name

        if not is_mot17_sequence(sequence_name):
            continue

        if sequence_detector(sequence_name) != detector:
            continue

        det_path = sequence_dir / "det" / "det.txt"
        gt_path = find_gt_path(mot_root, sequence_name)
        img_dir = sequence_dir / "img1"

        if not det_path.exists():
            continue

        seqinfo = read_seqinfo(sequence_dir)

        yield {
            "name": sequence_name,
            "path": sequence_dir,
            "detector": detector,
            "det_path": det_path,
            "gt_path": gt_path,
            "img_dir": img_dir if img_dir.exists() else None,
            "seq_length": seqinfo["seq_length"],
            "frame_rate": seqinfo["frame_rate"],
        }


def read_mot_detections(det_path, min_score=-1.0):
    detections_by_frame = {}

    with Path(det_path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 7:
                continue

            frame_id = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            score = float(row[6])

            if score < min_score:
                continue

            detection = {
                "xyxy": np.array([x, y, x + w, y + h], dtype=np.float32),
                "score": score,
                "class_id": 1,
            }

            detections_by_frame.setdefault(frame_id, []).append(detection)

    return detections_by_frame


def make_tracker(
    tracker_name,
    conf_thresh,
    track_buffer,
    frame_rate,
    match_thresh,
    min_hits,
    delta_t,
    fuse_score,
    gmc,
):
    name = normalize_tracker_name(tracker_name)

    if name == "sort":
        return SortTracker(
            conf_thresh=conf_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            min_hits=min_hits,
        )

    if name == "bytetrack":
        return ByteTracker(
            conf_thresh=conf_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )

    if name == "botsort":
        return BotSortTracker(
            conf_thresh=conf_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
            with_gmc=gmc,
        )

    if name == "ocsort":
        return OcSortTracker(
            conf_thresh=conf_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            delta_t=delta_t,
        )

    if name == "cbiou":
        return CBIoUTracker(
            conf_thresh=conf_thresh,
            track_buffer=track_buffer,
            motion="byte",
            frame_rate=frame_rate,
            match_thresh1=0.7,
            match_thresh2=0.6,
            b1=0.3,
            b2=0.5,
        )

    raise ValueError("Tracker inconnu: {}".format(tracker_name))


def reset_track_ids():
    BaseTrack.reset_id()


def expected_tracker_files(mot_root, detector):
    files = []

    for sequence in iter_mot17_sequences(mot_root, detector):
        files.append(sequence["name"] + ".txt")

    return files


def sequence_base_name(sequence_name):
    parts = sequence_name.split("-")

    if len(parts) < 3:
        return sequence_name

    return "-".join(parts[:2])


def find_gt_path(mot_root, sequence_name):
    base_name = sequence_base_name(sequence_name)

    preferred = mot_root / "{}-FRCNN".format(base_name) / "gt" / "gt.txt"

    if preferred.exists():
        return preferred

    for detector in DETECTORS:
        candidate = mot_root / "{}-{}".format(base_name, detector) / "gt" / "gt.txt"

        if candidate.exists():
            return candidate

    return None
