import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


SEQ_RE = re.compile(r"^MOT17-\d{2}-(DPM|FRCNN|SDP)$")


def is_sequence_key(key):
    return bool(SEQ_RE.match(key))


def is_result_key(key):
    return key == "COMBINED" or is_sequence_key(key)


def get_case_insensitive(mapping, key):
    key_lower = key.lower()

    for current_key, value in mapping.items():
        if str(current_key).lower() == key_lower:
            return value

    raise KeyError(key)


def as_float(value, label):
    try:
        return float(value)
    except Exception:
        raise ValueError("Valeur numérique invalide pour {}: {}".format(label, value))


def read_metric(block, metric):
    try:
        value = get_case_insensitive(block, metric)

        if not isinstance(value, dict):
            return as_float(value, metric)
    except KeyError:
        pass

    if metric == "MOTA":
        group_name = "CLEAR"
        metric_name = "MOTA"
    elif metric == "HOTA":
        group_name = "HOTA"
        metric_name = "HOTA"
    elif metric == "IDF1":
        group_name = "Identity"
        metric_name = "IDF1"
    else:
        raise ValueError("Métrique non supportée: {}".format(metric))

    group = get_case_insensitive(block, group_name)

    if not isinstance(group, dict):
        raise TypeError("Groupe métrique invalide: {}".format(group_name))

    value = get_case_insensitive(group, metric_name)

    return as_float(value, "{}.{}".format(group_name, metric_name))


def choose_results_container(data, source):
    if not isinstance(data, dict):
        raise TypeError("Objet JSON attendu dans {}".format(source))

    candidates = [
        data,
        data.get("results"),
        data.get("sequences"),
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        keys = [str(k) for k in candidate.keys()]

        if "COMBINED" in keys:
            return candidate

        for key in keys:
            if is_sequence_key(key):
                return candidate

    raise RuntimeError(
        "Impossible de trouver les résultats MOT17 dans {}".format(source)
    )


def infer_tracker_detector(json_file, eval_root, data):
    tracker = data.get("tracker")
    detector = data.get("detector")

    if tracker and detector:
        return tracker, detector

    relative = json_file.relative_to(eval_root)
    parts = relative.parts

    if len(parts) >= 2:
        tracker = parts[0]
        detector = Path(parts[1]).stem
        return tracker, detector

    stem = json_file.stem

    for detector_name in ["DPM", "FRCNN", "SDP"]:
        suffix = "_" + detector_name

        if stem.upper().endswith(suffix):
            return stem[:-len(suffix)], detector_name

    return stem, "UNKNOWN"


def parse_eval_json(json_file, eval_root):
    with json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    tracker, detector = infer_tracker_detector(json_file, eval_root, data)
    results = choose_results_container(data, json_file)

    combined = None
    sequence_rows = []

    for key, block in results.items():
        sequence_name = str(key)

        if not is_result_key(sequence_name):
            continue

        if not isinstance(block, dict):
            continue

        row = {
            "tracker": tracker,
            "detector": detector,
            "sequence": sequence_name,
            "MOTA": read_metric(block, "MOTA"),
            "HOTA": read_metric(block, "HOTA"),
            "IDF1": read_metric(block, "IDF1"),
        }

        if sequence_name == "COMBINED":
            combined = row
        else:
            sequence_rows.append(row)

    if combined is None:
        if not sequence_rows:
            raise RuntimeError("Aucun résultat exploitable dans {}".format(json_file))

        combined = {
            "tracker": tracker,
            "detector": detector,
            "sequence": "COMBINED",
            "MOTA": mean([row["MOTA"] for row in sequence_rows]),
            "HOTA": mean([row["HOTA"] for row in sequence_rows]),
            "IDF1": mean([row["IDF1"] for row in sequence_rows]),
        }

    return combined, sequence_rows


def should_skip_json(json_file):
    for part in json_file.parts:
        if part == "_sequence_json":
            return True

    return False


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_tracker(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["tracker"]].append(row)

    output = []

    for tracker, tracker_rows in sorted(groups.items()):
        output.append(
            {
                "tracker": tracker,
                "MOTA": mean([row["MOTA"] for row in tracker_rows]),
                "HOTA": mean([row["HOTA"] for row in tracker_rows]),
                "IDF1": mean([row["IDF1"] for row in tracker_rows]),
                "n_detectors": len(tracker_rows),
            }
        )

    return output


def aggregate_by_detector(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["detector"]].append(row)

    output = []

    for detector, detector_rows in sorted(groups.items()):
        output.append(
            {
                "detector": detector,
                "MOTA": mean([row["MOTA"] for row in detector_rows]),
                "HOTA": mean([row["HOTA"] for row in detector_rows]),
                "IDF1": mean([row["IDF1"] for row in detector_rows]),
                "n_trackers": len(detector_rows),
            }
        )

    return output


def fmt(value):
    return "{:.3f}".format(float(value))


def write_markdown(path, tracker_detector_rows, tracker_rows, detector_rows, sequence_rows):
    lines = []

    lines.append("# MOT17 benchmark")
    lines.append("")

    lines.append("## Par tracker")
    lines.append("")
    lines.append("| Rank | Tracker | MOTA | HOTA | IDF1 | N detectors |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    ranked_trackers = sorted(
        tracker_rows,
        key=lambda row: row["HOTA"],
        reverse=True,
    )

    for rank, row in enumerate(ranked_trackers, start=1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                rank,
                row["tracker"],
                fmt(row["MOTA"]),
                fmt(row["HOTA"]),
                fmt(row["IDF1"]),
                row["n_detectors"],
            )
        )

    lines.append("")
    lines.append("## Par tracker et détecteur")
    lines.append("")
    lines.append("| Tracker | Detector | MOTA | HOTA | IDF1 |")
    lines.append("|---|---|---:|---:|---:|")

    for row in sorted(tracker_detector_rows, key=lambda r: (r["tracker"], r["detector"])):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["tracker"],
                row["detector"],
                fmt(row["MOTA"]),
                fmt(row["HOTA"]),
                fmt(row["IDF1"]),
            )
        )

    lines.append("")
    lines.append("## Par détecteur")
    lines.append("")
    lines.append("| Detector | MOTA | HOTA | IDF1 | N trackers |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in sorted(detector_rows, key=lambda r: r["detector"]):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["detector"],
                fmt(row["MOTA"]),
                fmt(row["HOTA"]),
                fmt(row["IDF1"]),
                row["n_trackers"],
            )
        )

    lines.append("")
    lines.append("## Par séquence")
    lines.append("")
    lines.append("| Tracker | Detector | Sequence | MOTA | HOTA | IDF1 |")
    lines.append("|---|---|---|---:|---:|---:|")

    for row in sorted(sequence_rows, key=lambda r: (r["tracker"], r["detector"], r["sequence"])):
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["tracker"],
                row["detector"],
                row["sequence"],
                fmt(row["MOTA"]),
                fmt(row["HOTA"]),
                fmt(row["IDF1"]),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agrège les JSON d'évaluation MOT17."
    )

    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    json_files = []

    for json_file in sorted(args.eval_root.rglob("*.json")):
        if should_skip_json(json_file):
            continue

        json_files.append(json_file)

    if not json_files:
        raise RuntimeError("Aucun JSON trouvé dans {}".format(args.eval_root))

    tracker_detector_rows = []
    sequence_rows = []

    for json_file in json_files:
        combined, sequences = parse_eval_json(json_file, args.eval_root)
        tracker_detector_rows.append(combined)
        sequence_rows.extend(sequences)

    tracker_rows = aggregate_by_tracker(tracker_detector_rows)
    detector_rows = aggregate_by_detector(tracker_detector_rows)

    tracker_detector_rows.sort(key=lambda row: row["HOTA"], reverse=True)
    tracker_rows.sort(key=lambda row: row["HOTA"], reverse=True)
    detector_rows.sort(key=lambda row: row["HOTA"], reverse=True)

    write_csv(
        args.output_dir / "summary_by_tracker_detector.csv",
        tracker_detector_rows,
        ["tracker", "detector", "sequence", "MOTA", "HOTA", "IDF1"],
    )

    write_csv(
        args.output_dir / "summary_by_tracker.csv",
        tracker_rows,
        ["tracker", "MOTA", "HOTA", "IDF1", "n_detectors"],
    )

    write_csv(
        args.output_dir / "summary_by_detector.csv",
        detector_rows,
        ["detector", "MOTA", "HOTA", "IDF1", "n_trackers"],
    )

    write_csv(
        args.output_dir / "summary_by_sequence.csv",
        sorted(sequence_rows, key=lambda r: (r["tracker"], r["detector"], r["sequence"])),
        ["tracker", "detector", "sequence", "MOTA", "HOTA", "IDF1"],
    )

    write_markdown(
        args.output_dir / "summary.md",
        tracker_detector_rows,
        tracker_rows,
        detector_rows,
        sequence_rows,
    )

    print("[MOT17] Agrégation terminée")
    print("[MOT17] {}".format(args.output_dir / "summary.md"))


if __name__ == "__main__":
    main()
