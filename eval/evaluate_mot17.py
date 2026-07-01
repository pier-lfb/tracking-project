import os
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from statistics import mean

from eval.utils_mot17 import iter_mot17_sequences


def run_command(cmd, verbose=False):
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    if verbose:
        print("")
        print("[CMD] " + " ".join([str(x) for x in cmd]))

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

    if result.returncode != 0:
        print("")
        print("[MOT17] Erreur pendant l'évaluation")

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        result.check_returncode()


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
        raise TypeError(
            "Groupe métrique invalide pour {}: {}".format(
                group_name,
                type(group).__name__,
            )
        )

    value = get_case_insensitive(group, metric_name)

    return as_float(value, "{}.{}".format(group_name, metric_name))


def looks_like_metric_block(block):
    if not isinstance(block, dict):
        return False

    metric_names = ["MOTA", "HOTA", "IDF1", "CLEAR", "Identity"]

    for metric_name in metric_names:
        for key in block.keys():
            if str(key).lower() == metric_name.lower():
                return True

    return False


def find_metric_block(data, sequence_name):
    if not isinstance(data, dict):
        raise TypeError("JSON d'évaluation invalide")

    if looks_like_metric_block(data):
        return data

    for container_key in ["results", "sequences"]:
        if container_key in data and isinstance(data[container_key], dict):
            container = data[container_key]

            if sequence_name in container:
                return container[sequence_name]

            if "COMBINED" in container:
                return container["COMBINED"]

            if looks_like_metric_block(container):
                return container

    if sequence_name in data:
        return data[sequence_name]

    if "COMBINED" in data:
        return data["COMBINED"]

    for value in data.values():
        if looks_like_metric_block(value):
            return value

    raise RuntimeError(
        "Impossible de trouver les métriques dans le JSON pour {}".format(
            sequence_name
        )
    )


def read_eval_json(json_path, sequence_name):
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    block = find_metric_block(data, sequence_name)

    return {
        "MOTA": read_metric(block, "MOTA"),
        "HOTA": read_metric(block, "HOTA"),
        "IDF1": read_metric(block, "IDF1"),
    }


def validate_inputs(gt_dir, tracker_dir, detector):
    sequences = list(iter_mot17_sequences(gt_dir, detector))

    if not sequences:
        raise RuntimeError(
            "Aucune séquence GT trouvée pour detector={} dans {}".format(
                detector,
                gt_dir,
            )
        )

    missing_gt = []
    missing_tracker = []

    for sequence in sequences:
        if sequence["gt_path"] is None:
            missing_gt.append(sequence["name"])

        tracker_file = tracker_dir / "{}.txt".format(sequence["name"])

        if not tracker_file.exists():
            missing_tracker.append(str(tracker_file))

    if missing_gt:
        raise RuntimeError(
            "GT manquant pour les séquences:\n  - {}".format(
                "\n  - ".join(missing_gt)
            )
        )

    if missing_tracker:
        raise RuntimeError(
            "Fichiers tracker manquants:\n  - {}".format(
                "\n  - ".join(missing_tracker)
            )
        )

    return sequences


def evaluate_sequence(sequence, tracker_file, output_file, args):
    cmd = [
        "trackers",
        "eval",
        "--gt",
        str(sequence["gt_path"]),
        "--tracker",
        str(tracker_file),
        "--metrics",
    ]

    cmd.extend(args.metrics)
    cmd.append("--columns")
    cmd.extend(args.columns)
    cmd.extend(["--output", str(output_file)])

    run_command(cmd, verbose=args.verbose)


def write_final_json(output_path, tracker_name, detector, sequence_rows):
    results = {}

    for row in sequence_rows:
        results[row["sequence"]] = {
            "MOTA": row["MOTA"],
            "HOTA": row["HOTA"],
            "IDF1": row["IDF1"],
        }

    results["COMBINED"] = {
        "MOTA": mean([row["MOTA"] for row in sequence_rows]),
        "HOTA": mean([row["HOTA"] for row in sequence_rows]),
        "IDF1": mean([row["IDF1"] for row in sequence_rows]),
    }

    final = {
        "tracker": tracker_name,
        "detector": detector,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(final, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évalue les résultats d'un tracker MOT17 contre le GT."
    )

    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--tracker-dir", type=Path, required=True)
    parser.add_argument(
        "--detector",
        required=True,
        choices=["DPM", "FRCNN", "SDP"],
    )
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CLEAR", "HOTA", "Identity"],
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["MOTA", "HOTA", "IDF1"],
    )

    parser.add_argument("--tracker-name", default=None)
    parser.add_argument("--keep-sequence-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if shutil.which("trackers") is None:
        raise RuntimeError(
            "Commande 'trackers' introuvable. Active ton environnement Python."
        )

    if not args.gt_dir.exists():
        raise FileNotFoundError("GT dir introuvable: {}".format(args.gt_dir))

    if not args.tracker_dir.exists():
        raise FileNotFoundError(
            "Tracker dir introuvable: {}".format(args.tracker_dir)
        )

    sequences = validate_inputs(args.gt_dir, args.tracker_dir, args.detector)

    tracker_name = args.tracker_name

    if tracker_name is None:
        tracker_name = args.tracker_dir.parent.name

    sequence_json_dir = args.output.parent / "_sequence_json" / tracker_name / args.detector
    sequence_json_dir.mkdir(parents=True, exist_ok=True)

    sequence_rows = []

    print("")
    print(
        "[MOT17] Évaluation | tracker={} | detector={} | sequences={}".format(
            tracker_name,
            args.detector,
            len(sequences),
        )
    )

    print("")
    print("{:<15} {:>8} {:>8} {:>8}".format(
        "Sequence",
        "MOTA",
        "HOTA",
        "IDF1",
    ))
    print("-" * 50)

    for sequence in sequences:
        tracker_file = args.tracker_dir / "{}.txt".format(sequence["name"])
        sequence_output = sequence_json_dir / "{}.json".format(sequence["name"])

        evaluate_sequence(
            sequence,
            tracker_file,
            sequence_output,
            args,
        )

        metrics = read_eval_json(sequence_output, sequence["name"])

        sequence_rows.append(
            {
                "sequence": sequence["name"],
                "MOTA": metrics["MOTA"],
                "HOTA": metrics["HOTA"],
                "IDF1": metrics["IDF1"],
            }
        )

        gt_sequence = sequence["gt_path"].parent.parent.name

        print(
            "{:<15} {:>8.3f} {:>8.3f} {:>8.3f}".format(
                sequence["name"],
                metrics["MOTA"],
                metrics["HOTA"],
                metrics["IDF1"],
            )
        )

    combined_mota = mean([row["MOTA"] for row in sequence_rows])
    combined_hota = mean([row["HOTA"] for row in sequence_rows])
    combined_idf1 = mean([row["IDF1"] for row in sequence_rows])

    print("-" * 50)
    print(
        "{:<15} {:>8.3f} {:>8.3f} {:>8.3f}".format(
            "COMBINED",
            combined_mota,
            combined_hota,
            combined_idf1,
        )
    )
    print("")

    write_final_json(
        args.output,
        tracker_name,
        args.detector,
        sequence_rows,
    )

    if not args.keep_sequence_json:
        shutil.rmtree(sequence_json_dir, ignore_errors=True)

    print("[MOT17] JSON écrit: {}".format(args.output))


if __name__ == "__main__":
    main()
