import argparse
import subprocess
import sys
from pathlib import Path

from eval.utils_mot17 import DETECTORS, TRACKERS, normalize_tracker_name


def run_command(cmd, verbose=False):
    if verbose:
        print("")
        print("[CMD] " + " ".join([str(x) for x in cmd]))

    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MOT17 sur plusieurs trackers et détecteurs."
    )

    parser.add_argument("--mot-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/mot17_benchmark"),
    )

    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["sort", "bytetrack", "ocsort", "botsort", "cbiou"],
        choices=TRACKERS,
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DETECTORS,
        choices=DETECTORS,
    )

    parser.add_argument("--conf-thresh", type=float, default=0.1)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--delta-t", type=int, default=3)
    parser.add_argument("--min-det-score", type=float, default=-1.0)

    parser.add_argument("--no-fuse-score", action="store_true")
    parser.add_argument("--no-gmc", action="store_true")

    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    args.trackers = [normalize_tracker_name(tracker) for tracker in args.trackers]

    return args


def run_tracking(args, tracker, detector, tracker_output_dir):
    cmd = [
        sys.executable,
        "-m",
        "eval.run_mot17",
        "--mot-root",
        str(args.mot_root),
        "--tracker",
        tracker,
        "--detector",
        detector,
        "--output-dir",
        str(tracker_output_dir),
        "--conf-thresh",
        str(args.conf_thresh),
        "--track-buffer",
        str(args.track_buffer),
        "--frame-rate",
        str(args.frame_rate),
        "--match-thresh",
        str(args.match_thresh),
        "--min-hits",
        str(args.min_hits),
        "--delta-t",
        str(args.delta_t),
        "--min-det-score",
        str(args.min_det_score),
    ]

    if args.no_fuse_score:
        cmd.append("--no-fuse-score")

    if args.no_gmc:
        cmd.append("--no-gmc")

    run_command(cmd, verbose=args.verbose)


def run_evaluation(args, tracker, detector, tracker_output_dir, eval_output_path):
    cmd = [
        sys.executable,
        "-m",
        "eval.evaluate_mot17",
        "--gt-dir",
        str(args.mot_root),
        "--tracker-dir",
        str(tracker_output_dir),
        "--tracker-name",
        tracker,
        "--detector",
        detector,
        "--output",
        str(eval_output_path),
    ]

    if args.verbose:
        cmd.append("--verbose")

    run_command(cmd, verbose=args.verbose)


def run_aggregation(args, eval_root, summary_root):
    cmd = [
        sys.executable,
        "-m",
        "eval.aggregate_mot17",
        "--eval-root",
        str(eval_root),
        "--output-dir",
        str(summary_root),
    ]

    run_command(cmd, verbose=args.verbose)


def main():
    args = parse_args()

    tracks_root = args.results_root / "tracks"
    eval_root = args.results_root / "eval"
    summary_root = args.results_root / "summary"

    total_jobs = len(args.trackers) * len(args.detectors)
    current_job = 0

    print("")
    print("[MOT17] Benchmark")
    print("[MOT17] Trackers  : {}".format(", ".join(args.trackers)))
    print("[MOT17] Detectors : {}".format(", ".join(args.detectors)))
    print("[MOT17] Results   : {}".format(args.results_root))
    print("[MOT17] Jobs      : {}".format(total_jobs))

    for tracker in args.trackers:
        for detector in args.detectors:
            current_job += 1

            tracker_output_dir = tracks_root / tracker / detector
            eval_output_path = eval_root / tracker / "{}.json".format(detector)

            print("")
            print("=" * 70)
            print(
                "[MOT17] Job {}/{} | tracker={} | detector={}".format(
                    current_job,
                    total_jobs,
                    tracker,
                    detector,
                )
            )
            print("=" * 70)

            if args.skip_run:
                print("[MOT17] Skip extraction")
            else:
                run_tracking(args, tracker, detector, tracker_output_dir)

            if args.skip_eval:
                print("[MOT17] Skip évaluation")
            else:
                run_evaluation(
                    args,
                    tracker,
                    detector,
                    tracker_output_dir,
                    eval_output_path,
                )

    if args.skip_aggregate:
        print("")
        print("[MOT17] Skip agrégation")
    else:
        print("")
        print("=" * 70)
        print("[MOT17] Agrégation finale")
        print("=" * 70)
        run_aggregation(args, eval_root, summary_root)

    print("")
    print("[MOT17] Benchmark terminé")
    print("[MOT17] Résumé: {}".format(summary_root / "summary.md"))


if __name__ == "__main__":
    main()
