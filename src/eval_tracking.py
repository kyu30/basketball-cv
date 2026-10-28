"""Score tracker outputs with TrackEval: HOTA (primary) + MOTA + IDF1.

Reads MOT-format results written by track_clips.py, stages the SportsMOT test-clip
ground truth into the folder layout TrackEval expects, runs the official metrics,
and writes per-clip + overall numbers to outputs/tracking/metrics.csv.

HOTA is the SportsMOT paper's primary metric; MOTA and IDF1 are included for
comparison with older tracking literature (per the project's evaluation plan).

Usage:
    python src/eval_tracking.py                     # all trackers found under outputs/tracking/
    python src/eval_tracking.py --trackers iou bytetrack
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

# TrackEval (unmaintained) still uses np.float/np.int/np.bool, removed in NumPy 2.x.
# Restore the aliases to the builtins before importing it.
for _name, _builtin in {"float": float, "int": int, "bool": bool, "object": object}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import trackeval  # noqa: E402

from data_loading import get_basketball_clips, load_seqinfo  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"
GT_STAGE = TRACKING_DIR / "trackeval_gt"  # staged gt for TrackEval
BENCHMARK, SPLIT = "SPORTSMOT", "test"


def stage_gt() -> dict[str, int]:
    """Copy each test clip's gt.txt into GT_STAGE/<seq>/gt/gt.txt. Returns seq lengths."""
    seq_info = {}
    for clip_path in get_basketball_clips("val"):  # official val = our tracking test set
        info = load_seqinfo(clip_path)
        seq = info["name"]
        dst = GT_STAGE / seq / "gt"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip_path / "gt" / "gt.txt", dst / "gt.txt")
        seq_info[seq] = info["seqlength"]
    return seq_info


def discover_trackers(requested: list[str] | None) -> list[str]:
    found = sorted(p.name for p in TRACKING_DIR.iterdir()
                   if p.is_dir() and (p / "data").is_dir())
    if requested:
        missing = set(requested) - set(found)
        if missing:
            raise SystemExit(f"No results for tracker(s) {sorted(missing)} under {TRACKING_DIR}")
        return requested
    if not found:
        raise SystemExit(f"No tracker results under {TRACKING_DIR} - run src/track_clips.py first.")
    return found


def run_trackeval(trackers: list[str], seq_info: dict[str, int]) -> dict:
    eval_config = {**trackeval.Evaluator.get_default_eval_config(),
                   "USE_PARALLEL": False, "PRINT_CONFIG": False,
                   "PRINT_RESULTS": True, "OUTPUT_SUMMARY": False,
                   "OUTPUT_DETAILED": False, "PLOT_CURVES": False}
    dataset_config = {**trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
                      "GT_FOLDER": str(GT_STAGE),
                      "TRACKERS_FOLDER": str(TRACKING_DIR),
                      "SKIP_SPLIT_FOL": True,         # results already at <tracker>/data/<seq>.txt
                      "BENCHMARK": BENCHMARK, "SPLIT_TO_EVAL": SPLIT,
                      "TRACKERS_TO_EVAL": trackers,
                      "DO_PREPROC": False,            # single-class sport, no distractor classes
                      "SEQ_INFO": seq_info, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    output, _ = evaluator.evaluate([dataset], metrics)
    return output["MotChallenge2DBox"]


def extract_row(res: dict) -> dict:
    """Pull the headline scalars out of one sequence's result dict."""
    hota, clear, ident = res["HOTA"], res["CLEAR"], res["Identity"]
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
        "MOTA": float(clear["MOTA"]),
        "IDF1": float(ident["IDF1"]),
        "IDSW": int(clear["IDSW"]),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trackers", nargs="+", default=None, help="defaults to all found")
    args = parser.parse_args()

    trackers = discover_trackers(args.trackers)
    seq_info = stage_gt()
    print(f"Evaluating {trackers} on {len(seq_info)} clips")
    output = run_trackeval(trackers, seq_info)

    rows = []
    for tracker in trackers:
        per_seq = output[tracker]
        for seq, res in per_seq.items():
            label = "OVERALL" if seq == "COMBINED_SEQ" else seq
            rows.append({"tracker": tracker, "clip": label,
                         **extract_row(res["pedestrian"])})

    rows.sort(key=lambda r: (r["tracker"], r["clip"] != "OVERALL", r["clip"]))
    out_csv = TRACKING_DIR / "metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_csv}\n")
    print(f"{'tracker':<10}{'HOTA':>8}{'DetA':>8}{'AssA':>8}{'MOTA':>8}{'IDF1':>8}{'IDSW':>8}")
    for r in rows:
        if r["clip"] == "OVERALL":
            print(f"{r['tracker']:<10}{r['HOTA']:>8.4f}{r['DetA']:>8.4f}{r['AssA']:>8.4f}"
                  f"{r['MOTA']:>8.4f}{r['IDF1']:>8.4f}{r['IDSW']:>8d}")


if __name__ == "__main__":
    main()
