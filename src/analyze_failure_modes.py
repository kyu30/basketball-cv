"""Quantify how tracking quality relates to box size and occlusion, per clip.

Joins each test clip's ground-truth difficulty (median player box area; occlusion
rate = fraction of frames with any player-pair IoU > 0.3) with that clip's tracking
metrics from metrics.csv, then reports correlations and an easy-vs-hard split.
Turns the CI3 "small players / occlusion hurt tracking" claims into numbers.

Usage:
    python src/analyze_failure_modes.py --tracker reid_motion
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from collections import defaultdict

from data_loading import get_basketball_clips, load_gt, load_seqinfo
from iou import max_offdiag_iou, iou_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS = PROJECT_ROOT / "outputs" / "tracking" / "metrics.csv"
DETS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "dets"


def recall_by_size(conf_thresh=0.25):
    """Detection recall (gt matched to a detection at IoU>=0.5) binned by gt box area.
    Directly tests whether small/distant players are missed more."""
    areas, recalled = [], []
    for clip_path in get_basketball_clips("val"):
        name = load_seqinfo(clip_path)["name"]
        dpath = DETS_DIR / f"{name}.txt"
        if not dpath.exists():
            continue
        dets = defaultdict(list)
        for line in dpath.read_text().splitlines():
            f, x, y, w, h, c = line.split(",")
            if float(c) >= conf_thresh:
                dets[int(f)].append((float(x), float(y), float(x) + float(w), float(y) + float(h)))
        gt = load_gt(clip_path)
        for frame, g in gt.groupby("frame"):
            gb = g[["x", "y", "w", "h"]].to_numpy(dtype=float)
            gxyxy = gb.copy(); gxyxy[:, 2] += gxyxy[:, 0]; gxyxy[:, 3] += gxyxy[:, 1]
            db = np.array(dets.get(frame, []), dtype=float).reshape(-1, 4)
            ious = iou_matrix(gxyxy, db)
            hit = ious.max(axis=1) >= 0.5 if db.size else np.zeros(len(gb), bool)
            areas.extend((gb[:, 2] * gb[:, 3]).tolist())
            recalled.extend(hit.tolist())
    areas, recalled = np.array(areas), np.array(recalled)
    qs = np.quantile(areas, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"\ndetection recall by gt box-area quartile (conf>={conf_thresh}, {len(areas)} boxes):")
    print(f"{'quartile':<12}{'area range (px^2)':>22}{'recall':>9}")
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (areas >= lo) & (areas <= hi if i == 3 else areas < hi)
        print(f"Q{i+1:<11}{f'{lo:.0f}-{hi:.0f}':>22}{recalled[m].mean():>9.3f}")


def clip_difficulty(clip_path: Path):
    gt = load_gt(clip_path)
    areas = (gt["w"] * gt["h"]).median()
    occ = 0
    frames = gt.groupby("frame")
    for _, g in frames:
        boxes = g[["x", "y", "w", "h"]].to_numpy(dtype=float)
        if max_offdiag_iou(boxes) > 0.3:
            occ += 1
    return float(areas), occ / max(frames.ngroups, 1)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", default="reid_motion")
    args = parser.parse_args()

    metrics = {r["clip"]: r for r in csv.DictReader(open(METRICS)) if r["tracker"] == args.tracker}
    if not metrics:
        raise SystemExit(f"No per-clip rows for tracker '{args.tracker}' in {METRICS}")

    rows = []
    for clip_path in get_basketball_clips("val"):
        name = load_seqinfo(clip_path)["name"]
        if name not in metrics:
            continue
        area, occ = clip_difficulty(clip_path)
        m = metrics[name]
        rows.append({"clip": name, "box_area": area, "occ_rate": occ,
                     "HOTA": float(m["HOTA"]), "AssA": float(m["AssA"]),
                     "DetA": float(m["DetA"]), "IDSW": int(m["IDSW"])})

    rows.sort(key=lambda r: r["box_area"])
    print(f"tracker: {args.tracker}  ({len(rows)} clips, sorted by box size)\n")
    print(f"{'clip':<22}{'boxArea':>9}{'occ%':>7}{'HOTA':>8}{'AssA':>8}{'DetA':>8}{'IDSW':>7}")
    for r in rows:
        print(f"{r['clip']:<22}{r['box_area']:>9.0f}{100*r['occ_rate']:>7.0f}"
              f"{r['HOTA']:>8.3f}{r['AssA']:>8.3f}{r['DetA']:>8.3f}{r['IDSW']:>7d}")

    area = np.array([r["box_area"] for r in rows])
    occ = np.array([r["occ_rate"] for r in rows])
    hota = np.array([r["HOTA"] for r in rows])
    deta = np.array([r["DetA"] for r in rows])
    assa = np.array([r["AssA"] for r in rows])
    print("\ncorrelations (Pearson r):")
    print(f"  box area  vs HOTA: {pearson(area, hota):+.2f}   vs DetA: {pearson(area, deta):+.2f}")
    print(f"  occ rate  vs HOTA: {pearson(occ, hota):+.2f}   vs AssA: {pearson(occ, assa):+.2f}")

    half = len(rows) // 2
    small, large = rows[:half], rows[-half:]
    mean = lambda rs, k: np.mean([r[k] for r in rs])
    print(f"\nsmall-box clips (n={len(small)}): HOTA {mean(small,'HOTA'):.3f}  DetA {mean(small,'DetA'):.3f}")
    print(f"large-box clips (n={len(large)}): HOTA {mean(large,'HOTA'):.3f}  DetA {mean(large,'DetA'):.3f}")

    recall_by_size()


if __name__ == "__main__":
    main()
