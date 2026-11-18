"""Failure analysis: categorize tracker ID confusions as same-team vs cross-team.

Hypothesis (the jersey-similarity story): a learned appearance embedding should cut
CROSS-team confusions (teams wear different colors, easy to tell apart) far more
than SAME-team ones (teammates wear identical jerseys, near-indistinguishable by
appearance). This script quantifies that for one or more trackers.

Method, per clip:
  1. Assign every ground-truth player to one of two teams by clustering mean jersey
     color (torso region) with k-means (k=2).
  2. Match predicted boxes to ground-truth boxes each frame (IoU, Hungarian).
  3. For each predicted track, walk its timeline of matched gt identities; every
     change gt_a -> gt_b is one confusion event, labeled same- or cross-team.

Usage:
    python src/analyze_switches.py --trackers iou reid
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from data_loading import get_basketball_clips, load_gt, load_seqinfo
from iou import iou_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"


def mot_to_xyxy(df_like) -> np.ndarray:
    a = np.asarray(df_like, dtype=float)
    xyxy = a[:, :4].copy()
    xyxy[:, 2] += xyxy[:, 0]
    xyxy[:, 3] += xyxy[:, 1]
    return xyxy


def assign_teams(clip_path: Path, gt) -> dict[int, int]:
    """Cluster gt tracks into 2 teams by mean torso color."""
    colors, ids = [], []
    for tid, g in gt.groupby("track_id"):
        frames = g.sample(min(6, len(g)), random_state=0) if len(g) > 6 else g
        samples = []
        for frame, x, y, w, h in frames[["frame", "x", "y", "w", "h"]].itertuples(index=False):
            img = cv2.imread(str(clip_path / "img1" / f"{int(frame):06d}.jpg"))
            if img is None:
                continue
            # central torso patch (upper-middle of the box)
            cx1, cx2 = int(x + 0.3 * w), int(x + 0.7 * w)
            cy1, cy2 = int(y + 0.15 * h), int(y + 0.5 * h)
            patch = img[max(cy1, 0):cy2, max(cx1, 0):cx2]
            if patch.size:
                samples.append(patch.reshape(-1, 3).mean(axis=0))
        if samples:
            colors.append(np.mean(samples, axis=0))
            ids.append(int(tid))
    colors = np.array(colors, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(colors, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return {tid: int(lab) for tid, lab in zip(ids, labels.ravel())}


def load_pred(tracker: str, clip: str) -> dict[int, list]:
    by_frame = defaultdict(list)
    for line in (TRACKING_DIR / tracker / "data" / f"{clip}.txt").read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        by_frame[int(f)].append((int(tid), float(x), float(y), float(w), float(h)))
    return by_frame


def pred_to_gt_timeline(gt, pred_by_frame) -> dict[int, list]:
    """For each predicted id, the ordered list of matched gt track ids."""
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    timelines = defaultdict(list)
    for frame, prows in sorted(pred_by_frame.items()):
        if frame not in gt_by_frame:
            continue
        g = gt_by_frame[frame]
        gt_ids = g["track_id"].to_numpy()
        gt_boxes = mot_to_xyxy(g[["x", "y", "w", "h"]])
        pred_ids = np.array([r[0] for r in prows])
        pred_boxes = mot_to_xyxy([r[1:] for r in prows])
        ious = iou_matrix(pred_boxes, gt_boxes)
        pi, gi = linear_sum_assignment(-ious)
        for p, gq in zip(pi, gi):
            if ious[p, gq] >= 0.5:
                timelines[int(pred_ids[p])].append(int(gt_ids[gq]))
    return timelines


def count_confusions(timelines, team) -> tuple[int, int]:
    same = cross = 0
    for seq in timelines.values():
        compact = [g for i, g in enumerate(seq) if i == 0 or g != seq[i - 1]]  # drop repeats
        for a, b in zip(compact, compact[1:]):
            if a in team and b in team:
                if team[a] == team[b]:
                    same += 1
                else:
                    cross += 1
    return same, cross


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trackers", nargs="+", default=["iou", "reid"])
    args = parser.parse_args()

    clips = get_basketball_clips("val")
    teams = {}  # cache team assignment per clip (shared across trackers)
    totals = {t: [0, 0] for t in args.trackers}
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        gt = load_gt(clip_path)
        teams[info["name"]] = assign_teams(clip_path, gt)
        for tracker in args.trackers:
            tl = pred_to_gt_timeline(gt, load_pred(tracker, info["name"]))
            s, c = count_confusions(tl, teams[info["name"]])
            totals[tracker][0] += s
            totals[tracker][1] += c

    print(f"\n{'tracker':<12}{'same-team':>11}{'cross-team':>12}{'total':>8}{'cross %':>9}")
    for t in args.trackers:
        s, c = totals[t]
        print(f"{t:<12}{s:>11}{c:>12}{s + c:>8}{100 * c / max(s + c, 1):>8.1f}%")


if __name__ == "__main__":
    main()
