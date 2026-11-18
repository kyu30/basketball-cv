"""Heuristic basketball detector: color + shape blob proposals.

No ball ground truth exists in SportsMOT, so this proposes candidate ball
locations by orange-color segmentation plus round/small-blob filtering, and
overlays the top candidates on sample frames for visual assessment of how often a
plausible ball is found before committing to a learned detector.

Usage:
    python src/ball_heuristic.py --samples 16        # overlay candidates on sample frames
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ball"

# Basketball orange in HSV (OpenCV H in 0-179). Tuned to be saturated orange,
# narrower than wood-court tones.
HSV_LO = np.array([5, 120, 90])
HSV_HI = np.array([20, 255, 255])


def detect_ball(img, min_area=40, max_area=900, min_circularity=0.6, topk=3):
    """Return up to topk candidates [(cx, cy, r, score)] sorted by score."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)  # 1.0 = perfect circle
        if circularity < min_circularity:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r + 1e-6)  # how well the blob fills its circle
        score = circularity * fill
        cands.append((float(cx), float(cy), float(r), float(score)))
    cands.sort(key=lambda t: -t[3])
    return cands[:topk]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clips = get_basketball_clips("val")
    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    picks = []
    for _ in range(args.samples):
        clip = rng.choice(clips)
        info = load_seqinfo(clip)
        frame = rng.randint(1, info["seqlength"])
        picks.append((clip, info["name"], frame))

    for i, (clip, name, frame) in enumerate(picks):
        img = cv2.imread(str(clip / "img1" / f"{frame:06d}.jpg"))
        if img is None:
            continue
        cands = detect_ball(img)
        for rank, (cx, cy, r, score) in enumerate(cands):
            color = (0, 255, 0) if rank == 0 else (0, 200, 255)
            cv2.circle(img, (int(cx), int(cy)), max(int(r) + 4, 8), color, 2)
            cv2.putText(img, f"{score:.2f}", (int(cx) + 8, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"{name} f{frame}  ({len(cands)} cand)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out = OUT_DIR / f"ball_{i:02d}.jpg"
        cv2.imwrite(str(out), img)
    print(f"Wrote {len(picks)} overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
