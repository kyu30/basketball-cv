"""Smooth and gap-fill the ball trajectories from the detector.

The ball detector misses ~22% of frames, mostly during fast passes — exactly when
the pass/dribble signal matters. This interpolates short gaps (linear) and lightly
smooths the result, so the possession-transfer feature has a continuous ball to
follow through a pass. Long gaps (ball genuinely off-screen) are left empty.

Input:  data/actions/ball_traj/<clip>.csv
Output: data/actions/ball_traj_smooth/<clip>.csv   (same frame,cx,cy,conf format;
        interpolated rows get conf=0.0)

Usage:
    python src/smooth_ball.py --max-gap 8
"""

import argparse
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "actions" / "ball_traj"
OUT_DIR = PROJECT_ROOT / "data" / "actions" / "ball_traj_smooth"


def smooth_clip(path: Path, max_gap: int, win: int) -> list:
    detected = {}
    for r in csv.DictReader(open(path)):
        if r["cx"]:
            detected[int(r["frame"])] = (float(r["cx"]), float(r["cy"]), float(r["conf"]))
    if not detected:
        return []

    frames = sorted(detected)
    out = {f: (*detected[f][:2], detected[f][2]) for f in frames}  # (cx, cy, conf)
    # linear interpolation across short gaps
    for fa, fb in zip(frames, frames[1:]):
        if 1 < fb - fa <= max_gap:
            xa, ya, _ = detected[fa]
            xb, yb, _ = detected[fb]
            for f in range(fa + 1, fb):
                t = (f - fa) / (fb - fa)
                out[f] = (xa + t * (xb - xa), ya + t * (yb - ya), 0.0)

    # light moving-average smoothing over the (now denser) sequence
    allf = sorted(out)
    xs = np.array([out[f][0] for f in allf])
    ys = np.array([out[f][1] for f in allf])
    if win > 1 and len(allf) >= win:
        k = np.ones(win) / win
        xs = np.convolve(xs, k, mode="same")
        ys = np.convolve(ys, k, mode="same")
    return [(f, round(float(x), 1), round(float(y), 1), out[f][2]) for f, x, y in zip(allf, xs, ys)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gap", type=int, default=8, help="max missing-frame run to interpolate across")
    parser.add_argument("--win", type=int, default=3, help="moving-average window (1 = no smoothing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SRC_DIR.glob("*.csv")):
        rows = smooth_clip(path, args.max_gap, args.win)
        with open(OUT_DIR / path.name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "cx", "cy", "conf"])
            w.writerows(rows)
        print(f"  {path.stem}: {len(rows)} positioned frames (was fewer)")
    print(f"Wrote smoothed trajectories -> {OUT_DIR}")


if __name__ == "__main__":
    main()
