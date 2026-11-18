"""Export hand-labeled ball frames (data/ball/frames.csv) to a YOLO dataset.

Present frames become single-box ball labels (class 0); 'none' frames become
background negatives (empty label). Split by clip so train/val share no clip.

Output:
    data/ball_yolo/{train,val}/images/<clip>_<frame>.jpg
    data/ball_yolo/{train,val}/labels/<clip>_<frame>.txt
    data/ball_yolo/ball.yaml

Usage:
    python src/build_ball_dataset.py                 # 3 clips held out for val
    python src/build_ball_dataset.py --val-clips 4
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import yaml

from data_loading import get_basketball_clips, load_seqinfo
from mot_to_yolo import source_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALL_DIR = PROJECT_ROOT / "data" / "ball"
FRAMES_CSV = BALL_DIR / "frames.csv"
YOLO_ROOT = PROJECT_ROOT / "data" / "ball_yolo"
IMW, IMH = 1280.0, 720.0


def clip_path(split, name):
    for c in get_basketball_clips(split):
        if c.name == name:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", help="SportsMOT split the labeled clips came from")
    parser.add_argument("--val-clips", type=int, default=3, help="clips held out for detector val (clip split)")
    parser.add_argument("--val-videos", nargs="+", default=None,
                        help="source videos to hold out (video-disjoint split; honest generalization)")
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(FRAMES_CSV)) if r["state"] in ("y", "n")]
    if not rows:
        raise SystemExit("No labeled frames in data/ball/frames.csv yet - run src/label_ball.py first.")

    clips = sorted({r["clip"] for r in rows})
    if args.val_videos:
        val_clips = {c for c in clips if source_video(c) in set(args.val_videos)}
    else:
        val_clips = set(clips[:args.val_clips])  # deterministic clip holdout

    for sub in ("train", "val"):
        shutil.rmtree(YOLO_ROOT / sub, ignore_errors=True)
        (YOLO_ROOT / sub / "images").mkdir(parents=True)
        (YOLO_ROOT / sub / "labels").mkdir(parents=True)

    stats = {"train": [0, 0], "val": [0, 0]}  # [positives, backgrounds]
    for r in rows:
        split = "val" if r["clip"] in val_clips else "train"
        cdir = clip_path(args.split, r["clip"])
        src = cdir / "img1" / f"{int(r['frame']):06d}.jpg"
        stem = f"{r['clip']}_{int(r['frame']):06d}"
        img = cv2.imread(str(src))
        if img is None:
            continue
        cv2.imwrite(str(YOLO_ROOT / split / "images" / f"{stem}.jpg"), img)

        lines = []
        if r["state"] == "y":
            cx, cy, half = float(r["cx"]), float(r["cy"]), float(r["half"])
            x1, y1 = max(cx - half, 0), max(cy - half, 0)
            x2, y2 = min(cx + half, IMW), min(cy + half, IMH)
            lines.append(f"0 {(x1+x2)/2/IMW:.6f} {(y1+y2)/2/IMH:.6f} {(x2-x1)/IMW:.6f} {(y2-y1)/IMH:.6f}")
            stats[split][0] += 1
        else:
            stats[split][1] += 1
        (YOLO_ROOT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    cfg = {"path": YOLO_ROOT.as_posix(), "train": "train/images", "val": "val/images", "names": {0: "ball"}}
    with open(YOLO_ROOT / "ball.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    for split in ("train", "val"):
        p, b = stats[split]
        print(f"  {split:5s}: {p} ball frames + {b} background")
    print(f"Wrote {YOLO_ROOT / 'ball.yaml'} (val clips: {sorted(val_clips)})")


if __name__ == "__main__":
    main()
