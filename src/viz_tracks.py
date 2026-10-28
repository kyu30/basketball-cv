"""Draw tracker output onto clip frames for qualitative inspection.

Reads MOT-format results from track_clips.py and overlays each track's box and ID
(one stable color per ID) on the original frames, saving a few frames per clip to
outputs/tracking/viz/<tracker>/. Useful for spotting ID switches, missed players
and false positives for the failure analysis.

Usage:
    python src/viz_tracks.py --tracker iou --clip v_00HRwkvvjtQ_c001 --frames 1 100 200
    python src/viz_tracks.py --tracker bytetrack --clip v_5ekaksddqrc_c001 --every 150
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"


def color_for_id(tid: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color per track id."""
    h = (tid * 0.61803398875) % 1.0  # golden-ratio hue spacing
    hsv = (int(h * 179), 200, 255)
    bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_results(tracker: str, clip: str) -> dict[int, list[tuple]]:
    path = TRACKING_DIR / tracker / "data" / f"{clip}.txt"
    if not path.exists():
        raise SystemExit(f"No results at {path}")
    by_frame = defaultdict(list)
    for line in path.read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        by_frame[int(f)].append((int(tid), float(x), float(y), float(w), float(h)))
    return by_frame


def find_clip_dir(clip: str) -> Path:
    for c in get_basketball_clips("val"):
        if c.name == clip:
            return c
    raise SystemExit(f"Clip {clip} not in test set")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frames", type=int, nargs="+", help="specific frame numbers to render")
    parser.add_argument("--every", type=int, help="instead, render every Nth frame")
    args = parser.parse_args()

    clip_dir = find_clip_dir(args.clip)
    info = load_seqinfo(clip_dir)
    results = load_results(args.tracker, args.clip)

    if args.frames:
        frames = args.frames
    elif args.every:
        frames = list(range(1, info["seqlength"] + 1, args.every))
    else:
        frames = [1, info["seqlength"] // 2, info["seqlength"]]

    out_dir = TRACKING_DIR / "viz" / args.tracker
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = cv2.imread(str(clip_dir / "img1" / f"{frame:06d}.jpg"))
        if img is None:
            print(f"  skip frame {frame} (no image)")
            continue
        for tid, x, y, w, h in results.get(frame, []):
            c = color_for_id(tid)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(img, str(tid), (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(img, f"{args.tracker}  {args.clip}  f{frame}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out = out_dir / f"{args.clip}_f{frame:06d}.jpg"
        cv2.imwrite(str(out), img)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
