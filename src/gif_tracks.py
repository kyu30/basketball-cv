"""Render an animated GIF of tracker output over a clip segment.

With two trackers it shows them side-by-side (same frames, different IDs), which
makes the identity-stability difference visible in motion — e.g. IoU churning IDs
vs Re-ID+motion holding them. Good footage for the final video.

Usage:
    python src/gif_tracks.py --clip v_00HRwkvvjtQ_c001 --start 400 --length 90 --trackers iou reid_motion
    python src/gif_tracks.py --clip v_5ekaksddqrc_c001 --trackers reid_motion   # single panel
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_loading import load_seqinfo
from viz_tracks import color_for_id, load_results, find_clip_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "tracking" / "viz"


def draw(img, boxes, label, panel_w):
    img = img.copy()
    for tid, x, y, w, h in boxes:
        c = color_for_id(tid)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
        cv2.putText(img, str(tid), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (panel_w, int(h0 * panel_w / w0)))
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--trackers", nargs="+", default=["iou", "reid_motion"])
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--length", type=int, default=90, help="frames of clip to cover")
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth frame (smaller GIF)")
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    clip_dir = find_clip_dir(args.clip)
    info = load_seqinfo(clip_dir)
    results = {t: load_results(t, args.clip) for t in args.trackers}

    frames = []
    for f in range(args.start, min(args.start + args.length, info["seqlength"] + 1), args.stride):
        img = cv2.imread(str(clip_dir / "img1" / f"{f:06d}.jpg"))
        if img is None:
            continue
        panels = [draw(img, results[t].get(f, []), f"{t}  f{f}", args.panel_width) for t in args.trackers]
        combo = np.hstack(panels)
        frames.append(Image.fromarray(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{'_vs_'.join(args.trackers)}_{args.clip}_f{args.start}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"Wrote {out} ({len(frames)} frames, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
