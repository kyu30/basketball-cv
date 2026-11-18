"""Build a player re-identification dataset from SportsMOT ground truth.

Crops every ground-truth player box (strided to avoid near-duplicate frames) from
the 15 official-train clips and labels it with a per-clip identity (clip, track_id).
The 15 test clips (official val) are deliberately excluded — training Re-ID on them
would leak the exact test appearances into the embedding and void the fair tracking
comparison.

A disjoint-identity split holds out one source video as Re-ID validation (for a
Rank-1/mAP sanity metric on identities never seen in training); the rest train.

Output:
    data/reid/images/<identity>/<clip>_<frame>.jpg   (identity = <clip>__<track_id>)
    data/reid/manifest.csv   columns: path,identity,clip,track_id,source_video,split

Usage:
    python src/build_reid_dataset.py                 # stride 10, 128x256 crops
    python src/build_reid_dataset.py --stride 5 --min-height 40
"""

import argparse
import csv
from pathlib import Path

import cv2

from data_loading import get_basketball_clips, load_gt, load_seqinfo
from mot_to_yolo import source_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REID_ROOT = PROJECT_ROOT / "data" / "reid"
IMAGES_DIR = REID_ROOT / "images"

# Source video held out as the Re-ID retrieval (val) set — disjoint identities.
DEFAULT_VAL_VIDEO = "v_-6Os86HzwCs"


def official_train_clips() -> list[Path]:
    """The 15 official-train clips (3 source videos). This is already the full set
    of gt clips that are NOT the test set: get_basketball_clips("val") returns the
    official val clips, which ARE our held-out tracking test set, so they must not
    be included here (that would leak test appearances into the embedding)."""
    return get_basketball_clips("train")


def build(stride: int, size: tuple[int, int], min_height: int, val_video: str) -> list[dict]:
    w_out, h_out = size
    rows = []
    for clip_path in official_train_clips():
        info = load_seqinfo(clip_path)
        clip, imw, imh = info["name"], info["imwidth"], info["imheight"]
        svideo = source_video(clip)
        split = "val" if svideo == val_video else "train"
        gt = load_gt(clip_path)

        for frame, fgt in gt.groupby("frame"):
            if (frame - 1) % stride != 0:  # keep frames 1, 1+stride, ...
                continue
            img = None
            for track_id, x, y, w, h in fgt[["track_id", "x", "y", "w", "h"]].itertuples(index=False):
                x1, y1 = max(int(x), 0), max(int(y), 0)
                x2, y2 = min(int(x + w), imw), min(int(y + h), imh)
                if y2 - y1 < min_height or x2 <= x1:
                    continue
                if img is None:  # lazy-load the frame only when a crop is kept
                    img = cv2.imread(str(clip_path / "img1" / f"{frame:06d}.jpg"))
                    if img is None:
                        break
                identity = f"{clip}__{int(track_id)}"
                out_dir = IMAGES_DIR / identity
                out_dir.mkdir(parents=True, exist_ok=True)
                crop = cv2.resize(img[y1:y2, x1:x2], (w_out, h_out))
                rel = f"images/{identity}/{clip}_{frame:06d}.jpg"
                cv2.imwrite(str(REID_ROOT / rel), crop)
                rows.append({"path": rel, "identity": identity, "clip": clip,
                             "track_id": int(track_id), "source_video": svideo, "split": split})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=10, help="keep every Nth frame (~1 crop/sec at 25fps)")
    parser.add_argument("--size", type=int, nargs=2, default=(128, 256), metavar=("W", "H"))
    parser.add_argument("--min-height", type=int, default=48, help="skip crops shorter than this (px)")
    parser.add_argument("--val-video", default=DEFAULT_VAL_VIDEO, help="source video held out for Re-ID retrieval eval")
    args = parser.parse_args()

    if IMAGES_DIR.exists():
        import shutil
        shutil.rmtree(IMAGES_DIR)  # clean rerun
    IMAGES_DIR.mkdir(parents=True)

    print(f"Building Re-ID dataset (stride={args.stride}, size={tuple(args.size)}, min_h={args.min_height})")
    rows = build(args.stride, tuple(args.size), args.min_height, args.val_video)

    with open(REID_ROOT / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "identity", "clip", "track_id", "source_video", "split"])
        writer.writeheader()
        writer.writerows(rows)

    for split in ("train", "val"):
        srows = [r for r in rows if r["split"] == split]
        ids = {r["identity"] for r in srows}
        vids = {r["source_video"] for r in srows}
        print(f"  {split:5s}: {len(ids):3d} identities | {len(srows):6d} crops | {len(vids)} source video(s)")
    print(f"Wrote {REID_ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
