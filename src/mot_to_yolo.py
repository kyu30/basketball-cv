"""Convert SportsMOT basketball clips (MOT format) to a YOLO detection dataset.

Output layout (what ultralytics expects):
    data/yolo/{train,val,test}/images/<clip>_<frame>.jpg
    data/yolo/{train,val,test}/labels/<clip>_<frame>.txt
    data/yolo/sportsmot_basketball.yaml

SportsMOT withholds test-split labels, so the splits are rebuilt from the clips
that do have ground truth:
    train - official train clips, minus the held-out source videos
    val   - official train clips from the held-out source videos (early stopping)
    test  - all official val clips (never seen during training or model selection)
Clips cut from the same source video share a court, uniforms and camera, so the
val holdout is by source video rather than by clip to avoid leakage. The official
val clips come from different source videos than any train clip.

Usage:
    python src/mot_to_yolo.py --stride 3
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml

from data_loading import get_basketball_clips, load_gt, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES = {0: "player"}

# Smallest of the 3 source videos in official train (4 clips, ~15% of frames),
# with smaller boxes than the rest, closer to the test (official val) distribution.
DEFAULT_VAL_VIDEOS = ("v_-6Os86HzwCs",)


def source_video(clip_name: str) -> str:
    """'v_-6Os86HzwCs_c001' -> 'v_-6Os86HzwCs'."""
    return clip_name.rsplit("_c", 1)[0]


def build_splits(val_videos: tuple[str, ...]) -> dict[str, list[Path]]:
    official_train = get_basketball_clips("train")
    known = {source_video(c.name) for c in official_train}
    unknown = set(val_videos) - known
    if unknown:
        raise ValueError(f"--val-videos not in official train: {sorted(unknown)} (have {sorted(known)})")
    return {
        "train": [c for c in official_train if source_video(c.name) not in val_videos],
        "val": [c for c in official_train if source_video(c.name) in val_videos],
        "test": get_basketball_clips("val"),
    }


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Place src at dst. Hardlinks are preferred on Windows (no admin needed, no
    extra disk space); falls back to a copy if linking fails (e.g. cross-volume)."""
    if mode in ("hardlink", "symlink"):
        try:
            if mode == "hardlink":
                os.link(src, dst)
            else:
                os.symlink(src, dst)
            return mode
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def mot_to_yolo_lines(frame_gt, imw: int, imh: int) -> list[str]:
    """MOT (x, y, w, h) top-left pixel boxes -> YOLO normalized cx cy w h,
    clipped to the image. Boxes with no area left after clipping are dropped."""
    lines = []
    for x, y, w, h in frame_gt[["x", "y", "w", "h"]].itertuples(index=False):
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, imw), min(y + h, imh)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / imw
        cy = (y1 + y2) / 2 / imh
        bw = (x2 - x1) / imw
        bh = (y2 - y1) / imh
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def convert_split(split: str, clips: list[Path], stride: int, mode: str) -> dict:
    img_dir = YOLO_ROOT / split / "images"
    lbl_dir = YOLO_ROOT / split / "labels"
    # Start clean so a rerun with a different stride doesn't leave stale frames.
    # Removing hardlinks/symlinks never touches the original SportsMOT frames.
    shutil.rmtree(YOLO_ROOT / split, ignore_errors=True)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    stats = {"clips": 0, "images": 0, "boxes": 0, "empty": 0, "dropped": 0, "modes": set()}
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        gt = load_gt(clip_path)
        by_frame = dict(tuple(gt.groupby("frame")))
        stats["clips"] += 1

        # MOT frames are 1-indexed; keep frames 1, 1+stride, 1+2*stride, ...
        for frame in range(1, info["seqlength"] + 1, stride):
            src = clip_path / "img1" / f"{frame:06d}.jpg"
            if not src.exists():
                continue
            stem = f"{info['name']}_{frame:06d}"
            stats["modes"].add(link_or_copy(src, img_dir / f"{stem}.jpg", mode))

            frame_gt = by_frame.get(frame)
            lines = [] if frame_gt is None else mot_to_yolo_lines(frame_gt, info["imwidth"], info["imheight"])
            if frame_gt is not None:
                stats["dropped"] += len(frame_gt) - len(lines)
            # Empty label files are valid YOLO background images.
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            stats["images"] += 1
            stats["boxes"] += len(lines)
            stats["empty"] += not lines
    return stats


def write_dataset_yaml() -> Path:
    cfg = {
        "path": YOLO_ROOT.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }
    out = YOLO_ROOT / "sportsmot_basketball.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def render_samples(n: int, seed: int) -> list[Path]:
    """Draw boxes read back from the converted YOLO labels (not the MOT source),
    so the render checks the round-trip of the conversion itself."""
    labels = sorted(p for split in ("train", "val", "test") for p in (YOLO_ROOT / split / "labels").glob("*.txt"))
    labels = [p for p in labels if p.stat().st_size > 0]
    rng = random.Random(seed)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    written = []
    for i, lbl in enumerate(rng.sample(labels, min(n, len(labels)))):
        img_path = lbl.parent.parent / "images" / f"{lbl.stem}.jpg"
        img = cv2.imread(str(img_path))
        imh, imw = img.shape[:2]
        for line in lbl.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = int((cx - bw / 2) * imw), int((cy - bh / 2) * imh)
            x2, y2 = int((cx + bw / 2) * imw), int((cy + bh / 2) * imh)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        split = lbl.parent.parent.name
        cv2.putText(img, f"{split}: {lbl.stem}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out = OUTPUTS_DIR / f"yolo_sample_{i}.jpg"
        cv2.imwrite(str(out), img)
        written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=3, help="keep every Nth frame (consecutive frames are near duplicates)")
    parser.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink",
                        help="how to place images; falls back to copy if linking fails")
    parser.add_argument("--val-videos", nargs="+", default=list(DEFAULT_VAL_VIDEOS),
                        help="source-video IDs from official train to hold out as the early-stopping val set")
    parser.add_argument("--samples", type=int, default=3, help="number of converted samples to render to outputs/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    print(f"Converting SportsMOT basketball -> YOLO (stride={args.stride}, mode={args.mode})")
    splits = build_splits(tuple(args.val_videos))
    for split, clips in splits.items():
        s = convert_split(split, clips, args.stride, args.mode)
        videos = sorted({source_video(c.name) for c in clips})
        print(
            f"  {split:5s}: {len(videos)} videos, {s['clips']} clips, {s['images']} images, {s['boxes']} boxes, "
            f"{s['empty']} empty frames, {s['dropped']} boxes dropped (fully off-image), "
            f"placed via {'/'.join(sorted(s['modes']))}"
        )

    yaml_path = write_dataset_yaml()
    print(f"Wrote {yaml_path}")
    for p in render_samples(args.samples, args.seed):
        print(f"Rendered {p}")


if __name__ == "__main__":
    main()
