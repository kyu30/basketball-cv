"""Run a tracker over the SportsMOT test clips and write MOT-format results.

Two trackers, both driven by the same fine-tuned detector at the same confidence
floor, so the comparison isolates the association logic:

    iou        - our pure-IoU tracker (src/sort_tracker.py) on cached detections
                 from detect_clips.py
    bytetrack  - ultralytics ByteTrack (Kalman + two-stage association), run
                 end-to-end; it re-runs the same deterministic detector per frame

Output (MOT challenge format: frame,id,x,y,w,h,conf,-1,-1,-1):
    outputs/tracking/<tracker>/data/<clip>.txt

Usage:
    python src/detect_clips.py          # once, for the iou tracker's inputs
    python src/track_clips.py --tracker iou
    python src/track_clips.py --tracker bytetrack
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from data_loading import get_basketball_clips, load_seqinfo
from sort_tracker import IouTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolov8n_clipsplit" / "weights" / "best.pt"
DETS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "dets"
TRACKING_DIR = PROJECT_ROOT / "outputs" / "tracking"

MOT_ROW = "{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"


def load_cached_dets(clip_name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read detect_clips.py output -> {frame: (xyxy (n,4), confs (n,))}."""
    path = DETS_DIR / f"{clip_name}.txt"
    if not path.exists():
        raise SystemExit(f"No cached detections at {path} - run src/detect_clips.py first.")
    by_frame = defaultdict(list)
    for line in path.read_text().splitlines():
        frame, x, y, w, h, conf = line.split(",")
        by_frame[int(frame)].append((float(x), float(y), float(w), float(h), float(conf)))
    out = {}
    for frame, rows in by_frame.items():
        arr = np.array(rows, dtype=float)
        xyxy = arr[:, :4].copy()
        xyxy[:, 2] += xyxy[:, 0]  # w -> x2
        xyxy[:, 3] += xyxy[:, 1]  # h -> y2
        out[frame] = (xyxy, arr[:, 4])
    return out


def track_iou(clip_path: Path, args) -> list[str]:
    info = load_seqinfo(clip_path)
    dets = load_cached_dets(info["name"])
    tracker = IouTracker(iou_thresh=args.iou_thresh, max_age=args.max_age,
                         min_hits=args.min_hits, det_thresh=args.det_thresh)
    lines = []
    for frame in range(1, info["seqlength"] + 1):
        xyxy, confs = dets.get(frame, (np.zeros((0, 4)), np.zeros(0)))
        for tid, box, conf in tracker.update(xyxy, confs):
            x1, y1, x2, y2 = box
            lines.append(MOT_ROW.format(frame=frame, tid=tid, x=x1, y=y1,
                                        w=x2 - x1, h=y2 - y1, conf=conf))
    return lines


def track_bytetrack(model: YOLO, clip_path: Path, args) -> list[str]:
    frames = sorted((clip_path / "img1").glob("*.jpg"))
    lines = []
    # Per-frame with persist=True keeps track state within the clip; persist=False
    # on the first frame resets it so the next clip starts clean.
    for i, img_path in enumerate(frames):
        res = model.track(source=str(img_path), persist=(i > 0), tracker="bytetrack.yaml",
                          conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        if res.boxes.id is None:
            continue
        frame = int(img_path.stem)
        ids = res.boxes.id.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
            lines.append(MOT_ROW.format(frame=frame, tid=int(tid), x=x1, y=y1,
                                        w=x2 - x1, h=y2 - y1, conf=float(conf)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", choices=["iou", "bytetrack"], required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="detector (bytetrack only)")
    parser.add_argument("--conf", type=float, default=0.1, help="detector conf floor (bytetrack)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    # iou-tracker knobs
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--det-thresh", type=float, default=0.25,
                        help="conf threshold the iou tracker applies to cached detections")
    args = parser.parse_args()

    out_dir = TRACKING_DIR / args.tracker / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights) if args.tracker == "bytetrack" else None
    clips = get_basketball_clips("val")
    print(f"Tracking {len(clips)} clips with '{args.tracker}' -> {out_dir}")
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        if args.tracker == "iou":
            lines = track_iou(clip_path, args)
        else:
            lines = track_bytetrack(model, clip_path, args)
        out = out_dir / f"{info['name']}.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  {info['name']}: {len(lines)} track rows")


if __name__ == "__main__":
    main()
