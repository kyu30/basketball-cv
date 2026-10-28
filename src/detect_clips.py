"""Cache per-frame player detections for the SportsMOT test clips.

Tracking needs every frame (stride 1), unlike detector training which used
stride 3. Detections are run once with the fine-tuned detector and cached so the
IoU tracker and ByteTrack start from the same detection pool (ByteTrack re-runs
the same deterministic detector end-to-end in track_clips.py).

Output (one file per clip), MOT-detection style, one row per detection:
    outputs/tracking/dets/<clip>.txt   ->   frame,x,y,w,h,conf
where (x, y) is the top-left corner in pixels and (w, h) the box size.

Usage:
    python src/detect_clips.py
    python src/detect_clips.py --weights runs/detect/yolov8n_clipsplit/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "yolov8n_clipsplit" / "weights" / "best.pt"
DETS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "dets"


def detect_clip(model: YOLO, clip_path: Path, conf: float, imgsz: int, device: str, batch: int) -> Path:
    info = load_seqinfo(clip_path)
    frames = sorted((clip_path / "img1").glob("*.jpg"))
    DETS_DIR.mkdir(parents=True, exist_ok=True)
    out = DETS_DIR / f"{info['name']}.txt"

    rows = []
    # Chunked prediction: passing a whole clip at once OOMs an 8 GB GPU.
    for start in range(0, len(frames), batch):
        chunk = frames[start : start + batch]
        results = model.predict(source=[str(p) for p in chunk], conf=conf, imgsz=imgsz,
                                device=device, batch=batch, verbose=False)
        for img_path, res in zip(chunk, results):
            frame = int(img_path.stem)  # 000123.jpg -> 123 (MOT is 1-indexed)
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                rows.append(f"{frame},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.4f}")

    out.write_text("\n".join(rows) + ("\n" if rows else ""))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="fine-tuned detector checkpoint")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="low confidence floor; trackers apply their own thresholds on top")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights} - train the detector first.")

    model = YOLO(args.weights)
    clips = get_basketball_clips("val")  # official val = our held-out tracking test set
    print(f"Detecting {len(clips)} test clips (conf>={args.conf}) -> {DETS_DIR}")
    for clip_path in clips:
        out = detect_clip(model, clip_path, args.conf, args.imgsz, args.device, args.batch)
        n = sum(1 for _ in out.open())
        print(f"  {out.name}: {n} detections")


if __name__ == "__main__":
    main()
