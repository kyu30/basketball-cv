"""Run the trained ball detector over every clip and cache a ball trajectory.

For each frame, keeps the single highest-confidence ball detection (the ball is
singular). Output feeds ball features into the action classifier.

Output (one row per frame; blank cx/cy when no ball found):
    data/actions/ball_traj/<clip>.csv   ->   frame,cx,cy,conf

Usage:
    python src/detect_ball_clips.py
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "ball_yolov8n" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "data" / "actions" / "ball_traj"


def detect_clip(model, clip_path, conf, imgsz, device, batch):
    info = load_seqinfo(clip_path)
    frames = sorted((clip_path / "img1").glob("*.jpg"))
    rows = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        results = model.predict(source=[str(p) for p in chunk], conf=conf, imgsz=imgsz,
                                device=device, batch=batch, verbose=False)
        for img_path, res in zip(chunk, results):
            frame = int(img_path.stem)
            if len(res.boxes) == 0:
                rows.append((frame, "", "", ""))
                continue
            confs = res.boxes.conf.cpu().numpy()
            best = confs.argmax()
            x1, y1, x2, y2 = res.boxes.xyxy.cpu().numpy()[best]
            rows.append((frame, round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1), round(float(confs[best]), 4)))
    return info["name"], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Ball weights not found: {args.weights} - train src/train_ball_detector.py first.")

    model = YOLO(args.weights)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = get_basketball_clips("val")
    print(f"Detecting ball in {len(clips)} clips -> {OUT_DIR}", flush=True)
    for clip_path in clips:
        name, rows = detect_clip(model, clip_path, args.conf, args.imgsz, args.device, args.batch)
        with open(OUT_DIR / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "cx", "cy", "conf"])
            w.writerows(rows)
        hit = sum(1 for r in rows if r[1] != "")
        print(f"  {name}: ball in {hit}/{len(rows)} frames ({100*hit/len(rows):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
