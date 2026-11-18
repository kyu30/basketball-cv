"""Fine-tune a small YOLO ball detector on the hand-labeled ball dataset.

The ball is small and fast, so this trains at a higher image size than the player
detector (default 1280) to keep small objects resolvable. Single class (ball).
Results land in runs/detect/<name>/; key plots + a summary copy to outputs/ball_det/.

Run src/build_ball_dataset.py first.

Usage:
    python src/train_ball_detector.py --epochs 80 --name ball_yolov8n
    python src/train_ball_detector.py --epochs 3 --name ball_smoke      # smoke test
"""

import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "ball_yolo" / "ball.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "ball_det"
KEY_ARTIFACTS = ["results.png", "results.csv", "BoxPR_curve.png", "BoxF1_curve.png",
                 "confusion_matrix.png", "val_batch0_pred.jpg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280, help="large, since the ball is a small object")
    parser.add_argument("--batch", type=int, default=8, help="smaller batch to fit the larger imgsz in 8GB")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="ball_yolov8n")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("data/ball_yolo/ball.yaml missing - run src/build_ball_dataset.py first.")

    model = YOLO(args.model)
    start = time.time()
    model.train(data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, patience=args.patience,
                project=str(RUNS_DIR), name=args.name, exist_ok=True, single_cls=True, plots=True)
    elapsed = time.time() - start
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch,
                                  device=args.device, project=str(RUNS_DIR), name=f"{args.name}_val", exist_ok=True)

    dest = OUTPUTS_DIR / args.name
    dest.mkdir(parents=True, exist_ok=True)
    for fn in KEY_ARTIFACTS:
        if (run_dir / fn).exists():
            shutil.copy2(run_dir / fn, dest / fn)
    summary = (f"run: {run_dir}\nwall: {elapsed:.0f}s\nbest: {best}\n"
               f"val mAP50: {metrics.box.map50:.4f}\nval mAP50-95: {metrics.box.map:.4f}\n"
               f"val precision: {metrics.box.mp:.4f}\nval recall: {metrics.box.mr:.4f}\n")
    (dest / "summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
