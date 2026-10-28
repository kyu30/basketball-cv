"""Fine-tune a YOLOv8 player detector on the SportsMOT basketball YOLO dataset.

Run src/mot_to_yolo.py first. Results land in runs/detect/<name>/ (loss and PR
curves, results.csv, weights/best.pt); key plots and metrics are copied to outputs/.

Usage:
    python src/train_detector.py --epochs 3 --name smoke     # smoke test
    python src/train_detector.py --epochs 40 --name yolov8n_full
"""

import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "yolo" / "sportsmot_basketball.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Plot names vary slightly across ultralytics versions (PR_curve vs BoxPR_curve).
KEY_ARTIFACTS = [
    "results.png",
    "results.csv",
    "PR_curve.png",
    "BoxPR_curve.png",
    "F1_curve.png",
    "BoxF1_curve.png",
    "confusion_matrix.png",
    "val_batch0_pred.jpg",
]


def copy_key_artifacts(run_dir: Path, name: str) -> list[Path]:
    dest = OUTPUTS_DIR / "detector" / name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in KEY_ARTIFACTS:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
            copied.append(dest / fname)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n.pt", help="init weights (COCO-pretrained by default)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience on val fitness")
    parser.add_argument("--name", default="yolov8n")
    args = parser.parse_args()

    model = YOLO(args.model)
    start = time.time()
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        single_cls=True,
        plots=True,
    )
    elapsed = time.time() - start
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"

    # Re-validate best.pt on both splits. val drove early stopping/checkpoint
    # selection, so test is the only truly held-out number; report both.
    best_model = YOLO(str(best))
    splits = {}
    for split in ("val", "test"):
        splits[split] = best_model.val(
            data=str(DATA_YAML), split=split, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(RUNS_DIR), name=f"{args.name}_{split}",
            exist_ok=True,
        )

    copied = copy_key_artifacts(run_dir, args.name)
    epochs_run = len((run_dir / "results.csv").read_text().strip().splitlines()) - 1
    summary = (
        f"run: {run_dir}\n"
        f"epochs completed: {epochs_run}\n"
        f"wall time: {elapsed:.0f}s ({elapsed / max(epochs_run, 1):.0f}s/epoch incl. val)\n"
        f"best.pt: {best} ({best.stat().st_size / 1e6:.1f} MB)\n"
        f"{'metric':<12}{'val':>10}{'test (held-out)':>18}\n"
        f"{'mAP50':<12}{splits['val'].box.map50:>10.4f}{splits['test'].box.map50:>18.4f}\n"
        f"{'mAP50-95':<12}{splits['val'].box.map:>10.4f}{splits['test'].box.map:>18.4f}\n"
        f"{'precision':<12}{splits['val'].box.mp:>10.4f}{splits['test'].box.mp:>18.4f}\n"
        f"{'recall':<12}{splits['val'].box.mr:>10.4f}{splits['test'].box.mr:>18.4f}\n"
    )
    (OUTPUTS_DIR / "detector" / args.name / "summary.txt").write_text(summary)
    print("\n" + summary)
    print("Copied to outputs/:", *[p.name for p in copied])


if __name__ == "__main__":
    main()
