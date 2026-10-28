"""Naive non-fine-tuned baseline: off-the-shelf COCO-pretrained YOLOv8n run as a
"person" detector on the SportsMOT basketball test split.

This is the comparison point for the fine-tuned detector. The model is used with
zero training on basketball; it only keeps COCO's `person` class (index 0), which
is the closest off-the-shelf analogue to our single `player` class. Metrics are
computed against the same YOLO-format ground truth the fine-tuned model is scored
on, with a self-contained VOC/COCO-style matcher (AP at IoU 0.50), so the two
numbers are directly comparable.

Usage:
    python src/coco_baseline.py                 # test split, yolov8n.pt
    python src/coco_baseline.py --split val --conf 0.25
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from iou import iou_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "detector" / "coco_baseline"

COCO_PERSON_CLASS = 0
IOU_THRESH = 0.50


def load_gt_xyxy(label_path: Path, imw: int, imh: int) -> np.ndarray:
    """Read a YOLO label file (normalized cx cy w h) into pixel (x1, y1, x2, y2)."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=float)
    boxes = []
    for line in label_path.read_text().splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        x1 = (cx - bw / 2) * imw
        y1 = (cy - bh / 2) * imh
        x2 = (cx + bw / 2) * imw
        y2 = (cy + bh / 2) * imh
        boxes.append((x1, y1, x2, y2))
    return np.array(boxes, dtype=float)


def match_frame(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> np.ndarray:
    """Greedy per-frame matching (preds already sorted by confidence desc).
    Returns a bool array over preds: True = true positive at IOU_THRESH."""
    is_tp = np.zeros(len(pred_xyxy), dtype=bool)
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return is_tp
    ious = iou_matrix(pred_xyxy, gt_xyxy)
    gt_taken = np.zeros(len(gt_xyxy), dtype=bool)
    for i in range(len(pred_xyxy)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= IOU_THRESH and not gt_taken[j]:
            is_tp[i] = True
            gt_taken[j] = True
    return is_tp


def average_precision(confs: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    """COCO-style all-point AP from per-prediction confidence + TP flags."""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-confs)
    tps = tps[order]
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(~tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    # Monotonically decreasing precision envelope, then integrate over recall.
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained weights (not fine-tuned)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16, help="prediction batch size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold for the reported precision/recall/F1")
    args = parser.parse_args()

    img_dir = YOLO_ROOT / args.split / "images"
    lbl_dir = YOLO_ROOT / args.split / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images under {img_dir} - run src/mot_to_yolo.py first.")

    model = YOLO(args.model)
    all_conf, all_tp = [], []  # across all frames, for AP (low conf floor)
    n_gt_total = 0
    tp_at, fp_at, fn_at = 0, 0, 0  # at args.conf, for P/R/F1

    # conf=0.001 floor so the PR curve (hence AP) sees the full recall range.
    # Predict in fixed-size chunks; passing all frames at once OOMs an 8 GB GPU.
    for start in range(0, len(images), args.batch):
        chunk = images[start : start + args.batch]
        results = model.predict(source=[str(p) for p in chunk], classes=[COCO_PERSON_CLASS],
                                conf=0.001, imgsz=args.imgsz, device=args.device,
                                batch=args.batch, verbose=False)
        for img_path, res in zip(chunk, results):
            imh, imw = res.orig_shape
            gt = load_gt_xyxy(lbl_dir / f"{img_path.stem}.txt", imw, imh)
            n_gt_total += len(gt)

            confs = res.boxes.conf.cpu().numpy()
            pred = res.boxes.xyxy.cpu().numpy()
            order = np.argsort(-confs)
            confs, pred = confs[order], pred[order]

            is_tp = match_frame(pred, gt)
            all_conf.append(confs)
            all_tp.append(is_tp)

            # Metrics at the operating threshold.
            keep = confs >= args.conf
            tp_at += int(is_tp[keep].sum())
            fp_at += int((~is_tp[keep]).sum())
            fn_at += len(gt) - int(is_tp[keep].sum())

    all_conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    all_tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    ap50 = average_precision(all_conf, all_tp, n_gt_total)

    precision = tp_at / max(tp_at + fp_at, 1)
    recall = tp_at / max(tp_at + fn_at, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        f"COCO zero-shot baseline (no fine-tuning)\n"
        f"model: {args.model} (person class only)\n"
        f"split: {args.split} ({len(images)} images, {n_gt_total} gt boxes)\n"
        f"AP50: {ap50:.4f}\n"
        f"at conf>={args.conf}:  precision: {precision:.4f}  recall: {recall:.4f}  F1: {f1:.4f}\n"
    )
    (OUTPUTS_DIR / f"summary_{args.split}.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
