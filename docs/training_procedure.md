# Detector training procedure — Check-in 2

Neural baseline for the player-detection half of the pipeline: a COCO-pretrained
**YOLOv8n** fine-tuned on the SportsMOT basketball subset. This document records
the data, splits, loss, optimizer and hyperparameters so the run is reproducible.
Tracking (IoU/SORT association) and HOTA/MOTA/IDF1 are a separate step and not
covered here.

> **Status:** numbers below are from the clip-split run `yolov8n_clipsplit`
> (`src/train_detector.py`). The earlier `yolov8n_e60` run is kept for reference
> but its "held-out" metrics leaked (see *Data splits*), so it is superseded.

## Model

- **Architecture:** YOLOv8n (nano), the smallest YOLOv8 detector (~3.2M params,
  6.2 MB checkpoint), chosen to fit comfortably in 8 GB VRAM and train fast.
- **Initialization:** COCO-pretrained `yolov8n.pt` (transfer learning), not trained
  from scratch. The naive comparison baseline (`src/coco_baseline.py`) is this same
  checkpoint with **no** fine-tuning, used as a person detector.
- **Head:** single class (`player`, `single_cls=True`). SportsMOT labels every box
  with one class; there are no ball annotations.

## Data

Built by `src/mot_to_yolo.py`, which converts SportsMOT MOT-format clips to a YOLO
detection dataset.

- **Source:** 30 basketball clips that ship with ground truth (the official `train`
  + `val` splits; the 50 official `test` clips withhold labels and are unusable).
- **Frame sampling:** `--stride 3` keeps every 3rd frame (consecutive frames are
  near-duplicates), reducing redundancy and training time by ~3x.
- **Box conversion:** MOT `(x, y, w, h)` top-left pixel boxes → YOLO normalized
  `cx cy w h`, clipped to the image; boxes with no area after clipping are dropped
  (0 occurred in practice).

| Split | Source videos | Clips | Images (stride 3) | Boxes |
|-------|---------------|-------|-------------------|-------|
| train | 2             | 11    | 3,371             | 28,972 |
| val   | 1             | 4     | 761               | 7,344 |
| test  | 4             | 15    | 4,191             | 39,655 |

## Data splits (and why they changed)

The first run (`yolov8n_e60`) trained on the official train split and used the
official val split for both early stopping **and** final metrics — so the reported
"val" numbers were not held out from model selection.

The split was rebuilt **by source video** to fix this:

- **train** — official train clips, minus one held-out source video.
- **val** — the 4 clips of the held-out source video `v_-6Os86HzwCs`; used only for
  early stopping and checkpoint selection (`best.pt`).
- **test** — all 15 official val clips; **never** seen during training or model
  selection, so these are the only truly held-out metrics.

Splitting by source video (not by clip) matters because clips cut from the same
broadcast share a court, uniforms, players and camera setup — a clip-level split
would leak that into validation. The `v_-6Os86HzwCs` video was chosen for the val
holdout because its boxes are smaller (median area ~9–12k px² vs ~11–19k for the
other two train videos), which makes it closer to the test distribution than a
random holdout would be, so early stopping tracks test-relevant performance.

## Loss

Standard YOLOv8 detection loss, a weighted sum of three terms (defaults kept):

- **Box** (bounding-box regression, CIoU): weight `box = 7.5`
- **Classification** (BCE): weight `cls = 0.5`
- **DFL** (distribution focal loss on box distribution): weight `dfl = 1.5`

Losses are normalized to a nominal batch size `nbs = 64`.

## Optimizer & schedule

- **Optimizer:** `optimizer=auto`, which Ultralytics resolved to **AdamW**
  (`lr0=0.002`, `momentum=0.9`), overriding the SGD-style `lr0=0.01`/`momentum=0.937`
  defaults. Parameter groups: weights with `weight_decay=0.0005`, biases and norm
  weights with no decay.
- **LR schedule:** linear decay (`cos_lr=False`) from `lr0` to `lr0 * lrf`
  (`lrf=0.01`), with a **3-epoch warmup** (`warmup_epochs=3`, `warmup_momentum=0.8`,
  `warmup_bias_lr=0.1`).
- **Mixed precision:** AMP enabled.

## Key hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| epochs (cap)   | 60 |
| early-stopping patience | 15 epochs (on val fitness) |
| image size     | 640 |
| batch size     | 16 |
| seed / deterministic | 0 / true |
| device         | single GPU (RTX 4060 Ti, 8 GB) |

### Augmentation (Ultralytics defaults)

Mosaic `1.0` (disabled for the final 10 epochs, `close_mosaic=10`); HSV jitter
`h=0.015, s=0.7, v=0.4`; `translate=0.1`; `scale=0.5`; horizontal flip `fliplr=0.5`;
random erasing `0.4`; `auto_augment=randaugment`. Vertical flip, rotation, shear,
perspective, mixup and copy-paste are off.

## Running it

```bash
# 1. Build the clip-split YOLO dataset
python src/mot_to_yolo.py --stride 3

# 2. Fine-tune (early stopping on val, final metrics on val + held-out test)
python src/train_detector.py --epochs 60 --name yolov8n_clipsplit

# 3. Naive comparison baseline: COCO person detector, no fine-tuning
python src/coco_baseline.py --split test
```

Ultralytics writes full logs, PR/F1 curves, the confusion matrix and
`weights/best.pt` to `runs/detect/<name>/`; `train_detector.py` copies the key
plots and a `summary.txt` into `outputs/detector/<name>/`.

## Results

### Fine-tuned YOLOv8n (`yolov8n_clipsplit`)

Early-stopped at epoch 14 (best) of 29 run, ~11 min wall time (~22s/epoch incl.
val). Full numbers in `outputs/detector/yolov8n_clipsplit/summary.txt`.

| Metric   | val (early-stop) | test (held-out) |
|----------|------------------|-----------------|
| mAP50    | 0.9148 | 0.9539 |
| mAP50-95 | 0.5795 | 0.6687 |
| precision| 0.9123 | 0.9443 |
| recall   | 0.8458 | 0.8904 |

The held-out **test** split scores *higher* than val here: that is expected, not
a leak. val is a **single** source video (`v_-6Os86HzwCs`) deliberately chosen as
the hardest early-stop set — one court, uniform and camera — so it lower-bounds
performance, while test averages over four videos. Its smaller boxes (median area
~9–12k vs test's ~11k px²) contribute too, but that gap is modest and secondary to
the single- vs four-video effect.

Note the gap between mAP50 (0.954) and mAP50-95 (0.669) on test: players are
detected reliably, but box localization at tight IoU thresholds is only moderate.
This matters downstream — loose boxes weaken IoU-based tracking association.

### Naive COCO baseline (no fine-tuning), test split

Off-the-shelf `yolov8n.pt` as a person detector, same test clips and matcher
(`src/coco_baseline.py --split test`). Output in
`outputs/detector/coco_baseline/summary_test.txt`.

| Metric | COCO baseline | Fine-tuned (test) |
|--------|---------------|-------------------|
| AP50 / mAP50 | 0.6982 | 0.9539 |
| precision @ conf 0.25 | 0.5318 | — |
| recall @ conf 0.25 | 0.8229 | — |
| F1 @ conf 0.25 | 0.6460 | — |

Fine-tuning lifts AP50 by ~25 points. The baseline's weak spot is **precision**
(0.53): a generic person detector fires on referees, bench players and crowd, all
of which SportsMOT labels as non-players. Fine-tuning on the single `player` class
is mostly what removes those false positives.

### Reference: superseded `yolov8n_e60` run (leaked val)

Trained on official train, early-stopped at epoch 37 of 52, ~34 min wall time.
val mAP50 **0.975**, mAP50-95 **0.730**, precision **0.963**, recall **0.933** —
reported here only to show the leaked-split baseline these numbers replace.

## Environment

ultralytics 8.4.162, torch 2.5.1+cu121, Python 3.10, Windows 11, NVIDIA RTX 4060 Ti.
