# Check-in 2: Neural Baseline & Evaluation

---

## 1\. Neural/Naive Baseline

The core task is two stages: **detect every player in each frame**, then **link detections into consistent identities across frames**.

- **Neural model (required baseline):** YOLOv8n fine-tuned from COCO weights into a single-class (`player`) detector. YOLOv8n is the nano variant (smallest/fastest), comfortably trained on my 8 GB GPU.  
- **Tracker:** detections are linked by (a) **my own pure-IoU tracker** (SORT without the Kalman filter) and (b) **ByteTrack** (Kalman motion \+ two-stage association), compared head-to-head on identical detections.  
- **Naive baseline (optional, done):** off-the-shelf COCO YOLOv8n used as a "person" detector with no fine-tuning, to measure what fine-tuning buys.

## 2\. Data & splits

Of the 80 basketball clips, only 30 ship with labels (train/val) while the 50 test clips withhold labels. I rebuilt usable splits by source video so clips from the same broadcast can't leak across splits:

| Split | Role | Videos | Clips | Frames (stride 3\) |
| :---- | :---- | :---- | :---- | :---- |
| train | fit weights | 2 | 11 | 3,371 |
| val | early stopping / checkpoint | 1 | 4 | 761 |
| test | held-out metrics only | 4 | 15 | 4,191 |

The first run used the official val split for both early stopping and final metrics, so they weren’t actually held out. I re-split by source video, holding out the smallest-box video for validation (closest to the test distribution) so early stopping tracks test-relevant performance, and reserving all 15 official-val clips as the unseen test set.

## 3\. Training procedure (summary)

- **Init:** COCO-pretrained `yolov8n.pt` (transfer learning), single class.  
- **Loss:** YOLOv8 detection loss — box (CIoU, w=7.5) \+ classification (BCE, w=0.5)  
  + DFL (w=1.5).  
- **Optimizer:** AdamW (Ultralytics auto-selected), lr 0.002, 3-epoch warmup, linear decay, AMP mixed precision.  
- **Key settings:** 640px images, batch 16, early stopping (patience 15), seed 0\.  
- **Run:** early-stopped at epoch 14, \~11 min wall time on an RTX 4060 Ti.

Full hyperparameters and augmentation in `training_procedure.md`.

## 4\. Results (held-out test set)

### Detector: fine-tuned vs naive COCO baseline

| Metric | COCO baseline (no fine-tune) | Fine-tuned YOLOv8n |
| :---- | :---- | :---- |
| AP50 / mAP50 | 0.698 | **0.954** |
| mAP50-95 | \- | 0.669 |
| precision @ 0.25 | 0.532 | **0.944** |
| recall @ 0.25 | 0.823 | 0.890 |

Fine-tuning improves AP50 by \~25 points. The baseline's weakness is **precision** (0.53): a generic person detector fires on referees, bench and crowd, which SportsMOT labels as non-players. Fine-tuning on the single `player` class is mostly what removes those false positives.

### Tracking: HOTA / MOTA / IDF1 (TrackEval)

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **ByteTrack** | **0.382** | 0.661 | **0.222** | **0.825** | **0.394** | **1,815** |
| IoU (ours) | 0.331 | 0.700 | 0.157 | 0.808 | 0.287 | 3,433 |

**The main finding: detection is solid, identity association is the bottleneck.** DetA ≈ 0.66-0.70 but AssA is only 0.16-0.22, and HOTA (the balance of the two) is dragged down by association. ByteTrack's motion model and low-confidence recovery **nearly halve ID switches** and raise AssA \~41% and IDF1 by 11 points with a small DetA cost (its confirmation logic withholds a few boxes).

Absolute HOTA (\~0.33-0.38) is far below SportsMOT leaderboard numbers (\~70+ HOTA), which is expected for the smallest detector with no appearance/re-ID model. This is a good baseline to start from and lays out what needs to be improved here.

## 5\. Qualitative examples

Annotated frames in `outputs/tracking/viz/`:

- **Successes:** all players detected and boxed; referees, bench and crowd are never boxed (the fine-tuned precision showing); isolated players keep stable IDs over hundreds of frames.  
- **Failures:** contested/overlapping clusters (players boxing out, on-ball defense) — heavy box overlap makes IoU association ambiguous and both trackers swap or renumber IDs when players separate.

## 6\. Failure analysis (per the evaluation plan's four categories)

1. **ID switches under occlusion**: dominant failure; EDA flagged 67.7% of frames with player-player IoU \> 0.3. Where ByteTrack helps most, re-ID would help next.  
2. **Missed small/far players**: contributes to DetA \< mAP50; distant (small-box) players are dropped intermittently, breaking tracks.  
3. **False positives on non-players**: largely a non-issue after fine-tuning (high precision), unlike the COCO baseline.  
4. **Fast motion / blur**: large inter-frame displacement breaks IoU matches; ByteTrack's Kalman prediction partly recovers these.

## 7\. Limitations & next steps

- **Limitations:** nano detector \+ no appearance model → low absolute HOTA; tracker hyperparameters are near-default; failure analysis is currently descriptive.  
- **Next:** add appearance/re-ID (BoT-SORT) to attack association; tune the IoU tracker; quantify HOTA per-clip against box-size and occlusion rate; then begin the ball-tracking and play-classification extensions.

## 8\. Reproducing

python src/mot\_to\_yolo.py \--stride 3                 \# build clip-split dataset

python src/train\_detector.py \--epochs 60 \--name yolov8n\_clipsplit

python src/coco\_baseline.py \--split test             \# naive baseline

python src/detect\_clips.py                           \# cache detections for tracking

python src/track\_clips.py \--tracker iou

python src/track\_clips.py \--tracker bytetrack

python src/eval\_tracking.py                          \# HOTA/MOTA/IDF1

Env: ultralytics 8.4.162, torch 2.5.1+cu121, TrackEval (from GitHub), RTX 4060 Ti.