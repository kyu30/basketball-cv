# basketball-cv

## Problem Framing/Scope

**Task:** This project builds a multi-player tracking system for basketball video, using the basketball subset of the SportsMOT dataset. At its core, the task is to detect every basketball player on a court and maintain a consistent identity (track ID) for each player across frames given raw broadcast-style video frames. It would have to do this through occlusion, fast movement, and camera movement. As an extension, this project adds ball trajectory tracking and uses the combined player \+ ball trajectories to classify short segments into basic action types (pass, dribble, shot attempt). 

**Success Criteria/Goal**: 

* Core: fine-tuned detector/tracker that achieves competitive HOTA/MOTA/IDF1 on the SportsMOT dataset test split, with a clear, data-grounded account of where/why it fails (occlusion, small-scale misses, fast motion, jersey similarity).  
* Extension: ball tracking signal (heuristic-based, upgraded to a learned detector only if justified by EDA evidence) that is good enough to feed a play classifier and a comparison between a rule-based action classifier and a learned sequence model (LSTM/Transformer) on the same trajectory data  
  * Further extension (independent/time-dependent): identify specific play types in a longer segment/possession (pick-and-roll, dribble handoff, motion offense, etc.)  
* The project is successful if it produces an honest, well-evaluated pipeline with clear failure analysis

**Feasibility:** The core problem is scoped to build on an existing and well-annotated benchmark dataset (SportsMOT) rather than requiring custom data collection/annotation or new architecture design. This keeps the neural-fundamentals baseline in Check-in 2 achievable on a realistic timeline. The extension is also scoped down to be a heuristic instead of a learned detector from scratch, and play detection is limited to 3 classes for the main extension instead of a larger taxonomy. 

**Why this problem**: I’m pretty interested in basketball and I’ve wanted to do a deep learning project involving it for a while now. Computer vision is also a field I’ve wanted to learn more about since I didn’t get to dive deep into it during undergrad. 

## Dataset Access \+ Documentation

**Dataset:** SportsMOT (basketball subset), released under a CC BY-NC 4.0 license (non-commercial use only). 

**Source:** MCG-NJU/SportsMOT, hosted on Hugging Face ([https://huggingface.co/datasets/MCG-NJU/SportsMOT](https://huggingface.co/datasets/MCG-NJU/SportsMOT)); originally published at ICCV 2023\. 

**Size:** 240 total video sequences across basketball, football, and volleyball (over 150K frames, 1.6M+ bounding boxes). The basketball subset is selected via the provided basketball.txt split file. Videos are 720P, 25 FPS, collected from NCAA, Olympic, and professional broadcast footage. 

**Format**: MOT challenge format  
Per-clip folders containing:

* img1/ \- sequential JPEG frames  
* gt/ \- ground-truth bounding boxes and track IDs  
* Seqinfo.ini \- per-clip metadata (frame rate, resolution, frame count)

Official train/val/test splits are provided (45/45/150 clips for each split). The conversion script mot\_to\_coco.py is provided by the dataset authors for converting to COCO format if needed for detector fine-tuning

**Access Instructions**:

1. Request access on the Hugging Face dataset page (institute, use-case, license acknowledgment) and wait for approval  
2. Clone/download the dataset from the Hugging Face repo  
3. Filter to basketball-only clips using splits\_txt/basketball.txt  
4. Parse gt/ annotations per clip using the provided conversion utilities or a custom MOT-format parser

## Data Audit / EDA

Full analysis lives in `notebooks/01_eda.ipynb`; summary below. Of the 80 basketball clips listed in `splits_txt/basketball.txt`, only 30 (15 train, 15 val) actually ship with ground truth. The other 50 fall in the `test` split, which SportsMOT withholds labels for (competition holdout). All EDA, and any detector training/validation later, is scoped to those 30 clips: 24,941 frames, 227,673 player bounding boxes, uniformly 25fps at 1280x720.

**Representative samples & summary stats:** Sample frames with ground-truth boxes drawn on them (notebook Section 6) confirm boxes are well-aligned to players across a range of camera angles and crowd densities. Track length (how long a player keeps one ID) has a median of ~719 frames (~29s) with no meaningful short-track noise - identity persistence looks clean at the annotation level. Bounding box area is heavily right-skewed, which is expected since players would be at varying distances from the camera.

**Class balance / key distributions:** There is effectively no class balance question for detection. Every box in `gt.txt` is labeled the same single class (player) and there is no ball annotation anywhere in the dataset (see below). Players-per-frame peaks at exactly 10 (5-on-5), but ~36% of frames show fewer (3-9), which traces to occlusion/partial visibility rather than referees or bench personnel being mislabeled as players. Comparing train vs. val, track length and occlusion rate are close, but bounding-box size is not: val's median box area (~11.1k px²) runs about 24% smaller than train's (~14.7k px²), meaning val clips tend to be shot from further back.

**Obvious artifacts, biases, and likely failure modes:**
- The raw downloaded archive contained macOS AppleDouble metadata files (`._*.jpg`) that doubled naive frame counts under a plain glob; filtered these out and verified the corrected counts against each clip's `seqinfo.ini` (`seqLength`).
- `gt.txt`'s `visibility` and `conf` columns are constant placeholders (always 1) - SportsMOT does not label occlusion directly, so it has to be inferred. Two proxies were built: pairwise IoU between player boxes in the same frame (flags 67.7% of frames above an IoU of 0.3) and track-level frame-gap detection (55.3% of tracks disappear and reappear under the same ID at least once, with a median gap of ~87 frames / ~3.5s when it happens). The frame-gap check is the more direct signal of real re-identification difficulty; the IoU check mostly captures benign close-proximity overlap.
- No ball annotations exist anywhere in SportsMOT's ground truth. The ball-tracking extension will need an independent labeling source (heuristic, synthetic, or a small hand-labeled validation set), not this dataset's `gt.txt`.
- The train/val bounding-box-size gap above is worth remembering once a detector is trained: worse performance on val could reflect a genuinely harder (smaller-object) distribution rather than overfitting.

## Evaluation Plan

**Player Tracking (core):**

* Primary metric: HOTA (Higher Order Tracking Accuracy), balances detection accuracy (DetA) and association/identity accuracy (AssA) into a single interpretable score, is the standard metric in current MOT literature (including the original SportsMOT paper), so we can compare to published baselines  
* Secondary metrics: MOTA, IDF1, included for comparison against older/more widely-cited tracking literature used before HOTA  
* Failure-mode categories to track:  
  * ID switches under occlusion (players overlapping/crossing paths)  
  * Missed detections at small bounding-box scale (players far from camera)  
  * False positives on non-player entities (referees, team personnel, crowd)  
  * Degradation under fast motion/motion blur

**Train/Validation/Test Plan:** Uses SportsMOT’s aforementioned official split on the basketball subset. Validation split is used for hyperparameter tuning and early stopping during detector fine-tuning, test split held out for final reported metrics only

**Extension:**

* Ball tracking: no ground-truth annotations exist in SportsMOT, so evaluation will rely on a manually-labeled held-out sample (subset of frames labeled for validation purposes only, separate from any frames used to fine-tune the a learned detector if built) to compute detection rate/localization error  
* Play classification: F1 score per class (pass, dribble, shot), not raw accuracy, since dribble segments are expected to substantially outnumber pass and shot segments. Comparison is between a rule-based classifier (velocity/proximity heuristics on trajectories) and a learned sequence model (LSTM or small transformer) trained on the same trajectory data

## Initial Direction

Expected neural approach for Check-in 2: Fine-tune YOLOv8 (n or s variant, sized for 8GB VRAM) for player detection on the SportsMOT basketball subset, paired with a tracking algorithm (starting with IoU-based association/SORT) to link detections into consistent player tracks across frames. This constitutes the “real neural network baseline,” since the detector is trained, not used off-the-shelf and results are evaluated with HOTA/MOTA/IDF1 against SportsMOT’s ground truth

[Check-in 1 Video](https://youtu.be/KG7r7ordQW4)


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

## 9\. AI Use

Used Claude to clean up/format code (adding comments, code summaries, etc) and write documentation (excluding check-in writeup) 

[Check-in 1 Video](https://youtu.be/-_PUJXpZOpY)