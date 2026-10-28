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

