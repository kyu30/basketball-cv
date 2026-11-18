# Basketball CV — Final Report (WIP)

Multi-player tracking for basketball video (SportsMOT basketball subset), with a
learned-appearance tracking extension and a play/action-classification stretch.
This report consolidates Check-ins 1–3 and the follow-on improvements; detailed
method docs are linked per section.

---

## 1. Problem & data

**Task:** detect every player each frame and maintain a consistent identity across
frames, through occlusion, fast motion and camera movement; as extensions, track the
ball and classify short segments into basic actions.

**Data:** SportsMOT basketball subset. Of 80 clips, only 30 ship with ground truth
(official train+val); the 50-clip test split withholds labels. All work uses the 30
labeled clips. **EDA** (`notebooks/01_eda.ipynb`): 24,941 frames, 227,673 player
boxes; no ball annotations anywhere; 67.7% of frames have player-pair IoU > 0.3
(pervasive occlusion) — which foreshadows the association bottleneck.

**Split discipline:** rebuilt **by source video** so clips from one broadcast can't
leak across train/val/test. Held-out **test = 15 official-val clips**, never seen in
any training or model selection. Details in
[`training_procedure.md`](training_procedure.md).

## 2. Neural baseline (Check-in 2)

**Detector:** YOLOv8n fine-tuned from COCO, single `player` class.
**Tracker:** detections linked by a pure-IoU tracker and by ByteTrack, scored with
TrackEval (HOTA/MOTA/IDF1). See [`tracking.md`](tracking.md).

| Detector (held-out test) | AP50/mAP50 | mAP50-95 |
|--------------------------|-----------|----------|
| COCO zero-shot (no fine-tune) | 0.698 | — |
| **Fine-tuned YOLOv8n** | **0.954** | 0.669 |

Fine-tuning's gain is mostly precision (0.53 → 0.94): the generic person detector
fires on referees/crowd. **Key CI2 finding:** detection is strong (DetA ~0.70) but
**identity association is the bottleneck** (AssA ~0.16–0.22, thousands of ID switches).

## 3. Advanced extension — learned appearance Re-ID (Check-in 3)

Added a trained player **re-identification embedding** (OSNet, Market-1501 person-ReID
init, fine-tuned on basketball) to the association step, on identical detections and
the same test protocol. Two ablations (appearance weight; person-ReID init vs
basketball fine-tuning) and a same-team/cross-team failure analysis. Full detail in
[`checkin3_writeup.md`](checkin3_writeup.md).

**Result:** HOTA 0.331 → 0.459 (appearance) → 0.484 (+ motion), beating both CI2
baselines almost entirely through better association. Failure analysis shows the
residual error is **same-team (identical jersey) confusions** — a structural limit
appearance can't cross (the cross:same confusion ratio drops from ~3.0 to ~1.7 with
Re-ID).

## 4. Follow-on improvements

- **Appearance + motion** (constant-velocity prediction before IoU): HOTA 0.484.
- **Detection-threshold tuning:** the operating threshold was under-tuned (0.25);
  raising it removes low-confidence false-positive tracks → **+0.04 HOTA**.
- **Larger detector (YOLOv8s):** test mAP50 0.954 → 0.961; *at a fixed threshold it
  hurt tracking* (different confidence calibration, more false positives), but at its
  own optimal threshold it gives the best result. **Lesson: higher detector mAP ≠
  better tracking without re-tuning the operating point.**

**Best tracker: HOTA 0.536** (Re-ID + motion + YOLOv8s + tuned threshold).

## 5. Stretch — play/action classifier

Hand-labeled 600 segments (dribble/pass/shot/other) and 303+147 ball frames; trained
a ball detector (no ball GT exists) and compared a rule-based classifier vs an LSTM
on trajectory features. Detail in [`classifier_writeup.md`](classifier_writeup.md).

| method (clip-grouped 5-fold CV) | macro-F1 |
|---------------------------------|----------|
| majority (floor) | 0.147 |
| rule-based | 0.401 |
| LSTM, player-only | 0.459 |
| LSTM, player + ball | 0.516 ± 0.015 |
| + bidirectional + ensemble | 0.555 ± 0.014 |
| **+ feature engineering (ball-absent fix, handler motion)** | **0.563 ± 0.009** |

The ball **fixes shots** (F1 0.45 → 0.75) but **not pass vs dribble**: that boundary
needs the ball tracked *through* fast, motion-blurred flight, where the detector drops
it. Two targeted fixes (possession-transfer features; ball-trajectory smoothing) both
failed for this root cause — an honest, well-characterized limit.

**Ball-detector honesty check:** the ball detector scores mAP50 0.889 on a same-video
split but only **0.496 (recall 0.48) on a video-disjoint holdout** — it does not
generalize to unseen games. The classifier's ball-dependent gains are therefore a
*within-venue* result; they'd degrade on a new game. Same limited-source-diversity
limit that shapes the Re-ID extension.

## 6. Overall — what improved, and by how much

| Stage | Metric | Value |
|-------|--------|-------|
| Detector: COCO zero-shot → fine-tuned | test mAP50 | 0.698 → **0.954** |
| Detector: YOLOv8n → YOLOv8s | test mAP50-95 | 0.669 → **0.691** |
| Tracking: IoU baseline (CI2) | HOTA | 0.331 |
| Tracking: ByteTrack (CI2) | HOTA | 0.382 |
| Tracking: + learned Re-ID (CI3) | HOTA | 0.459 |
| Tracking: + motion | HOTA | 0.484 |
| Tracking: + YOLOv8s + tuned threshold | HOTA | **0.536** |
| **Tracking end-to-end** | **HOTA** | **0.331 → 0.536 (+62%)** |
| Action classifier: majority → rule → LSTM (best) | macro-F1 | 0.147 → 0.401 → **0.563** |

## 7. Limitations & future work

- **Same-team identity** (identical jerseys) is the residual tracking failure — an
  appearance+motion limit; temporal/transformer association could help.
- **Pass vs dribble** is bounded by ball-tracking reliability through fast motion.
- **Small/distant players**: smallest-quartile detection recall 0.927 vs ~0.96.
- **Action classifier is feature-limited, not data-limited:** scaling 600 → 800
  labeled segments left macro-F1 flat (0.563 → 0.554 ± 0.009). The ceiling is what
  player+ball trajectories reveal, not sample size.
- **Domain diversity:** both the ball detector (0.889 within-venue → 0.496 cross-venue)
  and Re-ID are limited by few source games; more games would improve generalization.
- **Play-type classification (pick-and-roll etc.) — attempted, not viable:** a
  45-window labeling pilot (`src/label_plays.py`) found footage is transition-dominated
  (53%) and half-court play types are each only ~9% — collecting enough would need
  ~450 windows, more than the ~15 clips contain. A data-limitation finding: 15 clips
  don't support fine-grained play typing. Effort redirected to the action classifier.
- **Jersey-number recognition — investigated, not viable:** it would target the
  same-team ID failure directly (numbers are unique per player where appearance isn't),
  but an OCR legibility check (`src/jersey_ocr_check.py`) over 300 player crops found a
  confident-read rate of only **~12%** (17% even on the largest players) at this 720p
  broadcast resolution. Numbers are mostly illegible and occluded exactly when needed,
  so the build cost (a number recognizer + its own labeling) is not justified by the
  likely payoff here. A higher-resolution source would change this.

## 8. Reproducing & artifacts

Per-stage commands are in each linked doc. Key outputs: `outputs/detector/*`,
`outputs/tracking/metrics.csv`, `outputs/tracking/viz/*`, `data/actions/*`.
Environment: ultralytics 8.4.162, torch 2.5.1+cu121, torchreid, TrackEval, RTX 4060 Ti.
