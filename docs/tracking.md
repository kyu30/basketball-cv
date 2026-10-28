# Player tracking — Check-in 2

Links the fine-tuned detector's per-frame boxes into player tracks across each
clip, and scores identity preservation with HOTA/MOTA/IDF1. This is the
multi-object-tracking half of the core task; detector training is in
[`training_procedure.md`](training_procedure.md). Ball tracking and play
classification are later extensions and not covered here.

## Setup

- **Detector:** the fine-tuned `yolov8n_clipsplit/best.pt`, unchanged.
- **Test set:** the 15 SportsMOT official-val clips (our held-out tracking test),
  run at **stride 1** (every frame — tracking needs continuity, unlike the stride-3
  detector training). ~12.5k frames total.
- **Detections cached once** (`src/detect_clips.py`) at a low confidence floor
  (conf ≥ 0.1), so both trackers start from the same detection pool. Track-row
  counts per clip match within ~1% between the two trackers, confirming this.

## Trackers compared

| Tracker | Association | Motion model | Low-conf recovery |
|---------|-------------|--------------|-------------------|
| **IoU (ours)** | Hungarian on IoU (`src/sort_tracker.py`) | none (last box = prediction) | no |
| **ByteTrack** | two-stage (high- then low-conf) | Kalman filter | yes |

Both consume the same detector at the same conf floor, so the only thing that
differs is the association logic — the comparison isolates what motion modeling
and low-confidence recovery buy. Our IoU tracker runs offline on cached
detections; ByteTrack runs end-to-end via Ultralytics (re-running the same
deterministic detector per frame).

IoU-tracker hyperparameters: IoU match threshold 0.3, max age 30 frames, min 3
hits to confirm, detection threshold 0.25.

## Results (held-out test, overall)

Computed with TrackEval (`src/eval_tracking.py`); HOTA is the SportsMOT paper's
primary metric. Per-clip numbers in `outputs/tracking/metrics.csv`.

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---------|------|------|------|------|------|-------------|
| **ByteTrack** | **0.382** | 0.661 | **0.222** | **0.825** | **0.394** | **1,815** |
| IoU (ours)    | 0.331 | 0.700 | 0.157 | 0.808 | 0.287 | 3,433 |

**Reading:**

- **Detection is solid; association is the bottleneck.** DetA ≈ 0.66–0.70 (in line
  with the detector's test mAP50-95 of 0.669), but AssA is only 0.16–0.22. HOTA,
  which is the geometric mean of the two, is dragged down by association.
- **ByteTrack nearly halves ID switches** (3,433 → 1,815) and lifts AssA by ~41%
  and IDF1 by 11 points. Its Kalman motion model and low-confidence recovery keep
  identities alive through the brief detection dropouts and crossings that make our
  memoryless IoU tracker spawn a fresh ID.
- **ByteTrack's DetA is slightly *lower*** (0.661 vs 0.700): its track-confirmation
  logic withholds some boxes our IoU tracker emits. It trades a little detection
  recall for much better identity — the right trade for a tracking metric, and MOTA
  (detection-dominated) still comes out ahead.
- **Absolute HOTA (~0.33–0.38) is well below SportsMOT leaderboard numbers (~70+
  HOTA).** Expected: this is the smallest detector (YOLOv8n) with no appearance/
  re-ID model. These are honest baselines, not a tuned system — the point is the
  relative comparison and the failure analysis.

## Qualitative examples

Rendered overlays in `outputs/tracking/viz/<tracker>/` (`src/viz_tracks.py`).
On `v_00HRwkvvjtQ_c001`:

- **Successes:** all five players per team are detected and boxed on every rendered
  frame; **referees, bench and crowd are never boxed**, confirming the fine-tuned
  detector's high precision (the COCO-person baseline would have flagged all of
  them). Isolated players in open court keep a stable ID across hundreds of frames.
- **Failures:** the hard cases are the contested/overlapping clusters (e.g. two
  players boxing out under the basket, or an on-ball defender overlapping the
  handler). When boxes overlap heavily, IoU association becomes ambiguous and both
  trackers are prone to swapping or re-numbering IDs once the players separate —
  the main driver of the high ID-switch counts.

## Failure modes (vs the evaluation plan's four categories)

1. **ID switches under occlusion** — the dominant failure. 1.8k–3.4k switches over
   15 clips; the EDA flagged 67.7% of frames with player-player IoU > 0.3, so
   overlap is pervasive. This is where ByteTrack's motion model helps most and
   where an appearance/re-ID model would help next.
2. **Missed detections at small scale** — contributes to DetA < mAP50; distant
   players (smaller boxes, the val-video weakness noted in training) are missed or
   dropped intermittently, breaking tracks.
3. **False positives on non-players** — largely *not* an issue here: the fine-tuned
   detector's high precision keeps referees/crowd out (visible in the overlays),
   unlike the COCO-person baseline.
4. **Fast motion / motion blur** — fast drives and camera pans move boxes far
   between frames; with no motion model our IoU tracker loses the match when
   displacement exceeds overlap, which ByteTrack's Kalman prediction partly
   recovers.

## Reproducing

```bash
python src/detect_clips.py                     # cache detections (stride 1, conf>=0.1)
python src/track_clips.py --tracker iou        # pure-IoU tracks
python src/track_clips.py --tracker bytetrack  # ByteTrack tracks (slower, GPU)
python src/eval_tracking.py                    # HOTA/MOTA/IDF1 -> outputs/tracking/metrics.csv
python src/viz_tracks.py --tracker iou --clip v_00HRwkvvjtQ_c001 --frames 50 250 500 750
```

Requires TrackEval (`pip install git+https://github.com/JonathonLuiten/TrackEval`);
a NumPy-2 compatibility shim for it lives at the top of `src/eval_tracking.py`.

## Next steps

- Add an appearance/re-ID cue (BoT-SORT) to attack the association bottleneck.
- Tune the IoU tracker (match threshold, max age) — currently default-ish.
- Break HOTA down per-clip against box-size and occlusion rate to quantify the
  small-scale and occlusion failure modes rather than describing them.
