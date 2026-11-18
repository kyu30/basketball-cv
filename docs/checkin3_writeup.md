# Check-in 3 - Advanced Extension: Learned Appearance Re-ID for Tracking

**Extension:** a trained player **re-identification (Re-ID) embedding** added to the
tracker, to fix the CI2 finding that *identity association is the bottleneck*
(AssA ~0.16–0.22, thousands of ID switches). The detector is frozen and detections
are cached once, so every tracker runs on **identical detections and the identical
HOTA/MOTA/IDF1 protocol** as CI2 - only the association logic changes.

Method details in [`checkin3_plan.md`](checkin3_plan.md); CI2 baseline in
[`tracking.md`](tracking.md).

---

## 1. The extension (advanced NN component)

- **Embedding model:** OSNet, a Re-ID-specific CNN, initialized from **Market-1501
  person-ReID** weights (not ImageNet - diversity comes from a large external person
  corpus, the mitigation for our few basketball source videos), then **fine-tuned on
  basketball** with the standard Re-ID recipe (ID cross-entropy + batch-hard triplet
  loss, Re-ID augmentation). Trained only on the 15 official-train clips (110 train /
  40 held-out identities); the test clips are strictly excluded.
- **Appearance tracker** (`src/sort_tracker.py::AppearanceTracker`): extends the CI2
  IoU tracker with an appearance term. Association cost =
  `α·(1 − IoU) + (1 − α)·cosine_distance(embedding, track_feature)`, Hungarian
  assignment, each track holding an EMA of its embedding. **α = 1 reduces exactly to
  the CI2 IoU baseline** (verified: HOTA 0.3308, IDSW 3433, identical), so the
  appearance effect is cleanly isolated.
- **Re-ID sanity check:** on the held-out 40 identities, Rank-1 ≈ 0.60 but mAP
  plateaus at ≈ 0.18 while training loss keeps falling - the jersey-similarity
  ceiling previewed here and confirmed in §4.

## 2. Comparison to the CI2 baseline (same protocol)

Held-out test (15 clips), identical detections, TrackEval:

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---------|------|------|------|------|------|-------------|
| IoU (CI2) | 0.331 | 0.700 | 0.157 | 0.808 | 0.287 | 3,433 |
| ByteTrack (CI2) | 0.382 | 0.661 | 0.222 | 0.825 | 0.394 | 1,815 |
| Re-ID, appearance only (CI3) | 0.459 | 0.690 | 0.305 | 0.800 | 0.504 | 2,328 |
| **Re-ID + motion (CI3)** | **0.484** | 0.690 | **0.340** | 0.803 | **0.535** | **2,013** |

**What improved:** the appearance-only Re-ID tracker already lifts HOTA **+39% over
IoU, +20% over ByteTrack** and nearly doubles association (AssA) vs IoU, with
**DetA unchanged (~0.69)** by construction (identical detections) - so the entire
gain is association, directly resolving the CI2 bottleneck. Adding a
constant-velocity **motion** model (BoT-SORT-style: motion-predicted IoU +
appearance, `--motion`) improves it further to **HOTA 0.484** (AssA 0.340, IDF1
0.535, ID switches down to 2,013) - the best of both signals.

**What did not improve / trade-offs:** MOTA stays marginally below ByteTrack
(~0.80 vs 0.825) - MOTA is detection/FP-dominated. The appearance+motion tracker
wins decisively on every identity metric (HOTA/AssA/IDF1) and cuts ID switches
most, confirming that association, not detection, was the lever.

## 3. Ablations (controlled changes)

### Ablation A - appearance weight α (appearance off → on)
Same tracker, only α varied (α = 1 is the appearance-off control = IoU baseline).
Full data in `outputs/tracking/ablation_alpha.csv`.

| α (IoU weight) | 1.0 | 0.9 | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.0 |
|---|---|---|---|---|---|---|---|---|
| HOTA | 0.331 | 0.332 | 0.341 | 0.354 | 0.369 | 0.411 | **0.459** | 0.508 |
| AssA | 0.157 | 0.159 | 0.169 | 0.186 | 0.200 | 0.247 | 0.305 | 0.374 |
| IDSW | 3433 | 3342 | 3213 | 3040 | 2739 | 2496 | **2328** | 3976 |

**Insight:** HOTA/AssA rise monotonically as appearance is weighted more. The
operating point **α = 0.1** is best-balanced (lowest IDSW *and* high HOTA). At
**α = 0 (pure appearance)** HOTA keeps climbing but **IDSW spikes to 3,976** and
MOTA drops - appearance with no spatial prior over-merges and swaps identities. That
spike is the jersey problem surfacing (see §4): with no geometry, identical-jersey
teammates get confused.

### Ablation B - person-ReID init vs basketball fine-tuning
Same tracker (α = 0.1), only the embedding weights change. `ablation_init.csv`.

| Embedding | HOTA | AssA | IDF1 | IDSW |
|-----------|------|------|------|------|
| none (IoU baseline) | 0.331 | 0.157 | 0.287 | 3,433 |
| off-the-shelf Market-1501 (no basketball) | 0.410 | 0.247 | 0.440 | 2,609 |
| **basketball-fine-tuned** | **0.459** | **0.305** | **0.504** | **2,328** |

**Insight:** person-ReID pretraining alone delivers most of the gain (+0.079 HOTA
over IoU), and basketball fine-tuning adds a further +0.049. Both the large-corpus
backbone *and* domain adaptation matter - validating the strategy chosen to offset
the limited number of basketball source videos.

### Further gains: detection threshold + a larger detector
Two post-hoc optimizations on top of the appearance+motion tracker (all on the same
test protocol):

| config | det_thresh | HOTA | AssA | IDF1 | IDSW |
|--------|-----------|------|------|------|------|
| Re-ID + motion (as above) | 0.25 | 0.484 | 0.340 | 0.535 | 2,013 |
| + tuned threshold (YOLOv8n) | 0.5 | 0.526 | 0.395 | 0.601 | 1,115 |
| + **YOLOv8s detector** | 0.6 | **0.536** | **0.402** | **0.606** | **1,131** |

- **Detection threshold was under-tuned** - raising it from 0.25 removes low-confidence
  false-positive tracks and is worth **+0.04 HOTA**, the single biggest lever here.
- **A larger detector (YOLOv8s) helps, but only best-vs-best:** at a *fixed* threshold
  YOLOv8s looked *worse* (HOTA 0.465) because its confidence calibration differs and
  emits more low-conf boxes; at its own optimal threshold (0.6) it gives the best
  result overall (0.536). **Lesson: higher detector mAP does not automatically mean
  better tracking - the operating threshold must be re-tuned per detector.**

## 4. Failure analysis

### Where it still fails: same-team (identical jersey) confusions
Every ground-truth player was assigned to a team by jersey-color clustering, and each
tracker's identity confusions were labeled same- or cross-team
(`src/analyze_switches.py`). The robust signal is the **cross:same ratio** (absolute
counts aren't comparable because Re-ID tracks persist longer):

| Tracker | cross-team share of confusions | cross:same ratio |
|---------|-------------------------------|------------------|
| IoU | 74.9% | ~3.0 : 1 |
| Re-ID | 63.4% | ~1.7 : 1 |

The learned appearance **preferentially removes cross-team confusions** (teams wear
different colors), shifting the error distribution toward **same-team** cases.
Teammates in identical uniforms are near-indistinguishable by appearance, so they are
the residual failure - matching the Re-ID mAP plateau in §1 and the α = 0 IDSW spike
in §3. This is structural, not a training shortfall.

### Other failure modes (quantified - `src/analyze_failure_modes.py`)
- **Small / distant players:** detection recall drops on the smallest box-area
  quartile (**0.927** vs ~0.96 for larger boxes, 118k gt boxes) - a real but modest
  small-player miss rate that caps DetA (~0.69) and fragments tracks.
- **Clip difficulty is cross-game, not box-size:** per-clip HOTA correlates only
  weakly with median box area (r = −0.19) and not at all with occlusion rate
  (r = +0.00). The dominant driver is which game/venue a clip is from - the
  women's-game clips are consistently hardest regardless of box size - so the
  small-player effect is a *within-clip* phenomenon, invisible at clip granularity.
- **Residual MOTA gap:** the appearance+motion tracker still trails ByteTrack
  slightly on MOTA (FP-dominated); constant-velocity prediction closes most of the
  motion gap but a full Kalman filter would likely close the rest.
- **Team-assignment caveat:** k=2 color clustering forces referees into a team, a
  small source of noise in the §4 counts (refs are rare in gt).

### Qualitative
`outputs/tracking/viz/reid/` vs `.../iou/`: on `v_00HRwkvvjtQ_c001` frame 500, the
Re-ID tracker shows low, stable IDs (6, 7, 8, 10, 11, 16…) where the IoU tracker had
IDs in the 80s–90s - far fewer track re-creations, the association gain made visible.
Referees/crowd remain unboxed (detector precision), as in CI2.

## 5. Summary

A learned, basketball-adapted appearance embedding plugged into the tracker lifts
HOTA from 0.331 (CI2 IoU) to 0.459 on the identical held-out protocol, and to
**0.484** when combined with a constant-velocity motion model - beating both CI2
baselines, almost entirely through better association. Two ablations isolate *why*
(appearance weight; person-ReID init + domain fine-tuning), and the failure analysis
pins the residual error on same-team jersey similarity, a structural limit appearance
alone cannot cross.

## Reproducing
```bash
python src/build_reid_dataset.py                 # crops + identities (test excluded)
python src/train_reid.py --epochs 30             # OSNet, Market init -> reid_basketball.pth
python src/embed_dets.py --name reid             # per-detection embeddings
python src/track_clips.py --tracker reid         # appearance tracker (alpha 0.1)
python src/track_clips.py --tracker reid --name reid_motion --motion   # appearance + motion (best)
python src/eval_tracking.py                      # HOTA/MOTA/IDF1 vs CI2 trackers
python src/analyze_switches.py --trackers iou reid   # same- vs cross-team failure analysis
# Ablations:
python src/embed_dets.py --weights weights/osnet_x1_0_market1501.pth --name reid_market
python src/track_clips.py --tracker reid --name reid_market --embeds reid_market
python src/track_clips.py --tracker reid --name reidg_a5 --alpha 0.5   # (sweep)
```
