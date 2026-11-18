# Check-in 3 Plan — Advanced Extension & Final Product

**Primary (polished, graded spine):** learned **appearance Re-ID** to improve player
tracking — attacks the CI2 finding that identity association is the bottleneck.
**Stretch (for the final product):** a learned **play/action classifier** on
trajectories (the README's headline extension).

The guiding principle, from the rubric: 40 of 60 points are *advanced extension*
(20) + *fair comparison to the CI2 baseline on the same evaluation protocol* (20).
So the primary extension improves the **core task** and is measured with the **exact
same HOTA/MOTA/IDF1 harness, detector, and test clips** as CI2. Only the association
logic changes — a genuinely controlled comparison.

---

## Part A — Primary extension: Appearance Re-ID for tracking

### A1. The idea
CI2 showed detection is strong (DetA ~0.7) but association is weak (AssA ~0.16–0.22,
1.8k–3.4k ID switches), because the trackers use only geometry (IoU) and motion
(Kalman). The extension adds a **learned appearance embedding**: a CNN trained with
metric learning to map each player crop to a vector where the *same* player is close
and *different* players are far. The tracker then associates on appearance **and**
motion, so identities survive occlusions and crossings that break a geometry-only
tracker.

The neural model stays central: the frozen CI2 detector feeds boxes; the new trained
component is the Re-ID embedding network.

### A2. Re-ID training data
- Extract player crops from **all 15 official-train clips** (train + the held-out
  detector-val clip) using ground-truth boxes — the detector's train/val split is
  irrelevant to Re-ID, which has no early-stopping-on-detector constraint. **The 15
  test clips stay strictly out** (training on them would leak the exact test
  appearances into the embedding and void the fair comparison).
- **Identity label = (clip, track_id):** every ground-truth track is one identity
  (150 identities across 3 source videos; ~87k+22k boxes, long tracks).
- **Disjoint-identity retrieval split** for the sanity metric: hold out one source
  video (`v_-6Os86HzwCs`, 40 ids) as the Re-ID *val* (query/gallery); train on the
  other two videos' 110 ids. A `--use-all` flag then trains the *deployed* embedding
  on all 150 ids for maximum appearance diversity before test inference.
- Resize crops to standard Re-ID size (128×256). Keep ~1 crop/sec (stride ~10) to
  avoid near-duplicate frames.
- **Known hard case, by design:** teammates wear identical jerseys, so many
  different-identity crops look nearly the same — the central challenge and source
  of the failure analysis (A7).
- New script: `src/build_reid_dataset.py`.

#### A2.1 Addressing limited source diversity
Only ~3 games'/videos' worth of jerseys exist in the labeled data. Three
complementary mitigations (see also A3):
1. **Person-ReID-pretrained backbone** (A3) — the main fix; diversity comes from a
   large external person-ReID corpus, not our few videos.
2. **Use all 15 official-train clips** (above) — free, 150 ids over 3 videos.
3. **Standard Re-ID augmentation** (A3) — random erasing, color jitter, flip, crop.
Softening nuance: tracking association is always **within one clip** (separate the
10 players present, fixed teams), so cross-game generalization matters mainly for
whether *test* gains hold — which is exactly what the pretrained backbone secures.

### A3. Re-ID model & training
- Backbone: initialize from a **person-ReID-pretrained** network (OSNet or ResNet
  trained on Market-1501 / MSMT17), **not** ImageNet — this is where appearance
  diversity enters the representation. Add a 256-d embedding head. (Via `torchreid`
  for the pretrained weights, or a manual weight download; accepts the heavier dep
  for the diversity gain.)
- **Augmentation:** random erasing, color jitter, horizontal flip, random crop.
- Loss: standard Re-ID recipe — **ID cross-entropy + triplet loss** (batch-hard).
- Trainer: `src/train_reid.py`, on the 4060 Ti.
- **Sanity metric before integration:** Rank-1 / mAP retrieval on the held-out
  `v_-6Os86HzwCs` identities, so we know the embedding works on its own (and can
  quantify how much jersey similarity caps it).
- The **off-the-shelf (no fine-tune) vs basketball-fine-tuned** contrast is reused
  directly as Ablation 2 (A6).

### A4. Appearance-augmented tracker
Extend the CI2 pure-IoU tracker into an **appearance tracker**
(`src/sort_tracker.py` → add `AppearanceTracker`):
- Each detection gets an embedding (Re-ID net run on its crop).
- Association cost = `α·(1 − IoU) + (1 − α)·cosine_distance(embedding, track_feature)`,
  solved with Hungarian assignment (same as CI2).
- Each track keeps an **EMA of its embedding** so appearance is robust to a single
  bad frame.
- Embeddings are cached per detection (`src/embed_dets.py`, reusing the CI2 cached
  detections) so tracking stays fast and reproducible.
- Output to `outputs/tracking/reid/data/<clip>.txt` — **`eval_tracking.py`
  auto-discovers it**, no harness changes.

*Fallback path if the custom tracker underperforms:* ultralytics **BoT-SORT** with
our trained Re-ID weights, same cached detections. Keep as a backup, not the main
deliverable (less "we built it").

### A5. Comparison to CI2 (rubric criterion 2 — 20 pts)
Same frozen detector, same cached detections, same 15 held-out test clips, same
TrackEval protocol. One table, one story:

| Tracker | Assoc. signal | From |
|---------|---------------|------|
| IoU (ours) | geometry only | CI2 |
| ByteTrack | geometry + motion | CI2 |
| **Appearance (ours)** | geometry + motion + **learned appearance** | **CI3** |

Report HOTA / DetA / AssA / MOTA / IDF1 / IDSW, and state **what improved and what
didn't** — expect AssA/IDF1 up and ID switches down *if* appearance helps; be honest
if gains are modest (jersey similarity caps it).

### A6. Ablations (rubric criterion 3 — 10 pts) — do both, both are cheap
- **Ablation 1 — appearance off vs on:** set α=1 (pure IoU = the CI2 baseline)
  vs α<1 (motion+appearance). Isolates exactly what appearance contributes.
- **Ablation 2 — off-the-shelf vs fine-tuned Re-ID:** ImageNet-pretrained ResNet18
  features (no basketball training) vs our basketball-fine-tuned embedding, tracker
  otherwise identical. Isolates the value of domain adaptation. Ties directly to the
  jersey-similarity story.
- *(Optional 3)* α sweep or embedding-EMA momentum sweep for a sensitivity curve.

### A7. Failure analysis (rubric criterion 4 — 10 pts) — the killer analysis
Categorize every ID switch by whether the confused pair is **same-team or
cross-team** (teams inferred by jersey color / embedding cluster). Hypothesis:
appearance Re-ID cuts **cross-team** confusions (different jerseys) but barely helps
**same-team** ones (identical jerseys). Quantify it; show qualitative frames of each
with `src/viz_tracks.py`. Plus the usual: occlusion, motion blur degrading
embeddings, small-scale crops. This turns a possibly-modest HOTA gain into a
genuinely insightful result.

---

## Part B — Stretch: play / action classifier (for the final product)

The README's headline extension. Clearly scoped and **dependency-flagged** — attempt
only once Part A is polished.

### B1. Honest dependencies (the gating costs)
- **No action labels exist in SportsMOT** → must **hand-label** a small dataset of
  short segments. This is the real cost; budget for it explicitly.
- **Pass/shot need the ball**, which isn't tracked and isn't annotated. Options:
  - **Heuristic ball tracker first** (color/size/motion blob near players), per the
    README plan — or a tiny ball detector fine-tuned on a few hundred hand-labeled
    frames if the heuristic is too noisy.
  - Or scope the first version to **ball-free classes** derivable from player motion
    alone, then add ball-dependent classes once the ball signal exists.

### B2. Minimal viable version
1. `src/track_to_trajectories.py` — convert tracker output (+ ball if available) to
   per-segment trajectory features (velocities, spacing, proximity).
2. Hand-label ~150–300 short segments into 3 classes (start: **dribble / pass /
   shot**, or a ball-free set if ball tracking slips).
3. Two classifiers on the same features: **rule-based** (velocity/proximity
   heuristics) vs **learned LSTM / small Transformer** (`src/train_action_model.py`).
4. Evaluate with **per-class F1** on a held-out segment set (dribble is expected to
   dominate, so F1 not accuracy).

### B3. How it scores
This is a *second* learned NN (sequence model) and a *controlled comparison*
(rule-based vs learned) in its own right — but on a **different protocol** (F1, not
HOTA), so it strengthens the final product rather than the CI3 same-protocol
comparison. Keep it clearly separate from the Part A comparison.

---

## Part C — Final product

A polished, honest, end-to-end pipeline + report + video:

1. **Pipeline:** detector → appearance-Re-ID tracker (core, polished) → (stretch)
   ball + action classifier.
2. **Final report** consolidating CI1–CI3: problem, data/EDA, neural baseline (CI2),
   Re-ID extension with the same-protocol comparison + ablations + failure analysis,
   and the action-classifier stretch. Build on the existing `docs/` writeups.
3. **Final video** walking through the whole project and the extension results.
4. **Reproducible repo:** scripts, pinned deps, clear run order; stale runs cleaned
   up.

---

## Sequencing (primary first, polish before stretch)

1. **Re-ID dataset** — `build_reid_dataset.py` (crops + identities).
2. **Train Re-ID** — `train_reid.py`; verify Rank-1/mAP.
3. **Embed + track** — `embed_dets.py`, `AppearanceTracker`; produce `reid` results.
4. **Compare** — run `eval_tracking.py`; the CI2-vs-CI3 table.
5. **Ablations** — appearance off/on, off-the-shelf/fine-tuned.
6. **Failure analysis** — same-team vs cross-team ID-switch breakdown + viz.
7. **Write up Part A** — `docs/checkin3_writeup.md` (the graded deliverable).
8. *(Stretch)* **Ball heuristic + label segments + action models + F1.**
9. **Final report + video + repo cleanup.**

Steps 1–7 are the graded Check-in 3. Steps 8–9 are the final product.

## Risks & mitigations
- **Re-ID gain is modest (jersey similarity).** Mitigation: that's a legitimate,
  well-interpreted result; the same-team/cross-team analysis (A7) makes it a finding,
  not a failure. The ablations still demonstrate the method works where appearance is
  informative.
- **Custom appearance tracker is finicky.** Mitigation: BoT-SORT-with-our-weights
  fallback (A4).
- **Action labeling is the stretch's bottleneck.** Mitigation: keep the labeled set
  small, classes few, and treat the stretch as optional — polish Part A first.
- **Scope creep from doing both.** Mitigation: hard gate — no Part B work until
  A1–A7 are done and written up.

## Reused infrastructure (already built in CI2)
Frozen detector + `detect_clips.py` cached detections, `eval_tracking.py` (TrackEval,
auto-discovers trackers), `sort_tracker.py` to extend, `viz_tracks.py`,
`metrics.csv`. The extension slots into all of it.
