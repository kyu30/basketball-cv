# Play/Action Classifier (Final-Product Extension)

Classifies short video segments into **dribble / pass / shot / other** and compares
a **rule-based** classifier against a **learned LSTM**, per the README's headline
extension. Built on the tracker output (player trajectories) plus a from-scratch
**ball detector**. Scored on per-class F1 (not accuracy — dribble dominates), with
clip-grouped cross-validation. Plan in [`classifier_plan.md`](classifier_plan.md).

---

## 1. Data (hand-labeled — none exists in SportsMOT)

- **Action segments:** 600 ~1.5s segments hand-labeled across all 15 clips —
  dribble 249, pass 173, shot 76, other 102. "Other" (rebounds, tips, blocks,
  steals, dead time) is a legitimate background class, not a throwaway.
- **Ball boxes:** 450 frames hand-labeled — 303 with the ball, 147 background
  negatives — to train the ball detector.
- Labeling tools: `src/label_segments.py` (keypress per segment) and
  `src/label_ball.py` (click the ball). Both resumable.

## 2. Ball detector (the enabling component)

No ball annotations exist, and a color heuristic (`src/ball_heuristic.py`) proved
unreliable — it latched onto warm-colored crowds and wood courts and missed the
small, fast, motion-blurred ball, especially across the dataset's very different
venues. So we trained one:

- YOLOv8n fine-tuned at **imgsz 1280** (small object → high resolution), single
  class, on the 450 labeled frames (`src/train_ball_detector.py`).
- **Clip-split val: mAP50 0.889, precision 0.955, recall 0.781** — but this val
  shares source videos with train, so it measures *within-venue* quality.
- **Video-disjoint val (honest generalization): mAP50 0.496, recall 0.483**
  (`--val-videos`, a held-out game). The ball detector **does not generalize to
  unseen venues** — on a new game its recall roughly halves. The 0.889 reflects the
  within-venue regime it actually operates in here (it was trained on frames sampled
  from all clips it later runs on — a feature-extractor convenience, not an
  action-label leak, since the action classifier still holds clips out).
- **Caveat for the results below:** because the classifier uses this within-venue
  ball signal, its ball-dependent gains (notably shot F1) would degrade on a
  genuinely unseen game — the same limited-source-diversity limit seen in Re-ID.
- Run over all clips (`src/detect_ball_clips.py`): ball found in **54–85% of
  frames** per clip (lower on the blurrier women's-game clips).

## 3. Features

Per frame within a segment (`src/track_to_trajectories.py`):
- **Player (10):** count, mean box area, centroid, spatial spread, max/min pairwise
  distance (spacing/congestion), mean/max player speed, centroid speed.
- **Ball (+7):** present, position, speed, **vertical velocity** (shots go up),
  distance to nearest player, ball height vs players.
- **Possession (+2, optional):** in-possession flag and `possessor_changed` (nearest
  player identity changes = a completed pass). Tested; see §5.

## 4. Methods & results

Three methods on identical clip-grouped 5-fold CV (no clip spans train/test);
out-of-fold predictions pooled (`src/train_action_model.py`).

| method | dribble | pass | shot | other | **macro-F1** |
|--------|---------|------|------|-------|--------------|
| majority (floor) | 0.587 | 0.000 | 0.000 | 0.000 | 0.147 |
| rule-based | 0.575 | 0.455 | 0.571 | 0.000 | 0.401 |
| LSTM, player-only | 0.552 | 0.437 | 0.451 | 0.396 | 0.459 |
| LSTM, player + ball | 0.533 | 0.430 | 0.750 | 0.355 | 0.517 |
| LSTM, + possession | 0.558 | 0.386 | 0.737 | 0.360 | 0.510 |
| + bidirectional + 5-seed ensemble | ~0.62 | ~0.47 | ~0.75 | ~0.36 | 0.555 ± 0.014 |
| + ball-absent fix (hold last pos) | ~0.65 | ~0.46 | ~0.78 | ~0.37 | 0.560 ± 0.003 |
| **+ handler features** | ~0.63 | **~0.49** | ~0.76 | ~0.36 | **0.563 ± 0.009** |

Macro-F1 is a 5-seed mean ± std (the single-config rows are one seed). Improvements
over the plain LSTM (0.516), none needing new labels:
- **bidirectional + ensemble**: +0.039 (variance reduction + full-sequence context);
  intervals (0.555 ± 0.014 vs 0.516 ± 0.015) don't overlap.
- **ball-absent fix**: holding the ball's last known position when undetected, instead
  of collapsing it to (0,0), is a wash on the mean but **cuts variance ~4×** (the
  (0,0) signal was noise).
- **handler features** (handler speed; ball-vs-handler relative speed): nudge the mean
  and specifically lift **pass (~0.47 → ~0.49)** — ball-relative-to-handler motion is a
  genuine pass/dribble cue — though within noise on the macro.

**Reading:**
- **Learned beats rule beats floor** (0.555 vs 0.401 vs 0.147) — the comparison the
  README asked for. The rule-based baseline is competitive on the three ball actions
  but can't model "other"; the LSTM is more balanced.
- **Ensembling + bidirectionality help all classes**, lifting dribble (~0.53 → ~0.62)
  and pass (~0.43 → ~0.47) — a modest dent in the pass/dribble confusion — by reducing
  variance and using full-sequence context, not by adding information.
- **The ball decisively fixes shots:** shot F1 **0.451 → 0.750** (63/76 shots correct,
  up from 41). Ball vertical velocity is a clean shot signal.

## 5. Failure analysis — the pass/dribble boundary

The dominant, stubborn error is **dribble ↔ pass** (confusion matrix: ~89 dribbles
called passes, ~65 passes called dribbles), and it **barely moves** whether or not
the ball is used.

- **Why player-only can't:** from body motion alone a dribble and a pass look nearly
  identical; the distinguishing event is the ball leaving the hand.
- **Why the ball doesn't fix it either:** the distinguishing event happens during the
  fast, motion-blurred pass flight — exactly the frames where the detector's ~78%
  recall drops. The ball is often *not detected* mid-pass.
- **We tested two targeted fixes and both failed:**
  1. **Explicit possession-transfer features** (`--possession`): a completed pass =
     the ball's nearest player changes. This *slightly hurt* pass (F1 0.430 → 0.386):
     with imperfect ball recall and crowded paints, the nearest player flickers
     between offense and defense, so `possessor_changed` fires noisily during
     dribbles too.
  2. **Ball-trajectory smoothing + gap interpolation** (`src/smooth_ball.py`) to feed
     the possession feature a continuous ball through pass flight: no improvement
     (macro-F1 0.501, pass 0.382). Interpolating *position* doesn't fix the noisy
     part — *which player* gains possession in a crowd.

**Conclusion:** the limitation is **ball-possession tracking through fast motion**,
not the classifier or features — demonstrated by two independent targeted fixes that
both fail for the same root cause. The best configuration remains player + ball
(no possession), macro-F1 0.517.

Other notes: "shot" and "other" both improve with the ball; the ball detector was
trained on frames sampled from the same clips it later runs on (a feature-extractor
convenience, not an action-label leak — the action CV still holds clips out).

## 6. Limitations & next steps

- **Feature-limited, not data-limited:** doubling toward more data (600 → 800 labeled
  segments) left macro-F1 flat (0.563 → 0.554 ± 0.009, intervals overlapping). The
  ceiling is *what player+ball trajectories reveal*, not sample size — more labeling
  won't help; richer signals (below) would.
- **Pass/dribble needs better ball tracking:** a higher-recall ball detector, temporal
  ball-trajectory smoothing/interpolation across detection gaps, or an explicit
  ball-in-flight model — the clearest path to lifting pass F1.
- **Fixed-size ball boxes** (28px) cap localization (mAP50-95 0.40); variable-size
  labeling would sharpen it.
- Segment-level labels only; frame-accurate action boundaries would enable detection
  rather than clip classification.

## 7. Reproducing

```bash
python src/label_segments.py                       # (manual) action labels
python src/label_ball.py                            # (manual) ball boxes
python src/build_ball_dataset.py                    # ball labels -> YOLO
python src/train_ball_detector.py --epochs 80       # ball detector (imgsz 1280)
python src/detect_ball_clips.py                     # ball trajectory per clip
python src/track_to_trajectories.py                 # player-only features
python src/track_to_trajectories.py --ball          # + ball features (best)
python src/track_to_trajectories.py --ball --possession   # + possession (ablation)
python src/train_action_model.py --features data/actions/features_ball.npz   # plain LSTM
python src/train_action_model.py --features data/actions/features_ball.npz --bidirectional --ensemble 5   # best
```
