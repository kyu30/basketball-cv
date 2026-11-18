# Play/Action Classifier Plan (Final-Product Stretch)

The README's headline extension: classify short segments into basic actions and
compare a **rule-based** classifier against a **learned sequence model**
(LSTM / small Transformer) on the same trajectory features. This builds on the
tracker output we already have (player trajectories) and is scored on **per-class
F1** — a *different* protocol from CI3's HOTA, so it strengthens the final product
but is kept separate from the CI3 comparison.

> **Status:** plan only. Gated behind the CI3 Re-ID extension, which is done.

## 1. The honest dependencies (read first)

Two hard constraints shape everything:

1. **No action labels exist in SportsMOT** → we must **hand-label** a small dataset.
   This is the real cost and the main risk to the timeline (your time, ~2–4 hr).
2. **pass and shot are ball-centric**, and the ball is neither annotated nor tracked.
   So a ball signal is a prerequisite for the full pass/dribble/shot taxonomy.

These two points, not the modeling, are what make this a *stretch*.

## 2. Class set — decision needed

- **Option A (README-faithful): pass / dribble / shot.** Requires a ball signal.
  Most aligned with the stated goal; highest value, highest risk.
- **Option B (ball-light): player-motion events** derivable from trajectories alone,
  e.g. fast break / half-court set / transition, or per-player locomotion states.
  Lower risk (no ball needed) but drifts from the README.
- **Recommended:** target **A**, with B as a documented fallback if the ball signal
  proves too noisy. Dribble is detectable with a weak ball signal; pass/shot need a
  usable one.

## 3. Ball signal — decision needed

The crux. Broadcast balls are small, fast and motion-blurred, and the court/jerseys
share the ball's color, so pure heuristics are unreliable.

- **Option 1 — heuristic** (color + size + motion blob near players). Cheapest, per
  the README's "heuristic first," but likely noisy in clutter.
- **Option 2 — tiny learned ball detector:** hand-label a few hundred frames with
  ball boxes, fine-tune a small YOLO (reuses the CI2 detector tooling). More reliable
  ball trajectory; adds a second labeling task.
- **Recommended:** try Option 1 first on a handful of clips; if detection rate is
  poor (likely), escalate to Option 2. The README explicitly allows this ("upgraded
  to a learned detector only if justified by EDA evidence").

## 4. Data & labeling

- **Unit:** short fixed-length segments (e.g. ~1.5 s / ~12 frames at stride 3) centered
  on or sampled across possessions, each given one action label.
- **Amount:** ~150–300 labeled segments to start; dribble will dominate, so collect
  extra pass/shot examples (class imbalance is expected and is why we score F1).
- **Tooling:** a small labeling helper (`src/label_segments.py`) that plays a segment
  and records a keypress → class, writing a manifest CSV. Makes labeling fast.
- **Split:** by **clip** (train/val/test segments from disjoint clips) to avoid
  leakage, mirroring the detector/Re-ID discipline.

## 5. Features (shared by both classifiers)

From tracker output (+ ball if available), per frame within a segment:
- player positions & velocities (court-normalized), inter-player spacing stats,
- ball position/velocity and ball-to-nearest-player distance (if ball available),
- simple derived signals: ball vertical velocity (shot), ball oscillation near a
  player (dribble), ball transfer between players (pass).

New script: `src/track_to_trajectories.py` (tracker MOT output → per-segment feature
tensors).

## 6. Models & comparison (the gradeable NN part)

- **Rule-based classifier** (`src/rule_classifier.py`): thresholds on the derived
  signals above. Transparent baseline.
- **Learned sequence model** (`src/train_action_model.py`): a small **LSTM or
  1-D Transformer** over the per-frame feature sequence → class. Kept small (dataset
  is small); dropout/weight-decay; optional k-fold given few samples.
- **Comparison:** per-class **F1** (and macro-F1) on the held-out segment split,
  rule-based vs learned, plus a confusion matrix. This is the controlled comparison.

## 7. Evaluation

- Primary: **per-class F1 + macro-F1** (not accuracy — dribble dominates).
- Secondary: confusion matrix; qualitative clips of correct vs mistaken predictions.
- Report where the learned model beats / loses to the rules, and tie errors back to
  ball-signal quality.

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Ball detection too noisy | escalate heuristic → tiny learned detector; or fall back to ball-light class set (Option B) |
| Labeling is slow / small dataset | labeling helper; keep model small; k-fold; class-balanced sampling |
| Learned model overfits tiny data | strong regularization; report honestly even if rules win (a legitimate finding) |
| Scope creep vs the final deadline | hard gate: this starts only after CI3 is written up; ship a smaller class set rather than nothing |

## 9. Sequence

1. Ball signal: heuristic prototype → evaluate detection rate → (if needed) label +
   train tiny ball detector.
2. `label_segments.py`; hand-label the segment set.
3. `track_to_trajectories.py`: features from tracks (+ ball).
4. `rule_classifier.py` (baseline) and `train_action_model.py` (LSTM/Transformer).
5. Evaluate (F1, confusion matrix), qualitative examples.
6. Write up (`docs/classifier_writeup.md`) + fold into the final report.

## 10. Reused infrastructure

Tracker output (`outputs/tracking/reid/`), the CI2 detector tooling (for a ball
detector), `data_loading`, the clip-split discipline, PyTorch. The trajectory and
labeling scripts are the main new pieces.

---

### Decisions to confirm before building
1. **Class set:** pass/dribble/shot (recommended) vs ball-light events.
2. **Ball signal:** heuristic-first then escalate (recommended) vs go straight to a
   tiny learned ball detector.
3. **Labeling budget:** how many segments you're willing to label (sets the ceiling
   on the learned model).
