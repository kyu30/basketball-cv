# Final Check-in — Video Script (~6–7 min, casual)

Conversational, first-person. `[SHOW: …]` = what to have on screen. Spoken lines are
what to actually say — paraphrase, don't read robotically. ~900 words (~6.5 min at a
relaxed pace). Trim the *italic* asides if long.

---

### 0:00 — Intro (~30s)
*[SHOW: the side-by-side tracker GIF — `outputs/tracking/viz/iou_vs_reid_motion_*.gif`]*

"Hey — this is the final walkthrough of my basketball tracking project. The goal:
take raw broadcast footage and track every player with a consistent ID, then figure
out what the ball-handler is doing. I'll walk through the whole thing — the detector,
the tracker, my main extension, and a stretch — and I'll be honest about what worked
and what didn't, because a lot of the interesting stuff is in the didn't."

### 0:30 — Data & the problem (~40s)
*[SHOW: a ground-truth frame with player boxes; the EDA notebook]*

"The data is SportsMOT, the basketball subset — about 30 clips that actually come
with labels. The core task is two steps: detect players, then link those detections
over time so each player keeps the same ID through occlusion and fast motion. And
right off the bat, my EDA flagged the hard part: in two-thirds of frames, players are
overlapping. So occlusion was always going to be the enemy."

### 1:10 — Detector baseline (~40s)
*[SHOW: README results table / `training_procedure.md`]*

"First the detector — I fine-tuned YOLOv8 from COCO. On the held-out test set it hits
**0.95 mAP**. To show what fine-tuning bought, I ran the off-the-shelf COCO model as a
baseline — that only gets **0.70**, and the gap is almost all *precision*: the generic
model boxes referees and the crowd, fine-tuning learns to ignore them. One thing I was
careful about: I split the data by *source video*, so clips from the same broadcast
can't leak between train and test. I actually caught a leak mid-project and fixed it."

### 1:50 — Tracking baseline + the key finding (~50s)
*[SHOW: the tracking HOTA table]*

"Then tracking. I built two baselines — a simple IoU tracker and ByteTrack — and
scored them with HOTA, the standard tracking metric. And here's the finding that
drove the whole rest of the project: **detection is good, but holding identities is
hard.** The detection part of the score is around 0.7, but the association part is
only ~0.16 to 0.22. Thousands of ID switches. So the bottleneck isn't seeing players —
it's telling them apart over time."

### 2:40 — The extension: learned Re-ID (~1:15)
*[SHOW: `checkin3_writeup.md` comparison table + ablations]*

"So my main extension attacks exactly that. I trained a player **re-identification
model** — basically a network that looks at a player and produces an appearance
fingerprint, so the tracker can re-link the same player after an occlusion. I started
from a model pretrained on a big person Re-ID dataset — that's important, because I
only have a few games' worth of jerseys — and fine-tuned it on basketball.

Plugged into the tracker, it lifts HOTA from **0.33 to 0.46** — beating both baselines,
almost entirely by fixing association. I ran two ablations to show *why*: one varying
how much I weight appearance, one comparing off-the-shelf versus basketball-tuned
weights — both the big-dataset pretraining and the fine-tuning matter.

*And the failure analysis is my favorite part:* I checked whether the remaining ID
mistakes were between opposing teams or between teammates. Appearance fixes the
cross-team ones — different jerseys — but **teammates wear identical uniforms**, so
those it basically can't fix. That's the structural limit, and the numbers show it:
the error mix shifts from 3-to-1 cross-team down to about 1.7-to-1."

### 3:55 — Pushing it further + a lesson (~45s)
*[SHOW: the extended tracking table ending at 0.536]*

"I kept pushing. Adding a motion model gets me to **0.48**. Then I tried a bigger
detector, YOLOv8s — and here's a lesson I didn't expect: at first it made tracking
*worse*. Higher detector accuracy, worse tracking. Turned out its confidence scores
are calibrated differently, so it flooded the tracker with false positives. Once I
re-tuned the threshold, it gave my best result — **HOTA 0.54**, which is **+62% over
where I started.** The takeaway: higher mAP doesn't automatically mean better tracking."

### 4:40 — The stretch: action classifier (~55s)
*[SHOW: `classifier_writeup.md` F1 table; a labeled segment]*

"The stretch was classifying what's happening — dribble, pass, shot, or other. No
labels exist, so I hand-labeled segments, and since passes and shots need the ball —
which also isn't labeled — I trained a little ball detector from scratch too.

I compared a rule-based classifier against a learned LSTM. The LSTM wins, **macro-F1
around 0.56** versus 0.40 for the rules. The ball **nails shots** — it can see the ball
go up. But pass versus dribble stays hard, and I figured out exactly why: telling them
apart needs the ball tracked *through* the pass, and that's the blurry fast moment
where the detector loses it. I even confirmed more data doesn't help — it's limited by
the signal, not the labels."

### 5:35 — What didn't work (~40s)
*[SHOW: final report limitations section]*

"I want to be upfront about the things I tried that *didn't* pan out, because I think
they're worth as much as the wins. I looked at classifying play types like
pick-and-roll — but the footage just doesn't have enough of them. And I checked
whether reading jersey numbers could fix the teammate problem — great idea in theory,
but at this video resolution the numbers are only legible about 12% of the time. So I
ruled both out, with evidence, instead of forcing them."

### 6:15 — Limitations & wrap (~30s)
*[SHOW: README front page / final report]*

"So — honest summary: a detector that works really well, a tracker that I improved
62% with a learned Re-ID model, and an action classifier with a clear picture of its
own ceiling. The repo's got a README front door, full reproduction instructions, and
a report tying it all together. The biggest limitation is really the dataset — a few
games, broadcast resolution — and most of my next steps come back to that. Thanks for
watching."

---

## Quick-reference numbers (don't misquote on camera)
- Detector test: **mAP50 0.954** (YOLOv8n), 0.961 (YOLOv8s); COCO baseline **0.698**; precision 0.53 → 0.94
- Tracking HOTA: 0.331 → 0.382 (ByteTrack) → **0.459** (Re-ID) → 0.484 (+motion) → **0.536** (best) — **+62%**; IDSW 3433 → 1131
- Re-ID ablation: off-the-shelf 0.410 vs fine-tuned 0.459; cross:same confusion 3.0 → 1.7
- Classifier macro-F1: majority 0.147 → rule 0.401 → **LSTM 0.563**; shot F1 0.45 → 0.75
- Ball detector: 0.889 within-venue → **0.496 cross-venue**
- Occlusion: 67.7% of frames; jersey legibility ~12%

## Screen-recording shot list
1. Side-by-side tracker GIF (`outputs/tracking/viz/iou_vs_reid_motion_*.gif`) — intro
2. A ground-truth frame + EDA notebook — data
3. README results tables (detector + tracking) — baselines
4. `checkin3_writeup.md` comparison + ablation tables — the extension
5. A Re-ID viz frame (`outputs/tracking/viz/reid/`) vs an IoU one — ID stability
6. `classifier_writeup.md` F1 table + a labeled segment — stretch
7. `final_report.md` limitations — what didn't work
8. README front page — wrap

## Tips
- Longest segment is the extension (2:40–4:40) — that's your headline, give it room.
- The honest-negatives beat (5:35) is a differentiator; don't cut it.
- If you run long, trim the italicized asides first; the shot list order matches the script.
