# Final Submission — Review & Rubric Map

A checklist to review before recording the video and submitting, mapping the work to
the four grading criteria and pointing at the evidence for each.

## Rubric coverage

### 1. Repo structure + documentation (15 pts)
*"Polished, navigable repo; README as front door; clear story CI1 → extension."*
- **`README.md`** — front door: results-at-a-glance tables, the CI1→extension story
  with per-step doc links, repo map, documentation index, key findings.
- **8 docs in `docs/`** indexed from the README; `final_report.md` is the "start here".
- ☐ **To verify:** click every link in the README — all 11 should resolve.
- ☐ **To verify:** the repo is committed (see §4) — "navigable repo" implies git.

### 2. Reproducibility + run instructions (20 pts)
*"Environment, data access, how to run train/eval, and a small demo path."*
- **`REPRODUCING.md`** — conda env + `requirements.txt`; SportsMOT access via Hugging
  Face + `download_data.py`; the full A–D pipeline (detector → tracking → Re-ID →
  classifier); a **small demo path** (zero-setup via committed outputs, and a 2-command
  runnable GIF); a gotchas section.
- ☐ **To verify:** skim the pipeline commands against `src/` filenames (all present).
- ☐ **To verify (ideally):** run the demo path end-to-end once on a clean checkout.

### 3. Technical completeness + results evidence (20 pts)
*"Final system evidence supporting claims (metrics, examples, comparisons)."*
- **Metrics:** `outputs/tracking/metrics.csv` (tracker comparison),
  `outputs/tracking/ablation_alpha.csv`, `outputs/tracking/ablation_init.csv`,
  `outputs/detector/*/summary.txt`.
- **Examples:** `outputs/tracking/viz/*` (per-tracker frames + the side-by-side GIF).
- **Comparisons/ablations:** documented in `checkin3_writeup.md`, `classifier_writeup.md`.
- ☐ **To verify:** every headline number in the README/report traces to a file above.

### 4. Final analysis (insight, limitations, reflection) (15 pts)
*"Honest interpretation, limitations, next steps; graduate-level communication."*
- **`final_report.md`** §6–7: the overall before/after, limitations, and next steps.
- The honest-negatives throughout (play-types, jersey numbers, possession, smoothing)
  and the lessons (detector-mAP ≠ tracking; feature- vs data-limited).
- ☐ **To verify:** you can explain the same-team/cross-team failure aloud (it's the
  centerpiece insight).

## Deliverables inventory

| kind | where |
|------|-------|
| Front door | `README.md` |
| Reproduction | `REPRODUCING.md`, `requirements.txt` |
| Final report | `docs/final_report.md` |
| Stage writeups | `docs/{training_procedure,tracking,checkin2_writeup,checkin3_writeup,classifier_writeup}.md` |
| Plans | `docs/{checkin3_plan,classifier_plan}.md`, `docs/Project Check-in 1.md` |
| Video script | `docs/final_video_script.md` |
| EDA | `notebooks/01_eda.ipynb` |
| Code | `src/*.py` (29 scripts, grouped in the README) |
| Evidence | `outputs/` (metrics, plots, tracker results, viz, GIF) |

## Pre-submission checklist
- ☐ README links all resolve; results tables match the latest numbers.
- ☐ No stale `TBD`/placeholder left: `grep -rni "tbd\|pending\|_placeholder" docs/ README.md REPRODUCING.md`
- ☐ Numbers in the video script's quick-reference match the docs.
- ☐ Demo path works on a fresh clone (or at least the zero-setup artifacts open).
- ☐ **Committed** on a branch (you're on `main`), with `logs/` and the stray
  `Project Check-in 1_files/` folder gitignored.
- ☐ Record the video from `docs/final_video_script.md` (shot list included).

## Honesty guardrails for the video / report
- Frame absolute HOTA (~0.54) as a **strong relative result** (+62%), not a
  leaderboard number — SportsMOT SOTA uses far bigger models.
- Present the four investigated-but-not-viable directions as **evidence-backed
  findings**, not failures — they're a differentiator.
- Be clear the ball detector's within-venue 0.889 doesn't generalize (0.496), and the
  classifier's ball-dependent gains are within-venue.

## Known caveats to have ready (in case you're asked)
- Ball detector trained on frames from all clips (feature-extractor convenience; the
  action CV still holds clips out) — and it doesn't generalize across venues.
- `features_ball_h.npz` currently reflects 800 segments (macro-F1 0.554); the headline
  **0.563** is the 600-segment best — both are documented as the data-saturation result.
