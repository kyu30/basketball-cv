"""Hand-label short video segments with their dominant ball action.

On first run this generates a pool of candidate segments (random windows across the
clips) into data/actions/labels.csv. Then it plays each unlabeled segment on a loop
in an OpenCV window and records your keypress. Resumable: quit anytime and rerun to
continue where you left off.

Keys while a segment plays:
    1 = dribble    2 = pass    3 = shot    0 = none / unclear (skip)
    b = back (re-label the previous segment)    q = save & quit

Labels are written to data/actions/labels.csv after every keypress (crash-safe).
You judge by watching — this needs no ball detector; that comes later, for features.

Usage:
    python src/label_segments.py                      # label with defaults
    python src/label_segments.py --per-clip 40 --seg-len 38
"""

import argparse
import csv
import random
from pathlib import Path

import cv2

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIONS_DIR = PROJECT_ROOT / "data" / "actions"
LABELS_CSV = ACTIONS_DIR / "labels.csv"
FIELDS = ["clip", "start", "end", "label"]
KEY_TO_LABEL = {ord("1"): "dribble", ord("2"): "pass", ord("3"): "shot", ord("0"): "none"}


def generate_pool(split: str, per_clip: int, seg_len: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for clip_path in get_basketball_clips(split):
        info = load_seqinfo(clip_path)
        last_start = info["seqlength"] - seg_len
        if last_start < 1:
            continue
        starts = rng.sample(range(1, last_start + 1), min(per_clip, last_start))
        for s in sorted(starts):
            rows.append({"clip": info["name"], "start": s, "end": s + seg_len - 1, "label": ""})
    return rows


def load_or_create_pool(split, per_clip, seg_len, seed) -> list[dict]:
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            return list(csv.DictReader(f))
    ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_pool(split, per_clip, seg_len, seed)
    save_pool(rows)
    print(f"Generated {len(rows)} candidate segments -> {LABELS_CSV}")
    return rows


def save_pool(rows: list[dict]):
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def add_segments(rows: list[dict], n: int, split: str, seg_len: int, seed: int) -> int:
    """Append n new, non-duplicate candidate segments to an existing pool. Returns count added."""
    existing = {(r["clip"], int(r["start"])) for r in rows}
    rng = random.Random(seed + 1000)  # offset so the new batch differs from the original
    clips = get_basketball_clips(split)
    per = n // len(clips) + 1
    new = []
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        last = info["seqlength"] - seg_len
        cands = [s for s in range(1, last + 1) if (info["name"], s) not in existing]
        for s in sorted(rng.sample(cands, min(per, len(cands)))):
            new.append({"clip": info["name"], "start": s, "end": s + seg_len - 1, "label": ""})
    rng.shuffle(new)
    new = new[:n]
    rows.extend(new)
    save_pool(rows)
    return len(new)


def clip_dir(split: str, clip_name: str) -> Path:
    for c in get_basketball_clips(split):
        if c.name == clip_name:
            return c
    raise SystemExit(f"Clip {clip_name} not found in split '{split}'")


def load_segment_frames(cdir: Path, start: int, end: int, width: int):
    frames = []
    for f in range(start, end + 1):
        img = cv2.imread(str(cdir / "img1" / f"{f:06d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        frames.append(cv2.resize(img, (width, int(h * width / w))))
    return frames


def counts(rows):
    c = {"dribble": 0, "pass": 0, "shot": 0, "none": 0}
    for r in rows:
        if r["label"] in c:
            c[r["label"]] += 1
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", help="clip split to sample from (clips with trajectories)")
    parser.add_argument("--per-clip", type=int, default=40, help="candidate segments per clip (pool size)")
    parser.add_argument("--seg-len", type=int, default=38, help="segment length in frames (~1.5s at 25fps)")
    parser.add_argument("--fps", type=int, default=20, help="playback speed")
    parser.add_argument("--width", type=int, default=960, help="display width")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add", type=int, default=0,
                        help="append N new candidate segments to the existing pool, then label them")
    args = parser.parse_args()

    rows = load_or_create_pool(args.split, args.per_clip, args.seg_len, args.seed)
    if args.add:
        added = add_segments(rows, args.add, args.split, args.seg_len, args.seed)
        print(f"Added {added} new segments (pool now {len(rows)}); starting at the first unlabeled.")
    delay = max(1, int(1000 / args.fps))

    # Start at the first unlabeled segment.
    idx = next((i for i, r in enumerate(rows) if not r["label"]), len(rows))
    win = "label segments  [1]dribble [2]pass [3]shot [0]skip  [b]back [q]quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(rows):
        r = rows[idx]
        cdir = clip_dir(args.split, r["clip"])
        frames = load_segment_frames(cdir, int(r["start"]), int(r["end"]), args.width)
        if not frames:
            idx += 1
            continue
        c = counts(rows)
        n_done = sum(c.values())
        bar = f"#{idx+1}/{len(rows)}  done:{n_done}  dribble:{c['dribble']} pass:{c['pass']} shot:{c['shot']}"
        hint = "[1]dribble [2]pass [3]shot [0]skip  [b]back [q]quit"

        decided = None
        i = 0
        while decided is None:
            img = frames[i % len(frames)].copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 56), (0, 0, 0), -1)
            cv2.putText(img, bar, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"{r['clip']} f{r['start']}-{r['end']}   {hint}", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(win, img)
            key = cv2.waitKey(delay) & 0xFF
            if key in KEY_TO_LABEL:
                r["label"] = KEY_TO_LABEL[key]
                save_pool(rows)
                decided = "next"
            elif key == ord("b"):
                decided = "back"
            elif key == ord("q"):
                save_pool(rows)
                cv2.destroyAllWindows()
                print(f"Saved. {n_done} labeled so far -> {LABELS_CSV}")
                return
            i += 1

        idx = max(0, idx - 1) if decided == "back" else idx + 1

    cv2.destroyAllWindows()
    c = counts(rows)
    print(f"Done - all segments visited. dribble:{c['dribble']} pass:{c['pass']} shot:{c['shot']} none:{c['none']}")


if __name__ == "__main__":
    main()
