"""Hand-label possession-length windows with their offensive play type (pilot tool).

Play types span a whole possession, so windows are ~10s (vs 1.5s for actions). This
is a PILOT: label ~30-45 windows, then we check class frequencies to decide whether
#9 (play-type classification) is viable with only 15 clips of footage.

On first run, samples windows into data/plays/labels.csv. Each window plays on a loop
(decimated for speed); press the play type you see.

Keys:
    1 = pick-and-roll   2 = isolation   3 = post-up   4 = transition/fast-break
    0 = other / unclear (skip)
    b = back (re-label previous)        q = save & quit

Labels written to data/plays/labels.csv after every keypress (resumable).

Usage:
    python src/label_plays.py                     # pilot: 3 windows/clip (~45)
    python src/label_plays.py --per-clip 6        # more
"""

import argparse
import csv
import random
from pathlib import Path

import cv2

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYS_DIR = PROJECT_ROOT / "data" / "plays"
LABELS_CSV = PLAYS_DIR / "labels.csv"
FIELDS = ["clip", "start", "end", "label"]
KEY_TO_LABEL = {ord("1"): "pick_and_roll", ord("2"): "isolation",
                ord("3"): "post_up", ord("4"): "transition", ord("0"): "other"}
CLASSES = ["pick_and_roll", "isolation", "post_up", "transition", "other"]


def generate_pool(split, per_clip, seg_len, seed):
    rng = random.Random(seed)
    rows = []
    for clip_path in get_basketball_clips(split):
        info = load_seqinfo(clip_path)
        last = info["seqlength"] - seg_len
        if last < 1:
            continue
        for s in sorted(rng.sample(range(1, last + 1), min(per_clip, last))):
            rows.append({"clip": info["name"], "start": s, "end": s + seg_len - 1, "label": ""})
    return rows


def load_or_create(split, per_clip, seg_len, seed):
    if LABELS_CSV.exists():
        return list(csv.DictReader(open(LABELS_CSV)))
    PLAYS_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_pool(split, per_clip, seg_len, seed)
    save(rows)
    print(f"Generated {len(rows)} possession windows -> {LABELS_CSV}")
    return rows


def save(rows):
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def clip_dir(split, name):
    for c in get_basketball_clips(split):
        if c.name == name:
            return c
    raise SystemExit(f"Clip {name} not found")


def load_window(cdir, start, end, play_stride, width):
    frames = []
    for f in range(start, end + 1, play_stride):
        img = cv2.imread(str(cdir / "img1" / f"{f:06d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        frames.append(cv2.resize(img, (width, int(h * width / w))))
    return frames


def counts(rows):
    c = {k: 0 for k in CLASSES}
    for r in rows:
        if r["label"] in c:
            c[r["label"]] += 1
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-clip", type=int, default=3, help="windows per clip (pilot default)")
    parser.add_argument("--seg-len", type=int, default=250, help="window length in frames (~10s at 25fps)")
    parser.add_argument("--play-stride", type=int, default=3, help="show every Nth frame (speed + memory)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_or_create(args.split, args.per_clip, args.seg_len, args.seed)
    delay = max(1, int(1000 / args.fps))
    idx = next((i for i, r in enumerate(rows) if not r["label"]), len(rows))
    win = "label plays  1=PnR 2=iso 3=post 4=transition 0=other  b=back q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(rows):
        r = rows[idx]
        frames = load_window(clip_dir(args.split, r["clip"]), int(r["start"]), int(r["end"]),
                             args.play_stride, args.width)
        if not frames:
            idx += 1
            continue
        c = counts(rows)
        bar = f"#{idx+1}/{len(rows)}  PnR:{c['pick_and_roll']} iso:{c['isolation']} post:{c['post_up']} trans:{c['transition']} other:{c['other']}"

        decided, i = None, 0
        while decided is None:
            img = frames[i % len(frames)].copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
            cv2.putText(img, bar, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(img, "1=PnR 2=iso 3=post 4=transition 0=other  b=back q=quit", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(win, img)
            key = cv2.waitKey(delay) & 0xFF
            if key in KEY_TO_LABEL:
                r["label"] = KEY_TO_LABEL[key]
                save(rows); decided = "next"
            elif key == ord("b"):
                decided = "back"
            elif key == ord("q"):
                save(rows); cv2.destroyAllWindows()
                print(f"Saved {sum(1 for x in rows if x['label'])} labels -> {LABELS_CSV}")
                return
            i += 1
        idx = max(0, idx - 1) if decided == "back" else idx + 1

    cv2.destroyAllWindows()
    c = counts(rows)
    print("Done.", " ".join(f"{k}:{v}" for k, v in c.items()))


if __name__ == "__main__":
    main()
