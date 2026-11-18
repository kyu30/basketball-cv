"""Hand-label the basketball location on sampled frames, to train a ball detector.

On first run, samples frames across the clips into data/ball/frames.csv. Then shows
each frame: click the ball's center, adjust the box size if needed, confirm. Frames
where the ball isn't visible are marked as background (empty label) — those are
useful negatives. Resumable; saves after every frame.

Controls:
    left-click      place ball center (a box preview appears)
    space / enter   confirm ball present at the click
    n               ball NOT visible (background negative)
    [ / ]           shrink / grow the box
    b               back (re-label previous frame)
    q               save & quit

Usage:
    python src/label_ball.py                     # defaults (30 frames/clip)
    python src/label_ball.py --per-clip 40
"""

import argparse
import csv
import random
from pathlib import Path

import cv2

from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALL_DIR = PROJECT_ROOT / "data" / "ball"
FRAMES_CSV = BALL_DIR / "frames.csv"
FIELDS = ["clip", "frame", "state", "cx", "cy", "half"]  # state: '' todo, 'y' present, 'n' background


def generate_pool(split, per_clip, seed):
    rng = random.Random(seed)
    rows = []
    for clip_path in get_basketball_clips(split):
        info = load_seqinfo(clip_path)
        for f in sorted(rng.sample(range(1, info["seqlength"] + 1), min(per_clip, info["seqlength"]))):
            rows.append({"clip": info["name"], "frame": f, "state": "", "cx": "", "cy": "", "half": ""})
    return rows


def load_or_create(split, per_clip, seed):
    if FRAMES_CSV.exists():
        return list(csv.DictReader(open(FRAMES_CSV)))
    BALL_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_pool(split, per_clip, seed)
    save(rows)
    print(f"Generated {len(rows)} frames to label -> {FRAMES_CSV}")
    return rows


def save(rows):
    with open(FRAMES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def clip_dir(split, name):
    for c in get_basketball_clips(split):
        if c.name == name:
            return c
    raise SystemExit(f"Clip {name} not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-clip", type=int, default=30)
    parser.add_argument("--disp-width", type=int, default=1280, help="display width (clicks mapped back to full res)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_or_create(args.split, args.per_clip, args.seed)
    idx = next((i for i, r in enumerate(rows) if not r["state"]), len(rows))
    win = "label ball  click center | space=confirm  n=none  [ ]=size  b=back  q=quit"
    cv2.namedWindow(win)
    state = {"cx": None, "cy": None, "half": 14, "scale": 1.0}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["cx"], state["cy"] = x / state["scale"], y / state["scale"]
    cv2.setMouseCallback(win, on_mouse)

    while 0 <= idx < len(rows):
        r = rows[idx]
        img0 = cv2.imread(str(clip_dir(args.split, r["clip"]) / "img1" / f"{int(r['frame']):06d}.jpg"))
        if img0 is None:
            idx += 1
            continue
        h, w = img0.shape[:2]
        state["scale"] = args.disp_width / w
        state["cx"] = state["cy"] = None  # reset click per frame
        n_pos = sum(1 for x in rows if x["state"] == "y")
        n_done = sum(1 for x in rows if x["state"])

        decided = None
        while decided is None:
            disp = cv2.resize(img0, (args.disp_width, int(h * state["scale"]))).copy()
            if state["cx"] is not None:
                sc = state["scale"]
                px, py, ph = int(state["cx"] * sc), int(state["cy"] * sc), int(state["half"] * sc)
                cv2.rectangle(disp, (px - ph, py - ph), (px + ph, py + ph), (0, 255, 0), 2)
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(disp, f"#{idx+1}/{len(rows)}  present:{n_pos} done:{n_done}  half={state['half']}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(disp, f"{r['clip']} f{r['frame']}  click ball, space=confirm, n=none, b=back, q=quit",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord(" "), 13) and state["cx"] is not None:
                r.update(state="y", cx=round(state["cx"], 1), cy=round(state["cy"], 1), half=state["half"])
                save(rows); decided = "next"
            elif key == ord("n"):
                r.update(state="n", cx="", cy="", half="")
                save(rows); decided = "next"
            elif key == ord("["):
                state["half"] = max(4, state["half"] - 2)
            elif key == ord("]"):
                state["half"] = min(60, state["half"] + 2)
            elif key == ord("b"):
                decided = "back"
            elif key == ord("q"):
                save(rows); cv2.destroyAllWindows()
                print(f"Saved. present:{n_pos} done:{n_done} -> {FRAMES_CSV}")
                return
        idx = max(0, idx - 1) if decided == "back" else idx + 1

    cv2.destroyAllWindows()
    print(f"Done. present:{sum(1 for x in rows if x['state']=='y')} / {len(rows)}")


if __name__ == "__main__":
    main()
