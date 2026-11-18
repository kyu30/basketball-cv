"""Turn tracker output + labeled segments into per-segment trajectory features.

For each hand-labeled segment, reads the Re-ID tracker's player boxes over the
segment's frames and builds a length-L sequence of court-level motion features
(player spread, congestion, speeds, team movement). No ball is used here — this is
the player-trajectory-only baseline; its F1 tells us whether a ball signal is worth
adding.

Output:
    data/actions/features.npz  -> X (N, L, D) float32, y (N,), clips (N,), label_names

Usage:
    python src/track_to_trajectories.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIONS_DIR = PROJECT_ROOT / "data" / "actions"
LABELS_CSV = ACTIONS_DIR / "labels.csv"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "reid" / "data"
BALL_DIR = ACTIONS_DIR / "ball_traj"
IMW, IMH = 1280.0, 720.0
LABEL_NAMES = ["dribble", "pass", "shot", "none"]
D = 10       # per-frame player feature dimension
# Full ball feature layout (11); flags select which columns are written:
#   base (7): present, bx, by, speed, dy, dist-to-nearest, height-vs-players
#   handler (2): handler_speed, ball-handler relative speed  (dribble=low, pass=high)
#   possession (2): in_possession, possessor_changed
D_BALL = 11
BALL_BASE = list(range(7))
BALL_HANDLER = [7, 8]
BALL_POSSESS = [9, 10]
POSSESS_RADIUS = 0.06  # normalized ball-to-player distance counted as "in possession"


def load_tracks(clip: str) -> dict[int, np.ndarray]:
    """{frame: array of (id, cx, cy, area_norm)} from the tracker MOT file."""
    by_frame = defaultdict(list)
    for line in (TRACKS_DIR / f"{clip}.txt").read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        x, y, w, h = float(x), float(y), float(w), float(h)
        by_frame[int(f)].append((int(tid), (x + w / 2) / IMW, (y + h / 2) / IMH, (w * h) / (IMW * IMH)))
    return {f: np.array(v, dtype=float) for f, v in by_frame.items()}


def frame_features(players: np.ndarray, prev_centers: dict) -> tuple[np.ndarray, dict]:
    """One frame's feature vector; returns (features, {id: center} for next frame)."""
    feat = np.zeros(D, dtype=np.float32)
    centers = {}
    if len(players) == 0:
        return feat, centers

    ids = players[:, 0].astype(int)
    cxy = players[:, 1:3]
    areas = players[:, 3]
    centers = {int(i): c for i, c in zip(ids, cxy)}

    feat[0] = len(players) / 10.0
    feat[1] = areas.mean()
    feat[2], feat[3] = cxy[:, 0].mean(), cxy[:, 1].mean()
    feat[4], feat[5] = cxy[:, 0].std(), cxy[:, 1].std()
    if len(players) >= 2:
        d = np.linalg.norm(cxy[:, None] - cxy[None], axis=2)
        iu = np.triu_indices(len(players), k=1)
        feat[6] = d[iu].max()
        feat[7] = d[iu].min()
    # per-id speed vs previous frame (matched by track id)
    speeds = [np.linalg.norm(c - prev_centers[i]) for i, c in centers.items() if i in prev_centers]
    if speeds:
        feat[8] = np.mean(speeds)
        feat[9] = np.max(speeds)
    return feat, centers


def load_ball(clip: str, ball_dir: Path = BALL_DIR) -> dict:
    """{frame: (bx_norm, by_norm, conf)} from the ball detector, or {} if absent."""
    path = ball_dir / f"{clip}.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if r["cx"]:
            out[int(r["frame"])] = (float(r["cx"]) / IMW, float(r["cy"]) / IMH, float(r["conf"]))
    return out


def ball_frame_features(ball, players, prev_ball, prev_centers, last_pos, last_possessor):
    """11 ball features for one frame. Returns (feat, ball_pos_or_held, new_last_possessor).
    When the ball is undetected, position is *held* at its last known value (present=0)
    instead of collapsing to (0,0)."""
    feat = np.zeros(D_BALL, dtype=np.float32)
    present = ball is not None
    pos = (ball[0], ball[1]) if present else last_pos
    if pos is None:
        return feat, None, last_possessor  # never seen yet
    bx, by = pos
    feat[0] = 1.0 if present else 0.0
    feat[1], feat[2] = bx, by
    ball_vel = np.zeros(2)
    if present and prev_ball is not None:
        ball_vel = np.array([bx - prev_ball[0], by - prev_ball[1]])
        feat[3] = float(np.hypot(*ball_vel))     # speed
        feat[4] = by - prev_ball[1]              # vertical velocity (neg=up)
    if len(players):
        pc = players[:, 1:3]
        dists = np.linalg.norm(pc - np.array([bx, by]), axis=1)
        j = int(dists.argmin())
        feat[5] = dists[j]                       # dist to nearest player (handler)
        feat[6] = by - pc[:, 1].mean()           # ball height vs players
        hid = int(players[j, 0])                 # handler track id
        # handler velocity vs its previous position -> ball-relative motion
        if hid in prev_centers:
            hvel = pc[j] - prev_centers[hid]
            feat[7] = float(np.hypot(*hvel))                    # handler speed
            feat[8] = float(np.hypot(*(ball_vel - hvel)))       # ball moves independently of handler = pass
        if dists[j] < POSSESS_RADIUS:
            feat[9] = 1.0
            if last_possessor is not None and hid != last_possessor:
                feat[10] = 1.0
            last_possessor = hid
    return feat, (bx, by), last_possessor


def segment_features(clip_tracks, start, end, ball_tracks=None, handler=False, possession=False):
    cols = BALL_BASE + (BALL_HANDLER if handler else []) + (BALL_POSSESS if possession else [])
    n_ball = len(cols) if ball_tracks is not None else 0
    seq = np.zeros((end - start + 1, D + n_ball), dtype=np.float32)
    prev, prev_ball, last_pos, last_possessor = {}, None, None, None
    for i, frame in enumerate(range(start, end + 1)):
        players = clip_tracks.get(frame, np.zeros((0, 4)))
        pfeat, cur = frame_features(players, prev)
        seq[i, :D] = pfeat
        if ball_tracks is not None:
            bfeat, bpos, last_possessor = ball_frame_features(
                ball_tracks.get(frame), players, prev_ball, prev, last_pos, last_possessor)
            seq[i, D:] = bfeat[cols]
            if bpos is not None:
                last_pos = bpos
                if ball_tracks.get(frame) is not None:
                    prev_ball = bpos
        prev = cur
    return seq


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ball", action="store_true", help="append ball-detector features (-> features_ball.npz)")
    parser.add_argument("--possession", action="store_true",
                        help="also add in_possession + possessor_changed (-> features_ball_poss.npz)")
    parser.add_argument("--handler", action="store_true",
                        help="also add handler speed + ball-handler relative speed (-> features_ball_h.npz)")
    parser.add_argument("--ball-dir", default="ball_traj", help="ball-trajectory subdir (e.g. ball_traj_smooth)")
    parser.add_argument("--tag", default="", help="suffix appended to the output filename")
    args = parser.parse_args()
    ball_dir = ACTIONS_DIR / args.ball_dir

    rows = [r for r in csv.DictReader(open(LABELS_CSV)) if r["label"]]
    label_idx = {n: i for i, n in enumerate(LABEL_NAMES)}

    tracks_cache, ball_cache = {}, {}
    X, y, clips = [], [], []
    for r in rows:
        clip = r["clip"]
        if clip not in tracks_cache:
            tracks_cache[clip] = load_tracks(clip)
            if args.ball:
                ball_cache[clip] = load_ball(clip, ball_dir)
        ball_tracks = ball_cache.get(clip) if args.ball else None
        X.append(segment_features(tracks_cache[clip], int(r["start"]), int(r["end"]),
                                  ball_tracks, args.handler, args.possession))
        y.append(label_idx[r["label"]])
        clips.append(clip)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    clips = np.array(clips)
    ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.ball:
        base = "features"
    else:
        base = "features_ball" + ("_h" if args.handler else "") + ("_poss" if args.possession else "")
    out = ACTIONS_DIR / f"{base}{('_' + args.tag) if args.tag else ''}.npz"
    np.savez(out, X=X, y=y, clips=clips, label_names=np.array(LABEL_NAMES))
    print(f"Wrote {out.name}: X={X.shape}, {len(set(clips))} clips")
    if args.ball:
        have = sum(1 for c in set(clips) if ball_cache.get(c))
        print(f"  ball trajectories loaded for {have}/{len(set(clips))} clips")
    for n, i in label_idx.items():
        print(f"  {n:8s}: {(y == i).sum()}")


if __name__ == "__main__":
    main()
