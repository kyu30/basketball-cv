"""Feasibility check: what fraction of player crops yield a readable jersey number?

Samples ground-truth player boxes, OCRs the torso region (digits only) with EasyOCR,
and reports the confident-read rate overall and by player size. This estimates
whether jersey-number recognition is worth building to fix same-team ID confusions
(a general OCR is a rough lower bound; a trained number recognizer would do better).

Usage:
    python src/jersey_ocr_check.py --samples 300
"""

import argparse
import random

import cv2
import numpy as np

from data_loading import get_basketball_clips, load_gt, load_seqinfo


def torso_crop(img, x, y, w, h):
    # upper-middle of the box, where the number sits (front or back)
    y1, y2 = int(y + 0.12 * h), int(y + 0.55 * h)
    x1, x2 = int(x + 0.12 * w), int(x + 0.88 * w)
    c = img[max(y1, 0):y2, max(x1, 0):x2]
    if c.size == 0:
        return None
    scale = 128 / max(c.shape[0], 1)
    return cv2.resize(c, (int(c.shape[1] * scale), 128), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.4, help="min OCR confidence for a 'read'")
    parser.add_argument("--min-h", type=int, default=60, help="skip boxes shorter than this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    rng = random.Random(args.seed)

    # gather candidate (clip, frame, box)
    clips = get_basketball_clips("val")
    pool = []
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        gt = load_gt(clip_path)
        for _ in range(args.samples // len(clips) + 2):
            frame = rng.randint(1, info["seqlength"])
            g = gt[gt.frame == frame]
            if len(g):
                r = g.iloc[rng.randrange(len(g))]
                if r.h >= args.min_h:
                    pool.append((clip_path, frame, float(r.x), float(r.y), float(r.w), float(r.h)))
    rng.shuffle(pool)
    pool = pool[:args.samples]

    results = []  # (height, read_bool, text)
    for clip_path, frame, x, y, w, h in pool:
        img = cv2.imread(str(clip_path / "img1" / f"{frame:06d}.jpg"))
        if img is None:
            continue
        crop = torso_crop(img, x, y, w, h)
        if crop is None:
            continue
        hit, text = False, ""
        for _bbox, t, conf in reader.readtext(crop, allowlist="0123456789"):
            if conf >= args.conf and 1 <= len(t) <= 2 and t.isdigit():
                hit, text = True, t
                break
        results.append((h, hit, text))

    heights = np.array([r[0] for r in results])
    hits = np.array([r[1] for r in results])
    print(f"\n{len(results)} player crops (min height {args.min_h}px), OCR conf>={args.conf}")
    print(f"overall confident-number rate: {hits.mean():.1%}")
    bins = [(0, 120, "small"), (120, 160, "medium"), (160, 9999, "large")]
    print(f"{'size':<8}{'height px':>12}{'n':>6}{'read rate':>11}")
    for lo, hi, name in bins:
        m = (heights >= lo) & (heights < hi)
        if m.sum():
            rng_lbl = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
            print(f"{name:<8}{rng_lbl:>12}{int(m.sum()):>6}{hits[m].mean():>10.1%}")


if __name__ == "__main__":
    main()
