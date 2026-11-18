"""Compute a Re-ID appearance embedding for every cached detection.

Reads the detections cached by detect_clips.py, crops each box from its frame,
runs the trained Re-ID model, and saves one L2-normalized embedding per detection,
row-aligned with the detection file. The appearance tracker consumes these so the
detector/detections stay identical to CI2 — only association changes.

Output (float16, shape (num_dets, 512), aligned 1:1 with dets/<clip>.txt rows):
    outputs/tracking/embeds/<name>/<clip>.npy

Usage:
    python src/embed_dets.py                                   # name=reid, market weights
    python src/embed_dets.py --weights weights/reid_imagenet.pth --name reid_imgnet
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from data_loading import get_basketball_clips, load_seqinfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "dets"
EMBEDS_DIR = PROJECT_ROOT / "outputs" / "tracking" / "embeds"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "reid_basketball.pth"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_reid_model(weights: Path, device: str):
    # load_pretrained_weights matches by key and discards the classifier head, so
    # it loads both our fine-tuned checkpoint and the raw Market-1501 one (the
    # off-the-shelf ablation); only backbone features are used for embeddings.
    model = build_model("osnet_x1_0", num_classes=1000, loss="triplet", pretrained=False)
    load_pretrained_weights(model, str(weights))
    return model.eval().to(device)


def read_dets(clip_name: str, dets_dir: Path) -> dict[int, list]:
    """{frame: [(row_index, x, y, w, h), ...]} preserving file row order."""
    by_frame = defaultdict(list)
    for i, line in enumerate((dets_dir / f"{clip_name}.txt").read_text().splitlines()):
        frame, x, y, w, h, _conf = line.split(",")
        by_frame[int(frame)].append((i, float(x), float(y), float(w), float(h)))
    return by_frame


def embed_clip(model, clip_path: Path, tfm, device: str, batch: int, dets_dir: Path) -> np.ndarray:
    info = load_seqinfo(clip_path)
    by_frame = read_dets(info["name"], dets_dir)
    n = sum(len(v) for v in by_frame.values())
    embs = np.zeros((n, 512), dtype=np.float16)

    buf_t, buf_idx = [], []

    def flush():
        if not buf_t:
            return
        with torch.no_grad():
            f = model(torch.stack(buf_t).to(device))
            f = torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float16)
        for j, row in enumerate(buf_idx):
            embs[row] = f[j]
        buf_t.clear(); buf_idx.clear()

    for frame in sorted(by_frame):
        img = cv2.imread(str(clip_path / "img1" / f"{frame:06d}.jpg"))
        if img is None:
            continue
        imh, imw = img.shape[:2]
        for row, x, y, w, h in by_frame[frame]:
            x1, y1 = max(int(x), 0), max(int(y), 0)
            x2, y2 = min(int(x + w), imw), min(int(y + h), imh)
            if x2 <= x1 or y2 <= y1:
                continue  # leaves a zero embedding; harmless for a degenerate box
            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            buf_t.append(tfm(Image.fromarray(crop)))
            buf_idx.append(row)
            if len(buf_t) >= batch:
                flush()
    flush()
    return embs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--name", default="reid", help="subdir under embeds/ (distinguishes ablation models)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dets-name", default="dets", help="dets subdir to read (match detect_clips)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_reid_model(Path(args.weights), device)
    tfm = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor(),
                              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    dets_dir = PROJECT_ROOT / "outputs" / "tracking" / args.dets_name
    out_dir = EMBEDS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = get_basketball_clips("val")  # test clips
    print(f"Embedding detections for {len(clips)} clips with '{args.name}' -> {out_dir}", flush=True)
    for clip_path in clips:
        info = load_seqinfo(clip_path)
        embs = embed_clip(model, clip_path, tfm, device, args.batch, dets_dir)
        np.save(out_dir / f"{info['name']}.npy", embs)
        print(f"  {info['name']}: {len(embs)} embeddings", flush=True)


if __name__ == "__main__":
    main()
