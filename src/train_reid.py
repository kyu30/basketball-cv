"""Fine-tune a player re-identification embedding on the SportsMOT basketball crops.

Backbone: OSNet (purpose-built for Re-ID), initialized from Market-1501 person-ReID
weights by default (not ImageNet) so appearance diversity comes from a large
external person corpus — the mitigation for our few basketball source videos.
Trained with the standard Re-ID recipe: ID cross-entropy + batch-hard triplet loss,
with Re-ID augmentation (flip, color jitter, pad-crop, random erasing).

Evaluated by Rank-1 / mAP retrieval on the held-out `v_-6Os86HzwCs` identities
(disjoint from training), a sanity check that the embedding separates players
before it is plugged into the tracker.

Usage:
    python src/train_reid.py                        # Market1501 init (default)
    python src/train_reid.py --init imagenet        # ablation: no person-ReID init
    python src/train_reid.py --epochs 30 --out weights/reid_basketball.pth
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from torchreid.reid.losses import TripletLoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REID_ROOT = PROJECT_ROOT / "data" / "reid"
MARKET_WEIGHTS = PROJECT_ROOT / "weights" / "osnet_x1_0_market1501.pth"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict, tfm):
        self.paths = df["path"].tolist()
        self.labels = [label_map[i] for i in df["identity"]]
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(REID_ROOT / self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


class PKSampler(Sampler):
    """Each batch = P identities x K instances, as batch-hard triplet mining needs."""
    def __init__(self, labels, p: int, k: int, batches: int):
        self.labels = np.array(labels)
        self.p, self.k, self.batches = p, k, batches
        self.by_label = {l: np.where(self.labels == l)[0] for l in set(labels)}
        self.unique = list(self.by_label)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for i in random.sample(self.unique, self.p):
                idxs = self.by_label[i]
                batch.extend(np.random.choice(idxs, self.k, replace=len(idxs) < self.k).tolist())
            yield batch


def build_transforms():
    train = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.Pad(10), transforms.RandomCrop((256, 128)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5),
    ])
    test = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, test


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        f = model(x.to(device))
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def evaluate_rank(feats, labels):
    """Rank-1 and mAP with each crop as a query against all others (self excluded)."""
    sim = feats @ feats.t()
    n = len(labels)
    sim.fill_diagonal_(-2.0)
    order = sim.argsort(dim=1, descending=True)
    matches = (labels[order] == labels[:, None]).float()
    rank1 = matches[:, 0].mean().item()
    # mAP
    aps = []
    for i in range(n):
        m = matches[i]
        if m.sum() == 0:
            continue
        cum = torch.cumsum(m, 0)
        prec = cum / torch.arange(1, n + 1)
        aps.append((prec * m).sum().item() / m.sum().item())
    return rank1, float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", choices=["market", "imagenet"], default="market",
                        help="backbone init: Market-1501 person-ReID (default) or ImageNet (ablation)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8,
                        help="stop if val mAP doesn't improve for this many epochs (0 disables)")
    parser.add_argument("--p", type=int, default=8, help="identities per batch")
    parser.add_argument("--k", type=int, default=4, help="crops per identity")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (lower = less memory)")
    parser.add_argument("--out", default="weights/reid_basketball.pth")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    df = pd.read_csv(REID_ROOT / "manifest.csv")
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    train_ids = sorted(train_df["identity"].unique())
    label_map = {i: n for n, i in enumerate(train_ids)}
    num_classes = len(train_ids)

    tfm_train, tfm_test = build_transforms()
    train_labels = [label_map[i] for i in train_df["identity"]]
    sampler = PKSampler(train_labels, args.p, args.k, batches=len(train_df) // (args.p * args.k))
    train_loader = DataLoader(CropDataset(train_df, label_map, tfm_train),
                              batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_label_map = {i: n for n, i in enumerate(sorted(val_df["identity"].unique()))}
    val_loader = DataLoader(CropDataset(val_df, val_label_map, tfm_test),
                            batch_size=128, num_workers=args.workers, pin_memory=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model("osnet_x1_0", num_classes=num_classes, loss="triplet", pretrained=True)
    if args.init == "market":
        load_pretrained_weights(model, str(MARKET_WEIGHTS))  # overrides ImageNet
    model = model.to(device)

    ce = nn.CrossEntropyLoss()
    tri = TripletLoss(margin=args.margin)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    print(f"Re-ID training | init={args.init} | {num_classes} train ids | "
          f"{len(train_df)} crops | {len(sampler)} batches/epoch | device={device}", flush=True)
    best_map = 0.0
    since_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, feats = model(x)  # triplet-mode OSNet returns (logits, features) in train
            loss = ce(logits, y) + tri(feats, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        feats, labels = extract(model, val_loader, device)
        rank1, mAP = evaluate_rank(feats, labels)
        tag = ""
        if mAP > best_map:
            best_map = mAP
            since_improve = 0
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            tag = " *saved"
        else:
            since_improve += 1
        print(f"epoch {epoch:2d}/{args.epochs} | loss {tot/len(train_loader):.3f} | "
              f"val Rank-1 {rank1:.3f} mAP {mAP:.3f}{tag}", flush=True)
        if args.patience and since_improve >= args.patience:
            print(f"early stopping: no val mAP improvement in {args.patience} epochs", flush=True)
            break

    print(f"\nBest val mAP {best_map:.3f} | saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
