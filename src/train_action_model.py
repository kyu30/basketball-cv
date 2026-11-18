"""Classify labeled segments into dribble/pass/shot/other and compare approaches.

Three methods on identical clip-grouped 5-fold splits (out-of-fold predictions
pooled for one honest confusion matrix / F1 per method):
    majority    - always predict the training-majority class (floor)
    rule-based  - hand heuristics on aggregate trajectory motion (README baseline)
    lstm        - learned sequence model over the per-frame feature sequence

Per-class F1 + macro-F1 (dribble dominates, so F1 not accuracy). Player-trajectory
features only (no ball) — this measures whether a ball signal is needed.

Usage:
    python src/train_action_model.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PROJECT_ROOT / "data" / "actions" / "features.npz"
N_FOLDS = 5


# ---- metrics ----------------------------------------------------------------
def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m

def per_class_f1(cm):
    f1 = []
    for c in range(len(cm)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return np.array(f1)


# ---- rule-based baseline ----------------------------------------------------
def rule_predict(X, names):
    """Heuristics on aggregate motion. Feature idx: 1 area, 4/5 spread, 8 mean spd, 9 max spd."""
    idx = {n: i for i, n in enumerate(names)}
    preds = []
    for seq in X:
        peak = seq[:, 9].max()
        mean_spd = seq[:, 8].mean()
        spread = seq[:, 4] + seq[:, 5]
        spread_change = spread[-spread.shape[0] // 3:].mean() - spread[: spread.shape[0] // 3].mean()
        area_change = seq[-seq.shape[0] // 3:, 1].mean() - seq[: seq.shape[0] // 3, 1].mean()
        if peak < 0.012 and mean_spd < 0.006:
            preds.append(idx["none"])          # little motion
        elif area_change > 0.004:
            preds.append(idx["shot"])          # players converge/zoom (rebound crash)
        elif peak > 0.03 and spread_change > 0:
            preds.append(idx["pass"])          # burst + players spreading
        else:
            preds.append(idx["dribble"])       # sustained ball-handler motion
    return np.array(preds)


# ---- lstm -------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, d_in, n_classes, hidden=32, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * (2 if bidirectional else 1), n_classes))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.lstm.bidirectional else h[-1]
        return self.head(h)


def train_lstm(Xtr, ytr, Xte, n_classes, epochs, device, bidir=False, ensemble=1, seed=0):
    """Returns softmax probabilities over the test set, averaged over `ensemble` seeds."""
    mean, std = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr_t = torch.tensor((Xtr - mean) / std, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor((Xte - mean) / std, dtype=torch.float32, device=device)

    counts = np.bincount(ytr, minlength=n_classes)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w)

    probs = torch.zeros(len(Xte), n_classes, device=device)
    for e in range(ensemble):
        torch.manual_seed(seed * 100 + e)
        model = LSTMClassifier(Xtr.shape[2], n_classes, bidirectional=bidir).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            ce(model(Xtr_t), ytr_t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs += torch.softmax(model(Xte_t), dim=1)
    return (probs / ensemble).argmax(1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", default=str(FEATURES),
                        help="feature npz (features.npz player-only, or features_ball.npz)")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--ensemble", type=int, default=1, help="models averaged per fold")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = np.load(args.features, allow_pickle=True)
    X, y, clips = data["X"], data["y"], data["clips"]
    names = list(data["label_names"])
    k = len(names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # clip-grouped folds (no clip spans train and test)
    uclips = sorted(set(clips))
    fold_of = {c: i % N_FOLDS for i, c in enumerate(uclips)}
    folds = np.array([fold_of[c] for c in clips])

    oof = {m: np.full(len(y), -1) for m in ("majority", "rule", "lstm")}
    for f in range(N_FOLDS):
        te = folds == f
        tr = ~te
        oof["majority"][te] = np.bincount(y[tr], minlength=k).argmax()
        oof["rule"][te] = rule_predict(X[te], names)
        oof["lstm"][te] = train_lstm(X[tr], y[tr], X[te], k, args.epochs, device,
                                     bidir=args.bidirectional, ensemble=args.ensemble, seed=args.seed)

    print(f"{len(y)} segments, {len(uclips)} clips, {N_FOLDS}-fold clip-grouped CV, device={device}\n")
    header = f"{'method':<10}" + "".join(f"{n[:7]:>9}" for n in names) + f"{'macroF1':>9}"
    print(header)
    results = {}
    for m, preds in oof.items():
        cm = confusion(y, preds, k)
        f1 = per_class_f1(cm)
        results[m] = (cm, f1)
        print(f"{m:<10}" + "".join(f"{v:>9.3f}" for v in f1) + f"{f1.mean():>9.3f}")

    print("\nLSTM confusion (rows=true, cols=pred):")
    print("          " + "".join(f"{n[:7]:>9}" for n in names))
    for i, row in enumerate(results["lstm"][0]):
        print(f"{names[i]:<10}" + "".join(f"{v:>9d}" for v in row))


if __name__ == "__main__":
    main()
