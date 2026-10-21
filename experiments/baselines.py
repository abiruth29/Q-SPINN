"""
experiments/baselines.py
========================
Task 15: baselines trained on the SAME windows / split / epochs / optimiser budget as Q-SPINN.

* DeepLOB (Zhang et al. 2019) re-implemented: 3 conv blocks over (time x level), inception module,
  LSTM(64), softmax.  Input here: [B, T, 144] FI-2010 vector -> reshaped to [B,1,T,40] using the first
  40 LOB columns (10 levels x ask/bid price/volume), as in the original paper.
* `train_classifier` - one generic loop (class-weighted CE, Adam, best-val-F1 snapshot) used for every
  baseline so tuning budgets are equal by construction.
"""
import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        lr = lambda: nn.LeakyReLU(0.01)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, (1, 2), (1, 2)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, (1, 2), (1, 2)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, (1, 10)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32),
                                   nn.Conv2d(32, 32, (4, 1)), lr(), nn.BatchNorm2d(32))
        self.inc1 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), padding="same"), lr(), nn.BatchNorm2d(64),
                                  nn.Conv2d(64, 64, (3, 1), padding="same"), lr(), nn.BatchNorm2d(64))
        self.inc2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), padding="same"), lr(), nn.BatchNorm2d(64),
                                  nn.Conv2d(64, 64, (5, 1), padding="same"), lr(), nn.BatchNorm2d(64))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), 1, padding=(1, 0)), nn.Conv2d(32, 64, (1, 1), padding="same"),
                                  lr(), nn.BatchNorm2d(64))
        self.lstm = nn.LSTM(192, 64, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):                                  # x: [B, T, >=40]
        x = x[:, :, :40].unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)   # [B,192,T',1]
        x = x.squeeze(-1).permute(0, 2, 1)
        return self.fc(self.lstm(x)[0][:, -1])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_classifier(model, Xtr, ytr, Xva, yva, device, f1_fn, epochs=20, lr=1e-3, bs=512, seed=0, patience=15):
    g = torch.Generator().manual_seed(seed)
    model = model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    cw = 1.0 / torch.bincount(ytr, minlength=3).float().clamp(min=1)
    ce = nn.CrossEntropyLoss(weight=(cw / cw.sum() * 3.0).to(device))
    best, snap, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            j = perm[i:i + bs]
            loss = ce(model(Xtr[j].to(device)), ytr[j].to(device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); p = []
        with torch.no_grad():
            for i in range(0, len(Xva), 1024):
                p.append(model(Xva[i:i + 1024].to(device)).argmax(1).cpu())
        f = f1_fn(torch.cat(p), yva)
        if f > best: best, snap, bad = f, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if patience and bad >= patience: break          # same early-stopping rule for every baseline
    if snap is not None: model.load_state_dict(snap)
    return model.eval()


@torch.no_grad()
def predict(model, X, device, bs=1024):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(X), bs)])


# ── naive microstructure baseline (does a trivial rule explain the "predictive" accuracy?) ─────────────────────
def past_return_lookup(Xtr, ytr, Xeval, yeval, k: int = 5, ret_idx: int = 0, n_bins: int = 3, n_classes: int = 3):
    """
    Lookup-table baseline: score = sum of the last `k` returns (channel `ret_idx`) in each window; bins = train
    quantiles; each bin predicts the TRAIN-majority label of that bin.  Captures bid-ask bounce / mean reversion /
    stale-price artefacts.  A learned model must beat this before its accuracy is read as signal.
    Returns (acc, macro_f1, table) with table[bin] = predicted class.
    """
    import torch
    def score(X): return X[:, -k:, ret_idx].double().sum(1)
    s_tr = score(Xtr)
    qs = torch.quantile(s_tr, torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1])
    def to_bin(s): return torch.bucketize(s, qs)
    b_tr = to_bin(s_tr)
    table = []
    for b in range(n_bins):
        m = b_tr == b
        table.append(int(torch.bincount(ytr[m], minlength=n_classes).argmax()) if m.any() else 1)
    table_t = torch.tensor(table)
    pred = table_t[to_bin(score(Xeval))]
    acc = float((pred == yeval).float().mean())
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (yeval == c)).sum()); fp = int(((pred == c) & (yeval != c)).sum()); fn = int(((pred != c) & (yeval == c)).sum())
        d = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
    return acc, sum(f1s) / n_classes, table
