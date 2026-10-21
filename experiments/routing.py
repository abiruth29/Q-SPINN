"""
experiments/routing.py
======================
Phase 2 (tasks 08-10).  Pure torch/numpy so every piece is unit-testable on synthetic data.

08  Prediction-aware routing with a compute budget
    * `benefit_targets`     - per-sample improvement of heavy over cheap (loss gap / margin gap)
    * `fit_benefit_head`    - ridge regression of that benefit on cheap-side features (closed form)
    * `confidence_score`    - cascade baseline: 1 - max softmax of the cheap expert
    * `select_threshold`    - threshold chosen for a routing fraction on VALIDATION scores only
    * `f1_vs_fraction`      - validation macro-F1 curve over routing fractions 0..1
09  Real conditional inference
    * `conditional_infer`   - heavy expert is called ONLY on selected rows, outputs scattered back
10  Forward early warning
    * `forward_instability_labels`, `platt_fit/apply`, `alert_metrics`, `rolling_vol_score`
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch


# ───────────────────────────── 08: routing ─────────────────────────────────
def _macro_f1(pred: np.ndarray, true: np.ndarray, n: int = 3) -> float:
    f = []
    for c in range(n):
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c)); d = 2 * tp + fp + fn
        f.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f))


def benefit_targets(cheap: torch.Tensor, heavy: torch.Tensor, y: torch.Tensor, kind: str = "loss"):
    """
    benefit_i = how much better the heavy expert is for sample i  (positive => route to heavy).
    kind="loss":   CE(cheap) - CE(heavy)
    kind="margin": margin(heavy) - margin(cheap) with margin = z_y - max_{c!=y} z_c
    """
    if kind == "loss":
        ce = torch.nn.functional.cross_entropy
        return (ce(cheap, y, reduction="none") - ce(heavy, y, reduction="none")).detach()
    zy = lambda z: z.gather(1, y[:, None]).squeeze(1)
    other = lambda z: z.masked_fill(torch.nn.functional.one_hot(y, z.shape[1]).bool(), -1e30).max(1)[0]
    mh, mc = zy(heavy) - other(heavy), zy(cheap) - other(cheap)
    return (mh - mc).detach()


def fit_benefit_head(feats: np.ndarray, benefit: np.ndarray, l2: float = 1e-3):
    """Closed-form ridge fit on TRAIN features. Returns (w, b); score = feats @ w + b."""
    X = np.c_[feats, np.ones(len(feats))]
    A = X.T @ X + l2 * np.eye(X.shape[1])
    wb = np.linalg.solve(A, X.T @ benefit)
    return wb[:-1], float(wb[-1])


def apply_benefit_head(feats: np.ndarray, head) -> np.ndarray:
    w, b = head
    return feats @ w + b


def confidence_score(cheap_logits: torch.Tensor) -> torch.Tensor:
    """Cascade baseline: low cheap-expert confidence => high score."""
    return 1.0 - torch.softmax(cheap_logits, dim=1).max(1)[0]


def rolling_vol_score(mid: np.ndarray, window: int = 20) -> np.ndarray:
    """Backward rolling realised-vol router baseline (strictly causal)."""
    r = np.r_[0.0, np.diff(mid)]
    out = np.zeros(len(r))
    for i in range(len(r)):
        out[i] = r[max(0, i - window + 1): i + 1].std()
    return out


def select_threshold(val_scores: np.ndarray, fraction: float) -> float:
    """Score threshold routing the top `fraction` of VALIDATION samples (no test labels involved)."""
    if fraction <= 0: return float("inf")
    if fraction >= 1: return float("-inf")
    return float(np.quantile(val_scores, 1.0 - fraction))


def routed_predictions(scores, cheap_pred, heavy_pred, thr):
    sel = scores >= thr
    return np.where(sel, heavy_pred, cheap_pred), sel


def f1_vs_fraction(scores, cheap_pred, heavy_pred, y, fractions: Sequence[float] = tuple(np.linspace(0, 1, 11))):
    """Macro-F1 and actual routed fraction as the budget varies (use on validation to choose)."""
    rows = []
    for f in fractions:
        thr = select_threshold(scores, f)
        pred, sel = routed_predictions(scores, cheap_pred, heavy_pred, thr)
        rows.append(dict(target_fraction=float(f), routed=float(sel.mean()), macro_f1=_macro_f1(pred, y)))
    return rows


def choose_budget(curve_val, min_f1_ratio_of_dense: float = 0.99):
    """Smallest validation fraction whose macro-F1 >= ratio * (all-heavy macro-F1)."""
    dense = curve_val[-1]["macro_f1"]
    for r in curve_val:
        if r["macro_f1"] >= min_f1_ratio_of_dense * dense:
            return r["target_fraction"]
    return 1.0


# ─────────────────────────── 09: conditional inference ─────────────────────
@torch.no_grad()
def conditional_infer(cheap_logits: torch.Tensor, sel: torch.Tensor,
                      heavy_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                      chunk: int = 4096) -> Dict[str, object]:
    """
    Run the heavy expert only on rows where `sel` is True and scatter its logits into a copy of the
    cheap logits.  Zero selected => `heavy_fn` is never called.  Diagnostics that need the heavy
    training path (covariance NLL, PDE residual, norm loss) are NOT produced here.
    """
    out = cheap_logits.clone()
    idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
    calls = 0
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        out[j] = heavy_fn(x[j])
        calls += len(j)
    return dict(logits=out, heavy_rows=calls, routed=len(idx) / max(len(sel), 1),
                diagnostics="unavailable (skipped in conditional inference)")


def timed(fn, *args, device: Optional[torch.device] = None, warmup: int = 2, reps: int = 5):
    """Median wall-clock seconds with warm-up and device synchronisation (CPU or CUDA)."""
    sync = (lambda: torch.cuda.synchronize()) if (device is not None and device.type == "cuda") else (lambda: None)
    for _ in range(warmup): fn(*args)
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(*args); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ─────────────────────────── 10: forward early warning ─────────────────────
def future_vol(returns: np.ndarray, tau: int) -> np.ndarray:
    """Realised vol over the strictly-future span (t, t+tau]; NaN where the span is incomplete."""
    n = len(returns)
    c1 = np.r_[0.0, np.cumsum(returns)]; c2 = np.r_[0.0, np.cumsum(np.square(returns))]
    fv = np.full(n, np.nan)
    i = np.arange(max(n - tau, 0))
    m = (c1[i + 1 + tau] - c1[i + 1]) / tau
    fv[i] = np.sqrt(np.maximum((c2[i + 1 + tau] - c2[i + 1]) / tau - m ** 2, 0.0))
    return fv


def forward_instability_labels(returns: np.ndarray, tau: int, train_end: int, q: float = 0.95):
    """
    label_t = 1 iff realised vol over the FUTURE span (t, t+tau] exceeds the q-quantile of future-vol
    computed on rows < train_end only.  The span starts strictly after t so it never overlaps a window
    ending at t.  Last `tau` rows are -1 (undefined).
    """
    fv = future_vol(returns, tau)
    thr = np.quantile(fv[:train_end][~np.isnan(fv[:train_end])], q)
    lab = (fv > thr).astype(np.int64); lab[np.isnan(fv)] = -1
    return lab, float(thr)


def platt_fit(scores: np.ndarray, y: np.ndarray, iters: int = 200, lr: float = 0.5):
    """Logistic calibration p = sigmoid(a*s + b) on VALIDATION data (Newton-free GD on standardised s)."""
    mu, sd = scores.mean(), scores.std() + 1e-12
    s = (scores - mu) / sd
    a, b = 1.0, float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * s + b)))
        a -= lr * np.mean((p - y) * s); b -= lr * np.mean(p - y)
    return a, b, mu, sd


def platt_apply(scores, params):
    a, b, mu, sd = params
    return 1 / (1 + np.exp(-(a * (scores - mu) / sd + b)))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def auprc(scores: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(-scores, kind="mergesort"); ys = y[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def alert_metrics(prob: np.ndarray, y: np.ndarray, alert_thr: float, tau: int):
    """AUROC, AUPRC, Brier, false-alarm rate and mean lead time (steps from first alert in an episode to onset)."""
    alert = prob >= alert_thr
    brier = float(np.mean((prob - y) ** 2))
    fa = float(np.mean(alert[y == 0])) if (y == 0).any() else float("nan")
    # an "event" onset = 0->1 transition of the label; lead = steps back (<= tau) to the earliest alert
    onsets = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    leads = []
    for t in onsets:
        w = alert[max(0, t - tau): t]
        if w.any():
            leads.append(len(w) - int(np.argmax(w)))
    return dict(auroc=auroc(prob, y), auprc=auprc(prob, y), brier=brier, false_alarm_rate=fa,
                events=int(len(onsets)), detected=len(leads),
                mean_lead=float(np.mean(leads)) if leads else float("nan"))
