"""Task 15: intervals for macro-F1.  Block bootstrap (windows are autocorrelated) + seed aggregation."""
import numpy as np


def _macro_f1(pred, true, n=3):
    f = []
    for c in range(n):
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c)); d = 2 * tp + fp + fn
        f.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f))


def block_bootstrap_f1(pred, true, block=200, n_boot=1000, alpha=0.05, seed=0):
    """Moving-block bootstrap CI of macro-F1. Returns (point, lo, hi)."""
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(true); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); starts_max = max(n - block, 1)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_max, nb)
        idx = (st[:, None] + np.arange(block)[None]).ravel()[:n]
        vals[b] = _macro_f1(pred[idx], true[idx])
    return _macro_f1(pred, true), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(pred_a, pred_b, true, block=200, n_boot=1000, seed=0):
    """CI of F1(a) - F1(b) on the same resampled blocks. Difference is significant if CI excludes 0."""
    pred_a, pred_b, true = map(np.asarray, (pred_a, pred_b, true))
    n = len(true); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); sm = max(n - block, 1)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.integers(0, sm, nb)[:, None] + np.arange(block)[None]).ravel()[:n]
        d[b] = _macro_f1(pred_a[idx], true[idx]) - _macro_f1(pred_b[idx], true[idx])
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def seed_summary(values):
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))
