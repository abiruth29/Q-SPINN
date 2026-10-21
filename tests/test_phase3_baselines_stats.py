import numpy as np, torch
from baselines import DeepLOB, count_params, train_classifier, predict
from stats import block_bootstrap_f1, paired_bootstrap_diff, seed_summary


def test_deeplob_shapes_and_trains_on_synthetic():
    torch.manual_seed(0)
    X = torch.randn(256, 20, 144); y = (X[:, -1, 0] > 0).long() + (X[:, -1, 1] > 0.5).long()
    m = DeepLOB(); assert count_params(m) > 1e4 and m(X[:4]).shape == (4, 3)
    f1 = lambda p, t: float((p == t).float().mean())
    m = train_classifier(m, X[:200], y[:200], X[200:], y[200:], torch.device("cpu"), f1, epochs=2, bs=64)
    assert predict(m, X[200:], torch.device("cpu")).shape == (56,)


def test_bootstrap_ci_contains_point_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    def data(n):
        t = rng.integers(0, 3, n); p = np.where(rng.random(n) < 0.7, t, rng.integers(0, 3, n)); return p, t
    p1, t1 = data(1000); p2, t2 = data(20000)
    f1, lo1, hi1 = block_bootstrap_f1(p1, t1, n_boot=200); f2, lo2, hi2 = block_bootstrap_f1(p2, t2, n_boot=200)
    assert lo1 <= f1 <= hi1 and lo2 <= f2 <= hi2 and (hi2 - lo2) < (hi1 - lo1)


def test_paired_diff_detects_real_gap_and_not_identical():
    rng = np.random.default_rng(1); t = rng.integers(0, 3, 4000)
    good = np.where(rng.random(4000) < .9, t, 0); bad = np.where(rng.random(4000) < .5, t, 0)
    m, lo, hi = paired_bootstrap_diff(good, bad, t, n_boot=200); assert lo > 0
    m, lo, hi = paired_bootstrap_diff(good, good, t, n_boot=100); assert lo == hi == 0.0


def test_seed_summary():
    s = seed_summary([0.7, 0.8, 0.9]); assert abs(s["mean"] - 0.8) < 1e-9 and s["n"] == 3
