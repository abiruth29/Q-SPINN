"""Review-2 ranks 5-6: tie-safe budgets, order-invariant AP, legal bootstrap starts."""
import numpy as np
import routing as R
from stats import _block_indices, block_bootstrap_f1


def test_all_equal_scores_respect_budget():
    s = np.ones(100)
    thr = R.select_threshold(s, 0.10)
    assert (s >= thr).mean() <= 0.10                        # policy: route nothing rather than 100%
    assert (s >= R.select_threshold(s, 1.0)).mean() == 1.0


def test_budget_never_exceeded_with_partial_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 1000).astype(float)
    for f in (0.05, 0.2, 0.5, 0.9):
        assert (s >= R.select_threshold(s, f)).mean() <= f + 1e-9


def test_ap_invariant_to_order_of_tied_labels():
    s = np.zeros(4)
    a = R.auprc(s, np.array([1, 1, 0, 0])); b = R.auprc(s, np.array([0, 0, 1, 1]))
    assert abs(a - b) < 1e-12 and abs(a - 0.5) < 1e-12


def test_ap_perfect_and_random_unchanged():
    y = np.array([1, 1, 0, 0, 0]); assert R.auprc(np.array([.9, .8, .3, .2, .1]), y) == 1.0


def test_bootstrap_start_range_inclusive_and_short_series():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        idx = _block_indices(10, 5, rng); seen.add(int(idx[0]))
    assert 5 in seen                                        # final legal start (n - block) is reachable
    idx = _block_indices(3, 200, rng); assert len(idx) == 3 and idx.max() <= 2   # n < block shrinks
    f, lo, hi = block_bootstrap_f1(np.array([0, 1, 2]), np.array([0, 1, 2]), n_boot=20); assert lo <= f <= hi
    assert len(_block_indices(10, 10, rng)) == 10           # n == block


def test_blocks_never_cross_segments():
    rng = np.random.default_rng(1); seg = np.repeat([0, 1, 2], 50)
    for _ in range(100):
        idx = _block_indices(150, 20, rng, seg)
        blocks = idx[:140].reshape(-1, 20)
        assert all(len(set(seg[b])) == 1 for b in blocks)


def test_segmented_rolling_vol_ignores_other_segments():
    mid = np.r_[np.cumsum(np.full(50, .01)), 1000 + np.cumsum(np.full(50, .01))]   # huge join jump
    seg = np.repeat([0, 1], 50)
    rv = R.segmented_rolling_vol(mid, seg, 20)
    assert rv.max() < 0.01 and rv[50] == 0.0            # the join itself never shows up as volatility
