"""Review-2 ranks 7-9: cutoff-safe thresholds, join-safe encoder fit, causal grid seeding."""
import numpy as np, pandas as pd, torch
from binance_align import align_on_grid, forward_labels, gap_fraction, frame_to_features
import routing as R
from qspinn_core import SpikeEncoder
from test_phase1_binance_align import _frame


def test_label_thresholds_ignore_post_cutoff_returns():
    rng = np.random.default_rng(0); r = rng.normal(size=300)
    base = forward_labels(r, 5, 100)[1]
    r2 = r.copy(); r2[100:105] += 50            # returns AT/after the cutoff only
    assert forward_labels(r2, 5, 100)[1] == base
    r3 = r.copy(); r3[:100] += 0.0; r3[100:] = rng.normal(size=200) * 100
    assert forward_labels(r3, 5, 100)[1] == base


def test_warning_thresholds_ignore_post_cutoff_returns():
    rng = np.random.default_rng(1); r = rng.normal(size=300)
    thr = R.forward_instability_labels(r, 5, 100, 0.9)[1]
    r2 = r.copy(); r2[100:105] = 1e3
    assert R.forward_instability_labels(r2, 5, 100, 0.9)[1] == thr


def test_encoder_fit_invariant_to_segment_offsets_with_valid_mask():
    rng = np.random.default_rng(2)
    seg = np.repeat([0, 1, 2], 100)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    X2 = X.copy(); X2[100:200] += 75.0; X2[200:] -= 40.0            # per-segment price offsets
    vp = seg[1:] == seg[:-1]
    t1 = SpikeEncoder(4).fit(X, valid_pair=vp).threshold
    t2 = SpikeEncoder(4).fit(X2, valid_pair=vp).threshold
    assert torch.allclose(t1, t2, atol=1e-4)
    assert not torch.allclose(t1, SpikeEncoder(4).fit(X2).threshold, atol=1e-1)   # unmasked would differ


def test_leading_hole_uses_past_price_not_future():
    btc = _frame(np.arange(0, 300), 100.0)
    sol = _frame(np.r_[np.arange(50, 99), np.arange(101, 300)], 10.0)     # price 10 at <=98, hole at 99,100
    sol.loc[101:, ["mid_sum", "total_pq"]] = 999.0                          # later price differs
    (b, s), grid = align_on_grid([btc, sol.loc[sol.index >= 0]])
    assert s.loc[99, "mid_price"] == 10.0 and s.loc[100, "mid_price"] == 10.0   # last KNOWN price, not 999
    # grid starts at 50; hole at the very start must not borrow a later price either
    sol2 = _frame(np.r_[np.arange(0, 3), np.arange(5, 300)], 10.0); sol2.loc[5:, ["mid_sum", "total_pq"]] = 999.0
    btc2 = _frame(np.arange(4, 300), 100.0)
    (b2, s2), g2 = align_on_grid([btc2, sol2])
    assert g2[0] == 4 and s2.loc[4, "mid_price"] == 10.0                    # seeded from the pre-grid observation


def test_long_gap_flagged():
    btc = _frame(np.arange(0, 2000), 100.0)
    sol = _frame(np.r_[np.arange(0, 100), np.arange(1500, 2000)], 10.0)
    (b, s), _ = align_on_grid([btc, sol], max_gap=300)
    assert s.loc[500, "long_gap"] == 1 and s.loc[150, "long_gap"] == 0
    assert 0.1 < gap_fraction([b, s]) < 0.6
