"""Task 11: common 1 s clock; train-only label thresholds; forward label has no look-back."""
import numpy as np, pandas as pd
from binance_align import align_on_grid, frame_to_features, forward_labels

COLS = ["mid_sum", "mid_count", "ofi", "buy_vol", "sell_vol", "trade_count", "total_pq", "total_qty"]


def _frame(seconds, price):
    n = len(seconds)
    d = dict(mid_sum=price * np.ones(n), mid_count=np.ones(n), ofi=np.ones(n), buy_vol=np.ones(n),
             sell_vol=np.zeros(n), trade_count=np.ones(n), total_pq=price * np.ones(n), total_qty=np.ones(n))
    return pd.DataFrame(d, index=pd.Index(seconds, name="second"))[COLS]


def test_assets_share_one_clock_and_row_t_is_second_t():
    btc = _frame(np.arange(100, 200), 100.0)
    sol = _frame(np.r_[np.arange(105, 150), np.arange(160, 190)], 20.0)   # gaps + later start
    (b, s), grid = align_on_grid([btc, sol])
    assert grid[0] == 105 and grid[-1] == 189 and len(b) == len(s) == len(grid)
    assert (b.index == s.index).all()
    assert s.loc[150, "stale"] == 1 and s.loc[150, "trade_count"] == 0 and s.loc[150, "mid_price"] == 20.0
    assert s.loc[149, "stale"] == 0


def test_no_nans_after_features():
    sol = _frame(np.r_[np.arange(0, 10), np.arange(20, 30)], 5.0)
    (g,), _ = align_on_grid([sol])
    f = frame_to_features(g)
    assert np.isfinite(f).all() and f.shape == (30, 6)
    assert f[12, 0] == 0.0                      # empty second -> zero return


def test_forward_label_uses_only_future_and_train_thresholds():
    r = np.zeros(100); r[51:56] = 1.0           # spike strictly after t=50
    lab, _ = forward_labels(r, horizon=5, train_end=40)
    assert lab[50] == 2                          # sees the spike in (50, 55]
    assert (lab[-5:] == -1).all()                # tail has no full future
    r2 = r.copy(); r2[0:5] = 1.0                 # past-only spike must not affect label at t=10
    lab2, _ = forward_labels(r2, horizon=5, train_end=40)
    assert lab2[10] == lab[10]                    # past-only spike leaves the t=10 label unchanged
    # thresholds depend only on [0, train_end): changing the test tail leaves them unchanged
    a = np.random.default_rng(0).normal(size=300)
    b = a.copy(); b[250:] += 100
    assert forward_labels(a, 5, 200)[1] == forward_labels(b, 5, 200)[1]
