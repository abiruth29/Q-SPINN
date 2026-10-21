"""
experiments/binance_align.py
============================
Task 11: put BTC / ETH / SOL on ONE common 1-second clock before windowing, and fit label
thresholds on the training span only.  Pure numpy/pandas so it is unit-testable on synthetic data.

Old behaviour: each asset's non-empty seconds were stacked after `reset_index(drop=True)` and then
truncated to the shortest series, so row t of BTC and row t of SOL could be minutes apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_COLS = ["signed_volume", "buy_vol", "sell_vol", "trade_count", "total_pq", "total_qty"]
PRICE_COLS = ["mid_sum", "mid_count"]


def align_on_grid(frames):
    """
    frames: list of DataFrames indexed by integer unix-second with the raw per-second aggregate
    columns.  Returns (list of frames on the shared grid, grid index).  Grid = [max(first), min(last)]
    at 1 s.  Empty seconds get zero flow and a carried-forward mid price (flag `stale`=1).
    """
    lo = max(int(f.index.min()) for f in frames)
    hi = min(int(f.index.max()) for f in frames)
    if hi <= lo:
        raise ValueError("assets have no overlapping time range")
    grid = pd.RangeIndex(lo, hi + 1, name="second")
    out = []
    for f in frames:
        g = f.reindex(grid)
        stale = g["mid_count"].isna().astype(np.float32)
        g[FLOW_COLS] = g[FLOW_COLS].fillna(0.0)
        mid = (g["mid_sum"] / g["mid_count"].where(g["mid_count"] > 0)).ffill().bfill()
        g = g.assign(mid_price=mid, stale=stale)
        out.append(g)
    return out, grid


def frame_to_features(g: pd.DataFrame) -> np.ndarray:
    """Six per-asset features on the aligned grid (same definitions as before)."""
    mid = g["mid_price"]
    logret = np.log(mid / mid.shift(1)).fillna(0.0)
    rv = logret.rolling(20, min_periods=1).std().fillna(0.0)
    vwap = g["total_pq"] / g["total_qty"].clip(lower=1e-8)
    vwap = vwap.where(g["total_qty"] > 0, mid)
    bp = (g["buy_vol"] / (g["buy_vol"] + g["sell_vol"]).clip(lower=1e-8)).where(
        (g["buy_vol"] + g["sell_vol"]) > 0, 0.5)
    spread = (vwap - mid).abs() / mid.clip(lower=1e-8)
    return np.column_stack([logret, g["signed_volume"], rv, bp, g["trade_count"], spread]).astype(np.float32)


def forward_labels(returns: np.ndarray, horizon: int, train_end: int, q=(33.33, 66.67)):
    """
    3-class label from the mean of returns over (t, t+horizon]; the last `horizon` rows have no
    full future and are marked -1.  Thresholds are percentiles of the TRAIN span [0, train_end)
    only (previously fit on the whole series).
    """
    n = len(returns)
    csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])
    fut = np.full(n, np.nan)
    idx = np.arange(n - horizon)
    fut[idx] = (csum[idx + 1 + horizon] - csum[idx + 1]) / horizon
    valid_train = fut[:train_end][~np.isnan(fut[:train_end])]
    lo, hi = np.percentile(valid_train, q[0]), np.percentile(valid_train, q[1])
    lab = np.ones(n, dtype=np.int64)
    lab[fut > hi] = 2
    lab[fut < lo] = 0
    lab[np.isnan(fut)] = -1
    return lab, (lo, hi)
