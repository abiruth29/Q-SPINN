"""
Exp C label / data diagnostics (CPU only, ~2-3 min on the real data).  Run BEFORE trusting any Exp C accuracy:

    python experiments/exp_C_multi_asset_binance/label_diagnostics.py [--max_ticks 350000] [--out label_diagnostics.json]

It answers: how much of the 3-class 5-second label is "the price did not move" / stale-price persistence, how
large the label thresholds are in dollars, whether 1-second returns are autocorrelated, and how a trivial
past-return lookup rule performs overall and split by stale/non-stale label windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from binance_align import forward_labels     # noqa: E402


def diagnose(mid: np.ndarray, stale: np.ndarray, horizon: int = 5, seq_len: int = 20, train_frac: float = 0.70,
             lookback: int = 5) -> dict:
    """mid, stale: BTC mid price and stale flag on the common 1 s grid (stale=1: no trade that second)."""
    mid = np.asarray(mid, dtype=np.float64); stale = np.asarray(stale, dtype=np.float64) > 0
    n = len(mid)
    logret = np.r_[0.0, np.diff(np.log(mid))]
    train_end = int(train_frac * (n - seq_len - horizon)) + seq_len
    lab, (lo, hi) = forward_labels(logret, horizon, train_end)
    valid = lab >= 0
    d = {}
    d["n_seconds"] = int(n)
    d["median_price"] = float(np.median(mid))
    d["label_threshold_logret_mean"] = [float(lo), float(hi)]
    d["label_threshold_usd_over_horizon"] = [float(lo * horizon * np.median(mid)), float(hi * horizon * np.median(mid))]
    d["zero_return_share_all"] = float(np.mean(logret == 0))
    d["zero_return_share_nonstale"] = float(np.mean(logret[~stale] == 0))
    # windows (t, t+H]: any stale second inside the label support?
    cs = np.r_[0, np.cumsum(stale)]
    idx = np.arange(n - horizon)
    stale_in = np.zeros(n, dtype=bool)
    stale_in[idx] = (cs[idx + 1 + horizon] - cs[idx + 1]) > 0
    all_stale = np.zeros(n, dtype=bool)
    all_stale[idx] = (cs[idx + 1 + horizon] - cs[idx + 1]) == horizon
    d["share_label_windows_with_any_stale"] = float(stale_in[valid].mean())
    d["share_label_windows_all_stale"] = float(all_stale[valid].mean())
    dprice = np.zeros(n); dprice[idx] = mid[idx + horizon] - mid[idx]
    d["share_windows_exact_zero_price_change"] = float((dprice[valid] == 0).mean())
    cls = lambda m: [float(np.mean(lab[m & valid] == c)) if (m & valid).any() else float("nan") for c in range(3)]
    d["class_shares_all"] = cls(np.ones(n, bool))
    d["class_shares_if_any_stale"] = cls(stale_in)
    d["class_shares_if_no_stale"] = cls(~stale_in)
    # autocorrelation of 1 s returns (non-stale seconds only, lags 1..5)
    r = logret.copy(); r[stale] = np.nan
    ac = []
    for k in range(1, 6):
        a, b = r[:-k], r[k:]; m = ~np.isnan(a) & ~np.isnan(b)
        ac.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan"))
    d["return_autocorr_lag1_to_5"] = ac
    # trivial rule: past `lookback`-second return tertile -> train-majority label
    cs_r = np.r_[0.0, np.cumsum(logret)]
    past = np.full(n, np.nan); past[lookback:] = cs_r[lookback:n] - cs_r[:n - lookback]   # return over the previous `lookback` s
    ok = valid & ~np.isnan(past)
    tr = ok.copy(); tr[train_end:] = False
    te = ok.copy(); te[:train_end] = False
    qs = np.quantile(past[tr], [1 / 3, 2 / 3])
    b = np.digitize(past, qs)
    table = [int(np.bincount(lab[tr & (b == k)], minlength=3).argmax()) for k in range(3)]
    pred = np.array(table)[b]
    d["lookup_table_bin_to_class"] = table
    d["lookup_acc_test"] = float((pred[te] == lab[te]).mean())
    d["lookup_acc_test_if_any_stale"] = float((pred[te & stale_in] == lab[te & stale_in]).mean()) if (te & stale_in).any() else float("nan")
    d["lookup_acc_test_if_no_stale"] = float((pred[te & ~stale_in] == lab[te & ~stale_in]).mean()) if (te & ~stale_in).any() else float("nan")
    d["majority_share_test"] = float(np.bincount(lab[te], minlength=3).max() / te.sum())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_ticks", type=int, default=350_000)
    ap.add_argument("--out", default=str(HERE / "label_diagnostics.json"))
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("expC_diag", HERE / "train.py")
    C = importlib.util.module_from_spec(spec); sys.modules["expC_diag"] = C; spec.loader.exec_module(C)
    from binance_align import align_on_grid
    frames = [C.engineer_features_for_symbol(C.DATA_DIR / f"symbol={s}", a.max_ticks) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
    aligned, grid = align_on_grid(frames)
    btc = aligned[0]
    d = diagnose(btc["mid_price"].values, btc["stale"].values, horizon=C.HORIZON, seq_len=C.SEQ_LEN, train_frac=C.TRAIN_FRAC)
    for k, v in d.items():
        print(f"{k:42s} {v}")
    Path(a.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
