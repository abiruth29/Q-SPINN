> **LEGACY RESULTS - NOT VALID UNDER THE v3.1 PROTOCOL.** Produced before the v3.1 corrections (concatenated-fold split with repeated days, batch-dependent encoder, random dt, noisy/attenuated routing, per-batch gate averaging, unaligned crypto series). Do not cite. They are kept only for comparison; regenerate with `docs/RUNBOOK_TEAMMATE.md`. Statements below about DeepLOB gains, crash probability, compute savings or BTC lead-lag are retracted.

# Experiment A — SNN Gate on Binance Trade Features

**Run date:** 2026-09-18T21:02:59.348539  
**Device:** NVIDIA GeForce RTX 3060 Laptop GPU  
**Source code:** [`train.py`](./train.py)

---

## Dataset

| Property | Value |
|---|---|
| Source | Binance historical trades (`data/curated/trades/`) |
| Assets | BTCUSDT · ETHUSDT · SOLUSDT |
| Features per asset | 6 (log_return, OFI, realized_vol, buy_pressure, trade_count, vwap_spread) |
| Total features | 18 |
| Train windows | 349,982 |
| Val windows | 74,996 |
| Test windows | 74,997 |
| Regime proxy | realized_vol > mu + 2.0sigma (rolling 200-tick window) |
| Regime % in train | 14.4% |

## Results

| Metric | Value |
|---|---|
| **Accuracy** | **0.408** |
| **Macro-F1** | **0.408** |
| Majority-class baseline | 0.343 |
| Hidden spike rate | 0.0079 |
| Neurons silent per tick | 99.2% |

## Training History

| Epoch | Loss | Val Macro-F1 |
|---|---|---|
| 1 | 1.7663 | 0.360 |
| 2 | 1.7611 | 0.382 |
| 3 | 1.7587 | 0.392 |
| 4 | 1.7576 | 0.396 |
| 5 | 1.7569 | 0.390 |
| 6 | 1.7565 | 0.394 |
| 7 | 1.7561 | 0.400 |
| 8 | 1.7555 | 0.401 |
| 9 | 1.7552 | 0.402 |
| 10 | 1.7547 | 0.405 |
| 11 | 1.7543 | 0.396 |
| 12 | 1.7541 | 0.396 |
| 13 | 1.7539 | 0.403 |
| 14 | 1.7539 | 0.387 |
| 15 | 1.7537 | 0.399 |
| 16 | 1.7535 | 0.399 |
| 17 | 1.7533 | 0.406 |
| 18 | 1.7533 | 0.403 |
| 19 | 1.7531 | 0.406 |
| 20 | 1.7530 | 0.394 |
| 21 | 1.7529 | 0.406 |
| 22 | 1.7527 | 0.405 |
| 23 | 1.7527 | 0.397 |
| 24 | 1.7524 | 0.401 |
| 25 | 1.7526 | 0.390 |

## Interpretation

This experiment validates that the SNN gate architecture can learn from **real Binance trade data**,
replacing the synthetic GBM generator used in the toy (SWAP-01 ✅).

The key improvements over the toy:

1. **Real OFI signal** — Order-flow imbalance computed from actual buy/sell classification
   rather than simulated AR(1) process.
2. **Train-window normalization** — Scaler fitted exclusively on the train split (SWAP-02 ✅),
   eliminating the future leakage present in the toy.
3. **Realized-volatility regime proxy** — The 2σ threshold replaces the synthetic ground-truth
   mask with a fully observable, reproducible label (SWAP-15 partial ✅).

## Known Limitations

- SWAP-03 still pending: delta encoding threshold is fixed. A Phase 2 ablation should compare
  delta vs. rate coding vs. latency-to-first-spike.
- SWAP-15 is a proxy, not ground truth. The regime supervision accuracy depends on whether
  volatility spikes correlate with the true structural shifts in BTC/ETH/SOL.
- Single seed, no walk-forward CV (SWAP-06/12 — deferred to Phase 3).

## SWAP Status

| # | Status | Note |
|---|---|---|
| SWAP-01 | ✅ Done | Real Binance trade data |
| SWAP-02 | ✅ Done | Scaler on train window only |
| SWAP-03 | ⏳ Pending | Fixed delta threshold |
| SWAP-15 | ⚠️ Partial | 2σ vol proxy — guide sign-off needed |
