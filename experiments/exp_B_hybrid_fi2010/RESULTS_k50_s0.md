> **LEGACY RESULTS - NOT VALID UNDER THE v3.1 PROTOCOL.** Produced before the v3.1 corrections (concatenated-fold split with repeated days, batch-dependent encoder, random dt, noisy/attenuated routing, per-batch gate averaging, unaligned crypto series). Do not cite. They are kept only for comparison; regenerate with `docs/RUNBOOK_TEAMMATE.md`. Statements below about DeepLOB gains, crash probability, compute savings or BTC lead-lag are retracted.

# Experiment B — Full Q-SPINN Hybrid on FI-2010 (k=50, Seed=0)

**Run date:** 2026-10-19T08:51:07.600956  
**Device:** NVIDIA GeForce RTX 3060 Laptop GPU  
**Source code:** [`train.py`](./train.py)

---

## Dataset

| Property | Value |
|---|---|
| Source | FI-2010 NoAuction Z-Score (`data/reference/fi2010/fi2010_noauction_zscore.parquet`) |
| Features | 144 (10 LOB levels × 4 quantities × 1 stock, + 4 derived) |
| Label | `label_k50` (3-class mid-price direction, 50-tick horizon) |
| Seed | 0 |
| Train | Folds 1–7 |
| Val | Fold 8 |
| Test | Fold 9 |
| Normalization | Pre-normalized by FI-2010 authors (Z-score, no leakage) |
| Regime proxy | realized_vol > μ + 2.0σ (rolling 200-tick window) |

## Ablation Results

| Model | Accuracy | Macro-F1 | Gate Rate | Hermiticity ‖H−H†‖ | $\omega(A_t)$ (EWS Margin) | Crash Prob ($\omega>0$) |
|---|---|---|---|---|---|---|
| LSTM baseline | 0.800 | 0.799 | n/a | nan | n/a | n/a |
| SNN only | 0.793 | 0.790 | 0.000 | 0.00e+00 | 0.0000 | 0.0% |
| PINN only (always awake) | 0.872 | 0.870 | 1.000 | 0.00e+00 | -0.0125 | 14.4% |
| Q-SPINN (full hybrid) | 0.797 | 0.794 | 0.053 | 0.00e+00 | -0.0125 | 14.4% |
| **Majority baseline** | **0.374** | — | — | — | — | — |

## Training Protocol

Three-stage training was used:

1. **Stage A (100 epochs):** SNN gate cheap head only (heavy path frozen)
2. **Stage B (100 epochs):** Heavy path only + EWS regularization (SNN frozen)
3. **Stage C (100 epochs):** Gate head only with sparsity calibration (everything else frozen)

## Key Findings

- The Hermiticity check `‖H − H†‖` is exactly 0.00e+00 by construction.
- Gate selectivity triggers on true regime shifts while keeping heavy compute sparse (~5-10%).
- Early Warning System: $\omega(A_t) \le 0$ during stable markets, transitioning to $\omega > 0$ during volatility bursts.
