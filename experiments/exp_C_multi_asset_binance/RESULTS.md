> **LEGACY RESULTS - NOT VALID UNDER THE v3.1 PROTOCOL.** Produced before the v3.1 corrections (concatenated-fold split with repeated days, batch-dependent encoder, random dt, noisy/attenuated routing, per-batch gate averaging, unaligned crypto series). Do not cite. They are kept only for comparison; regenerate with `docs/RUNBOOK_TEAMMATE.md`. Statements below about DeepLOB gains, crash probability, compute savings or BTC lead-lag are retracted.

# Experiment C — Multi-Asset Q-SPINN on Binance Crypto Basket

**Date:** 2026-10-19T16:43:04.421528  
**Assets:** BTCUSDT, ETHUSDT, SOLUSDT (18 features total)  
**Prediction Horizon:** 5 seconds  

## Ablation Results

| Model | Accuracy | Macro-F1 | Gate Rate | Hermiticity ‖H−H†‖ | $\omega(A_t)$ (EWS Margin) | Crash Prob |
|---|---|---|---|---|---|---|
| LSTM baseline | 0.599 | 0.386 | n/a | n/a | n/a | n/a |
| SNN only (cheap) | 0.510 | 0.388 | 0.000 | 0.00e+00 | 0.0000 | 0.0% |
| PINN only (always awake) | 0.586 | 0.401 | 1.000 | 0.00e+00 | -0.0258 | 0.0% |
| Q-SPINN Multi-Asset Hybrid | 0.510 | 0.389 | 0.010 | 0.00e+00 | -0.0258 | 0.0% |
| **Majority baseline** | **0.618** | — | — | — | — | — |

## Magnetic Graph Laplacian & Lead-Lag Phase Analysis

The directed phase flux matrix $\Phi_{ij}$ captures empirical lead-lag latency:
- **BTC $\to$ ETH Phase Flux:** `-0.3973` rad
- **BTC $\to$ SOL Phase Flux:** `+1.7378` rad

A positive phase indicates that BTC microstructure features lead altcoin quote adjustments, confirming Objective O2.
