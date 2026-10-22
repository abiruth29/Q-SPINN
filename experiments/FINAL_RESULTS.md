> **LEGACY RESULTS - NOT VALID UNDER THE v3.1 PROTOCOL.** Produced before the v3.1 corrections (concatenated-fold split with repeated days, batch-dependent encoder, random dt, noisy/attenuated routing, per-batch gate averaging, unaligned crypto series). Do not cite. They are kept only for comparison; regenerate with `docs/RUNBOOK_TEAMMATE.md`. Statements below about DeepLOB gains, crash probability, compute savings or BTC lead-lag are retracted.

# Q-SPINN — Final Experimental Results

**Date:** 2026-09-18
**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6.4 GB VRAM)

This document summarizes the final outcomes of the Phase 1 Real-Data validation for Q-SPINN, covering **Experiment A** (Binance LOB proxy) and **Experiment B** (FI-2010 Full Hybrid).

---

## Experiment A: SNN Gate on Binance
**Goal:** Validate that the spiking gate can learn real-world microstructure features (OFI, volume, spread) to predict volatility regimes, replacing the toy GBM generator (SWAP-01).

- **Data:** 349k windows (BTC, ETH, SOL)
- **Label:** 2σ Realized Volatility Proxy (SWAP-15)
- **Train Time:** ~30-45 minutes (primarily CPU Pandas aggregation)
- **Results:**
  - **Macro-F1:** `0.408` (baseline `0.343`)
  - **Neurons silent per tick:** `99.2%` (extreme sparsity achieved)
- **Outcome:** The bio-inspired delta-encoding mechanism successfully isolated structural volatility regimes from noise, validating the SNN gate for real market data.

---

## Experiment B: Full Hybrid on FI-2010
**Goal:** Validate the full Q-SPINN pipeline (SNN + Complex Covariance + PINN) on the academic standard FI-2010 dataset.

- **Data:** 1,017,738 train windows (144 features)
- **Label:** `label_k10` (mid-price direction)
- **Train Time:** ~25 minutes total (3 stages + Ablation)
- **Optimizations Applied:** 
  1. Memory-safe unfold views on GPU, batch size increased from 256 to 2048 to saturate GPU and eliminate PyTorch micro-kernel launch overhead.
  2. Phase 1 Optimizations (from `Q_SPINN_Experiments_and_Changes.md`): Soft-sigmoid gating, inverse-frequency class-balanced loss, and reduced sparsity weight penalty (`1e-4`).

### Ablation Results
| Model | Accuracy | Macro-F1 | Gate Rate | Hermiticity ‖H−H†‖ |
|---|---|---|---|---|
| LSTM baseline | 0.813 | 0.772 | n/a | n/a |
| SNN only | 0.804 | 0.751 | 0.000 | 0.00e+00 |
| PINN only (always awake) | 0.842 | 0.799 | 1.000 | 0.00e+00 |
| **Q-SPINN (full hybrid)** | **0.851** | **0.811** | **0.007** | **0.00e+00** |
| Majority baseline | 0.642 | — | — | — |

**Interpretation:**
The full Q-SPINN hybrid achieved a Macro-F1 of `0.811`, significantly outperforming both the standard LSTM (`0.772`) and the heavy "always awake" PINN (`0.799`). 
By replacing hard-binary spikes with soft-sigmoid gating and addressing class imbalances, the model learned a far superior blending logic. Most importantly, it achieved the highest predictive performance across all models while restricting the heavy PINN path to trigger on only **0.7%** of ticks (`Gate Rate = 0.007`). This confirms the core thesis of the paper beyond expectation: a **99.3% reduction** in compute for dense deep learning layers in HFT pipelines, actually *increasing* accuracy by dynamically skipping noise.
