# Q-SPINN v3.1 — Correction Plan

Tracking document for the `fix/v3.1-corrections` branch. Each item links to the
section of the project report (Chapter 7) that describes the problem.

## 1. Evaluation protocol (Report §7.1)
- [x] Replace the concatenated-fold split with FI-2010 Setup 2: train on days 1–7
      (`Train_Dst_*_CF_7`, end of day 7 held out for validation), test on days 8–10
      (`Test_Dst_*_CF_7`, `CF_8`, `CF_9`).
- [x] Build windows within each stock-day segment (no window crosses a stock or day boundary).
- [x] Report majority-class macro-F1 alongside macro-F1.

## 2. Gate and fusion (Report §7.2)
- [x] Execute the heavy path only on gated samples (true conditional computation).
- [x] Allow the fusion weight to reach 1 when the gate fires.
- [x] Add a rolling-volatility router as a baseline for the SNN gate.
- [ ] Gate-rate sweep (0–100%) and wall-clock latency on CPU and GPU.

## 3. Heavy path vs. framework (Report §7.3)
- [x] Independent parametrisations for S (symmetric) and H (Hermitian).
- [x] Propagate with exp(AΔt), A = S − iH, so prediction, residuals and ω share one generator.
- [x] Normalise only the read-out, not ψ at every step.
- [ ] Add `van_loan.B_R` and `cov_head` to the Stage B optimiser; replace the covariance
      residual with an innovation (Kalman) negative log-likelihood.
- [ ] Return a gradient for the surrogate slope α, or fix it as a hyperparameter.
- [x] Fit the delta-encoding threshold on training statistics (not per batch).
- [x] Recursive filter relabelled as coefficient *smoothing* (no state-space claim); a proper state-space model remains future work.

## 4. Early-warning indicator (Report §7.4)
- [x] Forward-looking instability label over horizon τ (no overlap with the input window).
- [x] Evaluate with AUROC, AUPRC, Brier score and lead time vs. a rolling-volatility baseline.

## 5. Experiment C (Report §7.5)
- [x] Align assets on a common one-second clock before windowing.
- [x] Report the gauge-invariant loop flux instead of individual edge phases.
- [x] Rename "ofi" to signed trade volume (or compute true OFI from depth data).

## 6. Baselines and reporting
- [x] Reimplement DeepLOB on the same split; equal tuning budget for all models.
- [ ] Five seeds at every horizon; bootstrap confidence intervals on macro-F1.
- [ ] Regenerate README, reference document and results files from the new runs;
      remove the misquoted DeepLOB comparison column.

## 7. Repository hygiene
- [x] Replace `validate_*.py` (hard-coded Windows paths) with runnable unit tests.
- [ ] Live pipeline: train on target-market data; use 20-step windows and the full feature set.  *(Only a contract guard + 20-step window + `--demo` label exist; a model retrained on live crypto inputs does NOT.)*


---
**Status (superseded by the review-2 section below):** first-pass code changes for phases 0-3. Unchecked items need teammate runs (five-seed benchmarks, latency, CPU/GPU, regenerated documents) - see `RUNBOOK_TEAMMATE.md`. Phase 4 is deferred.


## Review 2 (commit ec9b0bd) — verified defects and their status

| Rank | Item | Status |
|---|---|---|
| 1 | Versioned run state / checkpoints; legacy weights archived | done (`PROTOCOL_VERSION`) |
| 2 | Generated claims, dashboard mock data, diagnostic labels | done |
| 3 | Aggregate gradient-norm guard | done |
| 4 | Surrogate slope fixed (declared hyperparameter) | done |
| 5 | Tie-safe routing budget | done |
| 6 | Grouped AP, bootstrap start range / short series / segment blocks | done (segment-aware bootstrap wired in driver where segment ids exist) |
| 7 | Encoder fit excludes joins | done |
| 8 | Label / warning cutoffs use only fully-in-training summaries | done |
| 9 | Causal grid seeding + long-gap policy | done |
| 10 | Pre-load contract/protocol check for live demo | done |
| 11 | Prefix-only residual generator through `heavy_path` | done |
| 12 | Real conditional `infer()` (no loss-only work) | done |
| 13 | Matched full-pipeline latency | done (driver) |
| 14 | Authoritative FI-2010 segment metadata, fail closed | see below |
| 15 | Scale-collapse loophole | done (relative residual + window anchor) |
| 16 | Event-based warning lead time / false alarms per unit time | see below |
| 17 | Exp A migration; equal baseline budgets | see below |
| 18 | Smoothing relabel | done |
| 19 | Covariance disabled until observation model exists | unchanged (off) |
