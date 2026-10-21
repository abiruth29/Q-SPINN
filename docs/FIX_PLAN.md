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
| 6 | Grouped AP, bootstrap start range / short series / segment blocks | done (segment-aware blocks used in Exp B CIs) |
| 7 | Encoder fit excludes joins | done |
| 8 | Label / warning cutoffs use only fully-in-training summaries | done |
| 9 | Causal grid seeding + long-gap policy | done |
| 10 | Pre-load contract/protocol check for live demo | done |
| 11 | Prefix-only residual generator through `heavy_path` | done |
| 12 | Real conditional `infer()` (no loss-only work) | done |
| 13 | Matched full-pipeline latency | done (`routing_study.py`; numbers still need the teammate's CPU/GPU run) |
| 14 | Authoritative FI-2010 segment metadata, fail closed | code done; **needs verified segment_lengths.json from the teammate** |
| 15 | Scale-collapse loophole | done (relative residual + window anchor) |
| 16 | Event-based warning lead time / false alarms per unit time | done (`event_alert_metrics`, realised onsets, segment-safe) |
| 17 | Exp A migration; equal baseline budgets | Exp A excluded (guarded); baselines get equal epoch budget/loop/early stopping, no tuning for any model |
| 18 | Smoothing relabel | done |
| 19 | Covariance disabled until observation model exists | unchanged (off) |

## Review 3 (v3.1.1 recheck) — implemented
1. **StepGuard**: global norm computed overflow-safe (float64, max-scaled); clip applied with that norm. Finite-entry gradients whose naive norm overflows are now clipped, not zeroed or discarded. Non-finite entries still skip with no state change.
2. **Outages are fail-closed**: `assert_no_long_outage` raises on ANY flagged outage (> `max_gap` s) before dataset build. Split raw series at gaps instead of building across them.
3. **Event warning evaluation**: one alarm policy (episode start + renewal every tau rows) drives detection, lead and false alarms; at-risk clock = `eligible` rows; warm-up and segment-end events/alarms are CENSORED (reported as `events_censored`, `alarms_censored`), never counted as misses/false alarms; exposure = eligible rows. Rates are per `rows_per_unit_time`, not per hour.
4. External (teammate): verified FI-2010 `segment_lengths.json` with day/stock order and source-file identity; actual split = train days 1-6 / val day 7 / test days 8-10.
5. External (teammate): report actual epochs/updates executed for baselines vs hybrid; paired segment-aware differences vs baselines.

## Review 4 (remaining changes v3.1.1) — implemented in code/docs
01. `split_plan`: scaler and label thresholds restricted to train-window support; 500-row fixture tests.
02. ω classified as current-regime diagnostic; Stage B/C wording, dashboard and docs de-"EWS"-ed.
03. Benefit-trained gate (`expert_benefit`, `benefit_gate_loss`, `--gate_objective`), original router kept as default/comparator.
04. `live_step`: gate-first live inference; ω fresh/cached/unavailable; `--omega_mode`; end-to-end latency logged.
05. Runbook/README/presentation synced; docs tests guard against stale claims.
Evidence-dependent (teammate): 06 verified FI segment metadata, 07 real-crypto outage validation + live/train feature contract, 08 fresh weights + controlled studies, 09 warning-quality study on untouched forward targets.

## Review 5 (v3.1.2 recheck) — implemented
01 presentation 99%/1% tick claims removed (both decks). 02 integrated-ω display replaced by sample-based `omega_fresh_mean`. 03 lean `heavy_infer(return_omega=True)` used live (parity-tested vs `heavy_path`). 04 pyarrow installed, full suite 0 skipped, log in `docs/TEST_LOG_latest.txt`. 05 benefit-vs-legacy default decision awaits teammate's routing study. 06–10 remain external (verified FI segments, real crypto outages + live contract, controlled studies, held-out warning evaluation, live market-specific training).

## Review 6 (latest-archive recheck) — implemented
01 reference doc §3 retired with a LEGACY/INVALID banner; unmeasured DeepLOB and compute claims removed. 02 event-driven live input (`DepthFreshness`), input age + `pred` logged, `cadence` in the input contract; tests for consecutive no-depth ticks, throttling, event gaps, record == live_step. 03 `scripts/release_check.sh` + `QSPINN_REQUIRE_FULL=1` (skips fail) + `requirements-lock.txt`; 116 passed, 0 skipped locally. 04–09 remain data/experiment-dependent (FI segment audit, real crypto validation, benefit-vs-legacy study, controlled benchmarks, held-out warning study, target-market live model).

## Review 7 (teammate's first Windows run)
UTF-8 everywhere (AST test enforces it); suite runner reports failures + non-zero exit; Exp C missing-data error is actionable; `procure_fi2010 --setup2 --segments` checks the manifest before extracting the archive and explains where counts must come from. Open: obtain verified FI-2010 per-stock/day counts (not in the paper); obtain Binance history.
