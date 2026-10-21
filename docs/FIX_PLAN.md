# Q-SPINN v3.1 — Correction Plan

Tracking document for the `fix/v3.1-corrections` branch. Each item links to the
section of the project report (Chapter 7) that describes the problem.

## 1. Evaluation protocol (Report §7.1)
- [ ] Replace the concatenated-fold split with FI-2010 Setup 2: train on days 1–7
      (`Train_Dst_*_CF_7`, end of day 7 held out for validation), test on days 8–10
      (`Test_Dst_*_CF_7`, `CF_8`, `CF_9`).
- [ ] Build windows within each stock-day segment (no window crosses a stock or day boundary).
- [ ] Report majority-class macro-F1 alongside macro-F1.

## 2. Gate and fusion (Report §7.2)
- [ ] Execute the heavy path only on gated samples (true conditional computation).
- [ ] Allow the fusion weight to reach 1 when the gate fires.
- [ ] Add a rolling-volatility router as a baseline for the SNN gate.
- [ ] Gate-rate sweep (0–100%) and wall-clock latency on CPU and GPU.

## 3. Heavy path vs. framework (Report §7.3)
- [ ] Independent parametrisations for S (symmetric) and H (Hermitian).
- [ ] Propagate with exp(AΔt), A = S − iH, so prediction, residuals and ω share one generator.
- [ ] Normalise only the read-out, not ψ at every step.
- [ ] Add `van_loan.B_R` and `cov_head` to the Stage B optimiser; replace the covariance
      residual with an innovation (Kalman) negative log-likelihood.
- [ ] Return a gradient for the surrogate slope α, or fix it as a hyperparameter.
- [ ] Fit the delta-encoding threshold on training statistics (not per batch).
- [ ] Rename the recursive filter as smoothing, or implement a proper state-space filter.

## 4. Early-warning indicator (Report §7.4)
- [ ] Forward-looking instability label over horizon τ (no overlap with the input window).
- [ ] Evaluate with AUROC, AUPRC, Brier score and lead time vs. a rolling-volatility baseline.

## 5. Experiment C (Report §7.5)
- [ ] Align assets on a common one-second clock before windowing.
- [ ] Report the gauge-invariant loop flux instead of individual edge phases.
- [ ] Rename "ofi" to signed trade volume (or compute true OFI from depth data).

## 6. Baselines and reporting
- [ ] Reimplement DeepLOB on the same split; equal tuning budget for all models.
- [ ] Five seeds at every horizon; bootstrap confidence intervals on macro-F1.
- [ ] Regenerate README, reference document and results files from the new runs;
      remove the misquoted DeepLOB comparison column.

## 7. Repository hygiene
- [ ] Replace `validate_*.py` (hard-coded Windows paths) with runnable unit tests.
- [ ] Live pipeline: train on target-market data; use 20-step windows and the full feature set.
