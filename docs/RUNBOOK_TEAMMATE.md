# Runbook — regenerating Q-SPINN results under the v3.1 protocol

For the teammate who runs training. All code changes are on branch `fix/v3.1-corrections` (PR #1). Nothing in this
repository has been trained or evaluated under the corrected code: **every number must come from these runs.**

## 0. Setup (once)

```bash
git checkout fix/v3.1-corrections && git pull
pip install -r requirements.txt            # torch, pyarrow, pandas, pytest ...
pytest -q                                  # expect: all green (~10 s, CPU only, no data needed)
```
If any test fails, stop and report it — do not train.

## 1. Data

```bash
python procure_fi2010.py --setup2          # builds data/reference/fi2010/fi2010_setup2.parquet
cat data/reference/fi2010/segment_audit.json
```
**Inspect the audit.** Each partition lists `boundary_confidence` and `confident`. The train partition must have 35
segments (7 days × 5 stocks), each test day 5. If any partition is `confident: false` (confidence < 1.5), the
automatic stock/day boundary recovery is unreliable: take the per-stock/day row counts from the FI-2010 authors'
documentation and put them in `experiments/fi2010_setup2.py::detect_segments` before training. Send us the audit file.

Binance data (Exp C) is unchanged: `data/` must contain the curated BTCUSDT/ETHUSDT/SOLUSDT parquet partitions.

## 2. Smoke run (≈ minutes) — proves the pipeline end to end

```bash
python experiments/run_unattended_suite.py --test
```
Runs 1 epoch/stage for k=10 seed 0 plus a small Binance job. The numbers are meaningless; only crashes matter.

## 3. Full runs

```bash
python experiments/run_unattended_suite.py            # 3 horizons × 5 seeds on Setup 2, then Exp C
# or one job:
python experiments/exp_B_hybrid_fi2010/train.py --horizon 10 --seed 0
```
Output per run: `RESULTS_k{H}_s{S}.md`, weights `qspinn_fi2010_k{H}_s{S}.pt`, a `*_contract.json`, and a summary
containing macro-F1 with a 95 % block-bootstrap interval for the hybrid, SNN-only and heavy-only models, the
majority-class macro-F1, global gate rate, and two baselines trained with the same budget (LSTM, re-implemented
DeepLOB). The runner resumes after a crash (state in `suite_state.json`); `--reset` starts over.

Use `--legacy_split` only to reproduce the old protocol for a comparison table; label it legacy.

## 4. After training — the analyses that carry the paper's claims

```bash
W=experiments/exp_B_hybrid_fi2010
python $W/routing_study.py       --horizon 10 --seed 0 --weights $W/qspinn_fi2010_k10_s0.pt
python $W/early_warning_study.py --horizon 10 --seed 0 --weights $W/qspinn_fi2010_k10_s0.pt
```
* `routing_study_k10_s0.json` — validation macro-F1 vs routing fraction for the learned gate, confidence cascade,
  rolling-volatility router, benefit head and random; budget chosen on validation only; test result at that fixed
  threshold; CPU/GPU latency of conditional vs dense heavy inference (warm-up + synchronisation included).
  Run on **both CPU and GPU** if possible (`CUDA_VISIBLE_DEVICES=` for CPU).
* `early_warning_k10_s0.json` — AUROC / AUPRC / Brier / false-alarm rate / lead time of ω vs a rolling-vol
  baseline on untouched test days. If ω does not beat the baseline, ω stays a "latent-flow diagnostic" in the paper.

## 5. What to send back

`segment_audit.json`, every `RESULTS_*.md` and `*_s*.json`, `suite_state.json`, `MASTER_PUBLICATION_RESULTS.md`,
the training logs, and the CPU/GPU model names. Commit them under `results/v3.1/` on a separate branch.

## 6. Reading the results honestly (rules for the write-up)

1. Report macro-F1 **with intervals** and next to the majority-class macro-F1.
2. A compute-saving claim needs the measured latency table, not the gate rate alone.
3. ω is a diagnostic. "Crash probability" / "early warning" language only if section 4's early-warning study supports it.
4. No lead-lag claim from the magnetic-Laplacian phase; at most report the gauge-invariant loop flux.
5. Live-demo outputs are labelled DEMONSTRATION unless a model was retrained on the live input contract.

## 7. Not done on purpose (Phase 4 — needs a defensible data model first)

* Item 18: covariance training / uncertainty (`van_loan`, `cov_head` are excluded from optimisation; flag `USE_COVARIANCE`).
* Item 19: drift-Jacobian / anti-collapse ablation.
* Item 20: cross-asset lead-lag research on aligned series (the aligned data pipeline is ready; the study is not written).

## 8. Item → code map

| # | Change | Where | Tests |
|---|---|---|---|
| 01 | finite-check/clip **before** `opt.step()` | `qspinn_core.safe_step` | `test_phase0_core.py` |
| 02 | covariance flag (skip Van Loan/cov_head) | `QSPINN.use_cov` | `test_phase0_core.py` |
| 03 | deterministic Stage C (heavy frozen/eval/no-grad) | `qspinn_core.freeze_for_gate_training` | `test_phase0_core.py` |
| 04 | calibrated encoder, fixed dt | `qspinn_core.SpikeEncoder` | `test_phase0_core.py` |
| 05 | exact binary hand-over | `qspinn_core.route_exact` | `test_phase0_core.py` |
| 06 | bookkeeping (global rate, last window) | `RateMeter`, `window_starts` | `test_phase0_*.py` |
| 07 | FI-2010 Setup 2 + segments | `fi2010_setup2.py` | `test_phase1_setup2.py` |
| 08 | independent S/H, single generator, causal residual | `qspinn_core` generator | `test_phase1_generator.py` |
| 09 | Binance common 1 s clock | `binance_align.py` | `test_phase1_binance_align.py` |
| 10–12 | budgeted routing, conditional inference, early warning | `routing.py` + two drivers | `test_phase2_*.py` |
| 13–15 | controlled baselines, intervals, claims cleanup, live contract | `baselines.py`, `stats.py`, README, `live_contract.py` | `test_phase3_*.py` |
