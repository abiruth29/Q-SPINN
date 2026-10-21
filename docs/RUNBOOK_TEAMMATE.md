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

**FI-2010 stock/day boundaries must be verified — the build refuses to guess.**
The public files concatenate (days × 5 stocks) segments with no id column. Setup 2 uses Train CF7 (days 1–7) and Test CF7/CF8/CF9
(days 8, 9, 10). You must supply the verified per-segment row counts (in file order) as a JSON file:

```json
{"train": [<35 row counts>], "test_d8": [<5>], "test_d9": [<5>], "test_d10": [<5>]}
```
Take the counts from the FI-2010 authors' documentation / per-stock-per-day statistics (or reconstruct them by parsing the raw
LOBSTER-style source if you have it). Each list must sum **exactly** to the number of rows in the file; the build checks count,
positivity and coverage and stops otherwise.

```bash
python procure_fi2010.py --setup2 --segments segment_lengths.json
cat data/reference/fi2010/segment_audit.json     # every partition must say provenance: authoritative
```
`--allow-heuristic` exists for exploration only (jump-based guess); the training loader **refuses** a heuristic parquet.
If you cannot obtain verified counts, stop and tell us — do not train on guessed boundaries.

Note on the split actually used: **train = days 1–6, validation = day 7, test = days 8–10** (the last training day is held
out for model selection). Describe it that way in the paper.

Binance data (Exp C): `data/curated/trades/symbol=BTCUSDT|ETHUSDT|SOLUSDT/...` parquet partitions. The pipeline aligns the three
assets on a common 1 s clock and **stops on ANY flagged outage** (a gap longer than 300 s in any asset; there is no percentage tolerance). Split the series at the gaps first and tell us; do not work around it.

## 2. Smoke run (≈ minutes) — proves the pipeline end to end

```bash
python experiments/run_unattended_suite.py --test
```
Runs 1 epoch/stage for k=10 seed 0 plus a small Binance job. Old `suite_state.json` files are ignored: completed jobs from a different `PROTOCOL_VERSION` are retrained automatically (legacy state/weights are archived under `experiments/legacy_v3.0/` and cannot be loaded). The numbers are meaningless; only crashes matter.

## 3. Full runs

```bash
python experiments/run_unattended_suite.py            # 3 horizons × 5 seeds on Setup 2, then Exp C
# or one job:
python experiments/exp_B_hybrid_fi2010/train.py --horizon 10 --seed 0
```
Output per run: `RESULTS_k{H}_s{S}.md`, weights `qspinn_fi2010_k{H}_s{S}.pt`, a `*_contract.json`, and a summary
containing macro-F1 with a 95 % block-bootstrap interval for the hybrid, SNN-only and heavy-only models, the
majority-class macro-F1, global gate rate, and two baselines (LSTM, re-implemented DeepLOB) trained with the same loop,
optimiser, early-stopping rule and an epoch budget equal to the hybrid's total (`--baseline_epochs` overrides). No model is hyper-parameter tuned — state this as a limitation. The runner resumes after a crash (state in `suite_state.json`); `--reset` starts over.

Use `--legacy_split` only to reproduce the old protocol for a comparison table; label it legacy.

## 4. After training — the analyses that carry the paper's claims

```bash
W=experiments/exp_B_hybrid_fi2010
python $W/routing_study.py       --horizon 10 --seed 0 --weights $W/qspinn_fi2010_k10_s0.pt
python $W/early_warning_study.py --horizon 10 --seed 0 --weights $W/qspinn_fi2010_k10_s0.pt
```
* `routing_study_k10_s0.json` — validation macro-F1 vs routing fraction for the learned gate, confidence cascade,
  rolling-volatility router (per-segment), benefit head and random; budget chosen on validation only with a tie-safe rule;
  test result at that fixed threshold; **matched full-pipeline latency** (raw windows → logits, eval/no_grad, same chunking,
  warm-up, device sync) for cheap-only, dense heavy and the real conditional hybrid at several routing fractions.
  Run on **both CPU and GPU** if possible (`CUDA_VISIBLE_DEVICES=` for CPU).
* `early_warning_k10_s0.json` — AUROC / AUPRC / Brier (future-window label) plus **event-based** lead time, detection rate and false
  alerts (per 1000 rows) of ω vs a rolling-vol baseline against *realised* shock onsets, per stock/day segment, on untouched test days. If ω does not beat the baseline, ω stays a "latent-flow diagnostic" in the paper.

## 5. What to send back

`segment_audit.json`, every `RESULTS_*.md` and `*_s*.json`, `suite_state.json`, `MASTER_PUBLICATION_RESULTS.md`,
the training logs, and the CPU/GPU model names. Commit them under `results/v3.1/` on a separate branch.

## 6. Reading the results honestly (rules for the write-up)

1. Report macro-F1 **with intervals** and next to the majority-class macro-F1.
2. A compute-saving claim needs the measured latency table, not the gate rate alone.
3. ω is a diagnostic. "Crash probability" / "early warning" language only if section 4's early-warning study supports it.
4. No lead-lag claim from the magnetic-Laplacian phase; at most report the gauge-invariant loop flux.
5. Live-demo outputs are labelled DEMONSTRATION unless a model was retrained on the live input contract.

## 7. Not done on purpose

* Covariance training / uncertainty (`van_loan`, `cov_head` excluded; flag `QSPINN_USE_COV=1`) — needs an observation model and innovation likelihood first.
* Drift-Jacobian / anti-collapse *ablation* against unstructured alternatives; the anchor and relative residual are in, their ablation (`QSPINN_LAMBDA_SCALE=0`, `QSPINN_RESIDUAL=all`) is yours to run if the paper makes a physics claim.
* Cross-asset lead-lag research; Experiment A is excluded (legacy pipeline).
* A live model retrained on crypto inputs: the live pipeline refuses FI-2010 weights unless `--demo`.
* Hyper-parameter tuning of any model (equal-budget, untuned comparison only).

## 8. Change → code map

| Change | Where | Tests |
|---|---|---|
| finite-check/clip **before** `opt.step()` | `qspinn_core.StepGuard` | `test_phase0_core.py` |
| covariance flag (skip Van Loan / cov_head); `QSPINN_USE_COV=1` re-enables | `QSPINN.use_cov` | `test_phase0_core.py` |
| deterministic Stage C (heavy frozen / eval / no-grad) | `qspinn_core.stage_c_mode`, `heavy_teacher_ctx` | `test_phase0_core.py` |
| calibrated train-only encoder, fixed dt | `qspinn_core.SpikeEncoder` | `test_phase0_core.py` |
| exact binary hand-over | `qspinn_core.route_exact` | `test_phase0_core.py` |
| bookkeeping (global gate rate, last window kept) | `RateMeter`, `window_starts` | `test_phase0_*.py` |
| independent S/H, single generator exp((S−iH)dt), causal residual | `qspinn_core.propagate` and Exp B/C `HamiltonianNet` | `test_phase1_generator.py` |
| FI-2010 Setup 2 + stock/day segments | `fi2010_setup2.py`, `procure_fi2010.py --setup2` | `test_phase1_setup2.py` |
| Binance common 1 s clock, train-only labels | `binance_align.py` | `test_phase1_binance_align.py` |
| budgeted routing, conditional inference, early warning | `routing.py` + two drivers | `test_phase2_*.py` |
| DeepLOB baseline, intervals, claims cleanup, live contract | `baselines.py`, `stats.py`, README, `live_contract.py` | `test_phase3_*.py` |

## Review-3 additions
- A data outage is a HARD STOP (RuntimeError from `assert_no_long_outage`): split the raw Binance series at the gap and rebuild; do not work around it.
- FI-2010: `segment_lengths.json` must be verified against the source files; record file identity (names/hashes), day order, stock order, and confirm the split is train days 1-6, val day 7, test days 8-10.
- Early-warning results now include `events_total/evaluable/censored`, `alarms`, `false_alarms`, `alarms_censored`, `exposure_rows`, and the policy (tau, refractory, renew_every). Report them as-is.
- Report the actual epochs/updates executed for baselines and the hybrid, and paired segment-aware differences vs baselines.

## Review-4 additions (remaining-changes list)
- **Crypto split / scaler (`split_plan` in Exp C)**: scaler fits only on rows read as INPUT by train windows (`scaler_end`); label thresholds use only returns a train label reads (`label_support_end`); HORIZON windows are purged before each boundary. Val/test windows may read earlier rows as causal input context; that shared input history is reported, not counted as target leakage.
- **ω supervision**: Stage B aligns ω to a TRAILING volatility flag, so ω is a current-regime / latent-flow diagnostic. Use "early warning" language only if `early_warning_study.py` (untouched forward targets, same alarm policy, vs rolling-vol baselines) supports it.
- **Gate objective**: `--gate_objective legacy|benefit` (Exp B train). Run `routing_study.py` for BOTH and compare validation macro-F1 vs routed fraction against the confidence and rolling-vol cascades (threshold chosen on validation only). Keep `legacy` if `benefit` does not help.
- **Live pipeline**: `python pipeline.py --omega_mode gated|always`. `gated` (default) = gate first; heavy path/ω only on routed ticks, otherwise ω is shown cached (with age) or unavailable and zero heavy calls are made. `always` = heavy every tick, NO compute-saving claim. `latency_ms` in the log is measured end-to-end per tick.
- Presentation targets (FLOP reduction, F1, Sharpe, phase accuracy) are design targets, not results.

## Review-5 additions
- Live ω now uses the lean `heavy_infer(x, return_omega=True)` (same logits and ω as `heavy_path`, no training losses); gated mode still makes zero heavy calls on unrouted ticks.
- The live log reports `omega_fresh_mean` / `n_fresh_readings` (mean of recent FRESH ω readings). It is NOT a time integral.
- Please also measure CPU/GPU wall-clock latency (gated vs always, several routing budgets) and record hardware + package versions with the results; `docs/TEST_LOG_latest.txt` shows the format.
- Presentations no longer state tick-proportion numbers (99%/1%) as facts; re-add only a number measured on the corrected dataset, with its source.

## Review-6 additions
- **Release test run**: `bash scripts/release_check.sh` (installs requirements, runs pytest with `QSPINN_REQUIRE_FULL=1`, so ANY skip is a failure, and writes `docs/TEST_LOG_latest.txt` with package versions and hardware). `requirements-lock.txt` lists the versions the supplied log was produced with. Send me the generated log from YOUR machine.
- **Live input freshness**: a window step/inference now happens only after a NEW BTC depth update (`DepthFreshness`); trade-only ticks and feed gaps never re-push stale features, and every log record carries `pred`, `input_age_s`, `omega_status`, `heavy_calls`, `latency_ms`. The input contract has a `cadence` key (`new_depth_throttled_2hz` live vs `event_rows` for FI-2010); a model is production-eligible only if it was trained on the same cadence, otherwise the pipeline stays DEMONSTRATION-only.
- **Reference document**: `Q_SPINN_COMPLETE_REFERENCE_DOCUMENT.md` section 3 is now labelled LEGACY / INVALID; the unmeasured DeepLOB comparison and the "93% compute reduction" claim were removed.
- For `segment_audit.json` (FI-2010): record per-stock/day counts, file hashes and ordering evidence; every segment must be validated against the source records.

## Review-7 additions (from the first Windows run)
- **Segment manifest.** `segment_lengths.json` is NOT in the repo and is NOT published in the FI-2010 paper (arXiv 1705.03233 gives totals only: ~4 M events, 394,337 samples after sampling). Check the dataset README linked from the paper (Etsin/Fairdata page) or ask the authors for per-stock/day counts. Format: `{"train": [42 counts], "test_d8": [5], "test_d9": [5], "test_d10": [5]}`; each list must sum to the row count of its source file. Until a verified manifest exists, Setup 2 training is refused by design. Do NOT type in guessed numbers.
- **Pipeline smoke only (not citable):** `python procure_fi2010.py` (legacy parquet), then `python experiments/exp_B_hybrid_fi2010/train.py --horizon 10 --epochs_a 1 --epochs_b 1 --epochs_c 1 --legacy_split --run_name smoke`. Results from `--legacy_split` or `--allow-heuristic` are never evidence.
- **Binance data (Exp C).** If `data/curated/trades/symbol=BTCUSDT|ETHUSDT|SOLUSDT` is missing, Exp C now stops with a clear message; build it with `python procure_binance_data.py` first.
- **Suite runner.** `run_unattended_suite.py` now lists failed jobs and exits with code 1 when any job failed. It only says "ALL … SUCCEEDED" when none failed.
- **Windows.** All text reads/writes are explicitly UTF-8 (a test enforces this). GPU latency needs a CUDA machine: a CPU-only torch (e.g. 2.7.1+cpu) can only produce CPU numbers.
