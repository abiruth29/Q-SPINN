# Q-SPINN — Quantum-Inspired Spiking Physics-Informed Neural Network

**Research prototype** for compute-aware mid-price direction prediction on limit-order-book data:
a cheap event-driven spiking (LIF) gate decides, per window, whether an expensive quantum-inspired
"heavy" expert (generator A = S − iH, propagated with exp(AΔt)) is needed.

> **Status (v3.1, branch `fix/v3.1-corrections`).** The code has been corrected after an independent audit
> (see `docs/FIX_PLAN.md`). **All numbers from earlier versions are retracted** and live under
> `experiments/**/RESULTS*.md` with a LEGACY banner. New results are produced by the runbook in
> `docs/RUNBOOK_TEAMMATE.md`; until they exist, this repository makes **no** accuracy, compute-saving,
> early-warning or lead-lag claim.

## What the model is (and is not)

| Component | What it does | What is *not* claimed |
|---|---|---|
| SNN gate (`SNNGate`) | cheap expert + routing decision from a delta-encoded spike train (train-only thresholds) | not shown to predict where the heavy expert helps — measured in the routing study |
| Heavy expert (`heavy_path`) | complex amplitude ψ, one generator A = S − iH, exp(AΔt), density read-out | "quantum" means quantum-*inspired* linear algebra, not quantum computing |
| Routing | binary, exact hand-over (g=1 → heavy logits, g=0 → cheap logits); budget chosen on validation | no compute saving is claimed until wall-clock latency is measured (`routing_study.py`) |
| ω = λ_max(S) | latent-flow **diagnostic** (numerical abscissa of the symmetric part) | **not** a crash probability or validated early-warning signal |
| Magnetic Laplacian (Exp C) | learned antisymmetric phase matrix; only the gauge-invariant loop flux is reported | **no** causal lead-lag claim between assets |

## Data protocols

* **FI-2010 Setup 2** — train on days 1–7, test on days 8–10, built by `python procure_fi2010.py --setup2`
  (stock/day segments recovered and audited in `segment_audit.json`; windows never cross a segment; the last
  `k` rows of each segment are purged). The old concatenated-fold protocol is available only as `--legacy_split`.
* **Binance (Exp C)** — BTC/ETH/SOL trades aggregated to a *common* 1-second clock with stale-second flags;
  label thresholds are fit on the training span only; `ofi` was renamed `signed_volume` (it is trade-signed
  volume, not order-flow imbalance from depth).

## Repository map

| Path | Purpose |
|---|---|
| `experiments/qspinn_core.py` | shared, unit-tested core (safe optimiser step, spike encoder, exact routing, generator, windowing) |
| `experiments/exp_B_hybrid_fi2010/train.py` | FI-2010 training (three stages A/B/C); `routing_study.py`, `early_warning_study.py` run after it |
| `experiments/exp_C_multi_asset_binance/train.py` | 3-asset crypto experiment |
| `experiments/routing.py` | budgeted routing, conditional inference, early-warning metrics |
| `experiments/live_contract.py`, `pipeline.py`, `dashboard.py` | live demo; refuses mismatched weights unless `--demo` |
| `tests/` | synthetic-data unit tests for every correction (`pytest -q`, ~10 s, no data or GPU needed) |
| `docs/FIX_PLAN.md`, `docs/RUNBOOK_TEAMMATE.md` | what changed, and exactly what to run to regenerate results |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # must be green before any training run
python procure_fi2010.py --setup2           # inspect data/reference/fi2010/segment_audit.json
python experiments/exp_B_hybrid_fi2010/train.py --horizon 10 --seed 0
python experiments/exp_B_hybrid_fi2010/routing_study.py --horizon 10 --seed 0 --weights <weights.pt>
```

The long-form mathematical reference is `Q_SPINN_COMPLETE_REFERENCE_DOCUMENT.md`; it predates v3.1 and its
result tables are superseded.
