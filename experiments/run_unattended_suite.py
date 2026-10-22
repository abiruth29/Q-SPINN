"""
experiments/run_unattended_suite.py
===================================
Master Unattended Experimental Test & Training Suite for Q-SPINN.

Designed for long-running, uninterrupted execution (8-9 hours) without requiring
human interaction or approval.

JOB QUEUE:
---------
1. [FI-2010] k=10, Seed 0  (Calibrated EWS & Sparse Gate)
2. [FI-2010] k=20, Seed 0  (Horizon Sweep)
3. [FI-2010] k=50, Seed 0  (Horizon Sweep)
4. [FI-2010] k=10, Seed 1  (Multi-Seed Validation)
5. [FI-2010] k=10, Seed 2  (Multi-Seed Validation)
6. [FI-2010] k=10, Seed 3  (Multi-Seed Validation)
7. [FI-2010] k=10, Seed 4  (Multi-Seed Validation)
8. [Binance] 3-Asset Basket (BTC, ETH, SOL with Magnetic Graph Laplacian)
9. Compilation of Master Publication Table (experiments/MASTER_PUBLICATION_RESULTS.md)

FEATURES:
--------
- Zero human intervention: fully automated queue.
- Resilient checkpointing: suite_state.json tracks completion; skips completed stages on resume.
- Crash isolation: exceptions in any job are logged and trapped so subsequent jobs proceed.
- Auto-compilation of publication-ready markdown tables with mean ± std error bars.
- Smoke test mode (--test) for rapid verification (1 epoch per stage).
"""

import sys
import gc
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
sys.path.insert(0, str(EXP_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STATE_FILE = EXP_DIR / "suite_state.json"
LOG_FILE   = EXP_DIR / "suite_run.log"
MASTER_MD  = EXP_DIR / "MASTER_PUBLICATION_RESULTS.md"


def set_suite_files(test_mode: bool = False):
    global STATE_FILE, LOG_FILE, MASTER_MD
    if test_mode:
        STATE_FILE = EXP_DIR / "suite_state_test.json"
        LOG_FILE   = EXP_DIR / "suite_test.log"
        MASTER_MD  = EXP_DIR / "TEST_PUBLICATION_RESULTS.md"
    else:
        STATE_FILE = EXP_DIR / "suite_state.json"
        LOG_FILE   = EXP_DIR / "suite_run.log"
        MASTER_MD  = EXP_DIR / "MASTER_PUBLICATION_RESULTS.md"


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    try:
        print(formatted, flush=True)
    except Exception:
        try:
            print(formatted.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"Warning reading state file: {e}. Starting fresh.")
    return {"jobs": {}, "started_at": datetime.now().isoformat(), "completed_at": None}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def generate_master_report(state: dict):
    from qspinn_core import PROTOCOL_VERSION
    # only results produced under the CURRENT protocol may appear in a corrected report
    jobs = {k: v for k, v in state.get("jobs", {}).items()
            if v.get("status") != "COMPLETED" or v.get("protocol") == PROTOCOL_VERSION}
    
    # 1. Horizon Benchmark Table
    horizon_rows = ""
    for k in [10, 20, 50]:
        ms = [jobs[f"fi2010_k{k}_s{sd}"]["metrics"] for sd in range(5)
              if jobs.get(f"fi2010_k{k}_s{sd}", {}).get("status") == "COMPLETED" and "metrics" in jobs[f"fi2010_k{k}_s{sd}"]]
        if ms:
            ms_ = lambda key, f="{:.3f}": f"{f.format(np.mean([m[key] for m in ms]))} ± {f.format(np.std([m[key] for m in ms], ddof=1) if len(ms) > 1 else 0.0)}"
            horizon_rows += (f"| k={k} (n={len(ms)}) | {ms_('accuracy')} | {ms_('macro_f1')} | {ms_('gate_rate')} | "
                             f"{ms_('omega', '{:.4f}')} | {np.mean([m['majority'] for m in ms]):.3f} | "
                             f"{np.mean([m.get('majority_macro_f1', float('nan')) for m in ms]):.3f} |\n")
        else:
            horizon_rows += f"| k={k} | Pending | — | — | — | — | — |\n"

    seed_f1s = []
    seed_accs = []
    seed_gates = []
    seed_omegas = []
    seed_crashes = []
    seed_rows = ""
    for s in range(5):
        job_key = f"fi2010_k10_s{s}"
        j = jobs.get(job_key, {})
        if j.get("status") == "COMPLETED" and "metrics" in j:
            m = j["metrics"]
            seed_f1s.append(m["macro_f1"])
            seed_accs.append(m["accuracy"])
            seed_gates.append(m["gate_rate"])
            seed_omegas.append(m["omega"])
            seed_crashes.append(m["crash_p"])
            seed_rows += f"| Seed {s} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['gate_rate']:.3f} | {m['omega']:.4f} | {m['crash_p']*100:.1f}% |\n"
        else:
            seed_rows += f"| Seed {s} | Pending | — | — | — | — |\n"

    stats_row = ""
    if len(seed_f1s) > 1:
        acc_mean, acc_std = np.mean(seed_accs), np.std(seed_accs)
        f1_mean, f1_std   = np.mean(seed_f1s), np.std(seed_f1s)
        gate_mean, gate_std = np.mean(seed_gates), np.std(seed_gates)
        omega_mean, omega_std = np.mean(seed_omegas), np.std(seed_omegas)
        crash_mean = np.mean(seed_crashes) * 100
        stats_row = f"| **Mean ± Std (N={len(seed_f1s)})** | **{acc_mean:.3f} ± {acc_std:.3f}** | **{f1_mean:.3f} ± {f1_std:.3f}** | **{gate_mean:.3f} ± {gate_std:.3f}** | **{omega_mean:.4f} ± {omega_std:.4f}** | **{crash_mean:.1f}%** |\n"

    # 3. Multi-Asset Binance
    binance_job = jobs.get("binance_3asset", {})
    binance_row = ""
    binance_lead_lag = ""
    if binance_job.get("status") == "COMPLETED" and "metrics" in binance_job:
        bm = binance_job["metrics"]
        binance_row = f"| **Q-SPINN Multi-Asset** | {bm['acc']:.3f} | {bm['f1']:.3f} | {bm['gate']:.3f} | ✅ Trained |\n"
        binance_lead_lag = f"""
### Phase-matrix diagnostic (no lead-lag claim)
- **BTC-ETH-SOL loop flux (gauge-invariant):** `{bm.get('loop_flux', float('nan')):+.4f}` rad
- Individual edge phases are gauge-dependent and are not reported. A lead-lag statement requires an out-of-sample predictive
  increment over directional temporal baselines, which this run does not test.
"""
    else:
        binance_row = "| **Q-SPINN Multi-Asset** | Pending | — | — | — |\n"

    now = datetime.now().isoformat()
    hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    md = f"""# Q-SPINN — Master Publication Experimental Results

**Last Updated:** {now}  
**Hardware:** {hw}  
**Execution Mode:** Unattended Automated Experimental Suite  

---

## 1. Multi-Horizon Benchmarking on FI-2010 (Section 25)

Benchmark results across standard FI-2010 prediction horizons (Setup 2) ($k=10, 20, 50$ ticks).

| Horizon (mean ± std over seeds) | Accuracy | Macro-F1 | Gate Rate | $\\omega(A_t)$ (latent-flow diagnostic) | Majority acc | Majority macro-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{horizon_rows}

*Baselines (LSTM, re-implemented DeepLOB) are trained on the same Setup 2 split and epoch budget inside each run; see the per-run RESULTS_k*_s*.md. Literature numbers are not comparable (different split) and are not quoted.*

---

## 2. Multi-Seed Statistical Validation (SWAP-12, k=10)

Statistical significance across 5 random seeds (Seeds 0, 1, 2, 3, 4):

| Run | Accuracy | Macro-F1 | Gate Rate | $\\omega(A_t)$ | ω>0 rate (unvalidated) |
|:---|:---:|:---:|:---:|:---:|:---:|
{seed_rows}{stats_row}

---

## 3. Multi-Asset Crypto Basket (Experiment C — BTC, ETH, SOL)

Validates cross-asset Magnetic Graph Laplacian coupling on Binance microstructure data.

| Architecture | Accuracy | Macro-F1 | Gate Rate | Status |
|:---|:---:|:---:|:---:|:---:|
{binance_row}
{binance_lead_lag}
---

## 4. How to read these tables

1. The gate rate is a routing-budget statistic, not a compute saving; use the measured latency in `routing_study_k*_s*.json`.
2. $\\omega(A_t)$ is a latent-flow diagnostic; whether it forecasts instability is decided only by `early_warning_k*_s*.json`.
3. $\\|H-H^\\dagger\\|=0$ because $H$ is constructed Hermitian; it is a code check, not a result.
4. Every number here was produced under protocol `{PROTOCOL_VERSION}`; results from other protocols are ignored.
"""
    MASTER_MD.write_text(md, encoding="utf-8")
    log(f"Master publication report updated → {MASTER_MD}")


SMOKE_DIR = Path(__file__).resolve().parent / "_smoke"   # smoke runs never touch the real result folders
SKIPPED: set = set()        # jobs found COMPLETED with an identical signature (not re-run)
ATTEMPTED: set = set()      # job ids this invocation actually ran or resumed (older failures are reported separately)


def _job_signature(job_id: str, kwargs: dict) -> str:
    """Identity of a job run: its configuration (epochs, ticks, smoke/full via out_dir presence, ...)."""
    from checkpoint import fingerprint_of
    return fingerprint_of(job=job_id, smoke=("out_dir" in kwargs),
                          cfg={k: v for k, v in kwargs.items() if k != "out_dir"})


def run_job(job_id: str, desc: str, fn, *args, **kwargs):
    ATTEMPTED.add(job_id)
    state = load_state()
    from qspinn_core import PROTOCOL_VERSION
    sig = _job_signature(job_id, kwargs)
    old = state["jobs"].get(job_id, {})
    if old.get("status") == "COMPLETED" and old.get("protocol") == PROTOCOL_VERSION and old.get("signature") != sig:
        log(f"♻️  [STALE-CONFIG] {job_id}: earlier COMPLETED record has a different (or no) run signature "
            "(e.g. a smoke run or an older configuration) -> ignoring it and re-running.")
        state["jobs"].pop(job_id); save_state(state)
    if old.get("status") == "COMPLETED" and old.get("protocol") != PROTOCOL_VERSION:
        log(f"♻️  [STALE] {job_id}: completed under protocol {old.get('protocol')!r}, current is "
            f"{PROTOCOL_VERSION!r} -> ignoring legacy result and retraining.")
        state["jobs"].pop(job_id)
        save_state(state)
    if job_id in state["jobs"] and state["jobs"][job_id].get("status") == "COMPLETED":
        SKIPPED.add(job_id)
        log(f"⏩ [SKIP] {desc} (already COMPLETED with an identical configuration; use --reset to force a re-run)")
        return state["jobs"][job_id].get("metrics")

    log("=" * 70)
    log(f"🚀 [STARTING] {desc} (Job ID: {job_id})")
    log("=" * 70)
    
    state["jobs"][job_id] = {"status": "RUNNING", "started_at": datetime.now().isoformat()}
    save_state(state)
    t0 = time.time()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        metrics = fn(*args, **kwargs)
        elapsed = time.time() - t0

        state = load_state()
        state["jobs"][job_id] = {
            "status": "COMPLETED",
            "protocol": PROTOCOL_VERSION,
            "signature": sig,
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "metrics": metrics
        }
        save_state(state)
        log(f"✅ [SUCCESS] {desc} completed in {elapsed/60:.1f} minutes.")
        generate_master_report(state)
        return metrics

    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        log(f"❌ [FAILED] {desc} encountered error after {elapsed/60:.1f} minutes:")
        log(err_msg)

        state = load_state()
        state["jobs"][job_id] = {
            "status": "FAILED",
            "failed_at": datetime.now().isoformat(),
            "error": err_msg
        }
        save_state(state)
        generate_master_report(state)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q-SPINN Master Unattended Suite")
    parser.add_argument("--test", action="store_true", help="Smoke test mode: run 1 epoch per stage to test entire flow")
    parser.add_argument("--reset", action="store_true", help="Reset state file and start fresh from job 1")
    parser.add_argument("--skip-binance", action="store_true", help="Skip Binance multi-asset and run only FI-2010 suite")
    parser.add_argument("--only-binance", action="store_true", help="Run only Binance multi-asset experiment")
    parser.add_argument("--gate_objective", choices=["legacy", "benefit"], default="legacy",
                        help="Gate training objective for Exp B and Exp C (non-default runs get their own job id / output dir)")
    parser.add_argument("--seed", type=int, default=None, help="Exp C seed (default 0); run >=5 seeds before reporting loop flux")
    parser.add_argument("--price_mode", choices=["mean", "last"], default="mean",
                        help="Exp C price series: per-second mean trade price (legacy) or last trade of the second")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override Exp C epochs per stage (default 50; validation F1 plateaus by ~15-20)")
    args = parser.parse_args()
    gsuf = "" if args.gate_objective == "legacy" else f"_{args.gate_objective}"
    gkw = {} if args.gate_objective == "legacy" else {"gate_objective": args.gate_objective}

    set_suite_files(args.test)

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log(f"🧹 Reset state file: {STATE_FILE}")

    mode_str = "SMOKE TEST MODE (1 epoch per stage)" if args.test else "FULL PRODUCTION TRAINING MODE (~7-8 hours)"
    log("*" * 70)
    log(f"  Q-SPINN UNATTENDED SUITE RUNNER")
    log(f"  Mode: {mode_str}")
    log(f"  Start: {datetime.now().isoformat()}")
    log("*" * 70)

    # Epoch settings based on mode
    ep_a = 1 if args.test else 100
    ep_b = 1 if args.test else 100
    ep_c = 1 if args.test else 100
    binance_ep = 1 if args.test else (args.epochs or 50)
    binance_ticks = 25_000 if args.test else 350_000

    from exp_B_hybrid_fi2010.train import main as train_fi2010

    if not args.only_binance:
        # Task 15: every horizon gets 5 seeds (smoke test: one job only). Setup 2 data, same budget for all models.
        horizons = (10,) if args.test else (10, 20, 50)
        seeds    = (0,)  if args.test else range(5)
        for k in horizons:
            for sd in seeds:
                run_job(f"fi2010_k{k}_s{sd}{gsuf}", f"FI-2010 Setup 2, k={k}, seed {sd}, gate={args.gate_objective}",
                        train_fi2010, horizon=k, seed=sd, epochs_a=ep_a, epochs_b=ep_b, epochs_c=ep_c, **gkw,
                        **({"out_dir": SMOKE_DIR / f"exp_B{gsuf}"} if args.test else
                           ({"out_dir": Path(__file__).resolve().parent / "runs" / f"exp_B{gsuf}_k{k}_s{sd}"} if gsuf else {})))

    if not args.skip_binance:
        # ── 5. Binance 3-Asset Basket (BTC, ETH, SOL with Magnetic Laplacian)
        from exp_C_multi_asset_binance.train import main as train_binance
        csuf = (gsuf + (f"_{args.price_mode}" if args.price_mode != "mean" else "")
                + (f"_s{args.seed}" if args.seed is not None else "")
                + (f"_e{args.epochs}" if args.epochs and not args.test else ""))
        ckw = dict(gkw, **({"seed": args.seed} if args.seed is not None else {}),
                   **({"price_mode": args.price_mode} if args.price_mode != "mean" else {}))
        run_job(
            f"binance_3asset{csuf}",
            f"Binance 3-Asset Hybrid with Magnetic Graph Laplacian (BTC, ETH, SOL), gate={args.gate_objective}",
            train_binance,
            epochs_a=binance_ep, epochs_b=binance_ep, epochs_c=binance_ep,
            max_ticks=binance_ticks, **ckw,
            **({"out_dir": SMOKE_DIR / f"exp_C{csuf}"} if args.test else
               ({"out_dir": Path(__file__).resolve().parent / "runs" / f"exp_C{csuf}"} if csuf else {}))
        )

    # ── 9. Final Master Report Generation ──────────────────────────────────
    state = load_state()
    jobs = state.get("jobs", {})
    failed = sorted(j for j, v in jobs.items() if v.get("status") == "FAILED" and j in ATTEMPTED)
    ok = sorted(j for j, v in jobs.items() if v.get("status") not in (None, "FAILED") and j in ATTEMPTED
                and j not in SKIPPED)
    skipped = sorted(j for j in SKIPPED if j in ATTEMPTED)
    stale_failed = sorted(j for j, v in jobs.items() if v.get("status") == "FAILED" and j not in ATTEMPTED)
    state["finished_at"] = datetime.now().isoformat()
    state["n_failed"] = len(failed)
    if not failed:
        state["completed_at"] = state["finished_at"]
    save_state(state)
    generate_master_report(state)

    log("*" * 70)
    if failed:
        log(f"  ❌ SUITE FINISHED WITH {len(failed)} FAILED JOB(S): {', '.join(failed)}")
        log(f"     Succeeded: {', '.join(ok) if ok else 'none'}")
        log("     Read the error above/in the state file, fix the cause, then re-run (finished jobs are skipped).")
    else:
        log(f"  ✅ {len(ok)} JOB(S) RAN AND SUCCEEDED" + (f"; {len(skipped)} SKIPPED (already completed: {', '.join(skipped)})" if skipped else ""))
    if stale_failed:
        log(f"  ℹ️  Earlier FAILED job(s) NOT run this time: {', '.join(stale_failed)}")
    log(f"  Master Report Available at: {MASTER_MD}")
    log("*" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
