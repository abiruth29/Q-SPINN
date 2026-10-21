"""
*** EXCLUDED FROM v3.1 CORRECTED COMPARISONS ***
Experiment A still uses the pre-v3.1 pipeline (independent per-asset truncation instead of a common clock,
label thresholds fitted on the full series, old encoder/surrogate). Its results are LEGACY and must not appear
next to Exp B/C numbers. `main()` refuses to run unless QSPINN_ALLOW_LEGACY_EXPA=1 is set explicitly.

experiments/exp_A_snn_binance/train.py
=======================================
experiments/exp_A_snn_binance/train.py
=======================================
Experiment A — SNN Gate trained on Binance trade-derived features.

WHAT THIS DOES
--------------
1. Samples ~500K 1-second aggregation windows from the 36 Binance
   trade Parquet files (3 assets × 12 months).
2. Engineers 6 features per asset per window:
     log_return, signed_volume, realized_vol, buy_pressure, trade_count, vwap_spread
3. Constructs a regime mask proxy: windows where realized_vol > μ + 2σ
   (rolling 200-tick window). See Q-SPINN_Demo_Architecture.md §8.6.
4. Trains the SNN gate (LIF stack) with surrogate gradients.
5. Saves weights + auto-generates RESULTS.md.

SWAP STATUS
-----------
SWAP-01: ✅ Real Binance trade data
SWAP-02: ✅ Scaler fitted on train window only
SWAP-03: Delta encoding, fixed threshold (unchanged from toy)
SWAP-04: arctan surrogate (unchanged from toy — snntorch deferred to Phase 2)
SWAP-05: Hand-written LIF (unchanged from toy — snntorch deferred to Phase 2)
SWAP-15: ✅ Realized-volatility 2σ proxy regime mask
"""

import sys
import time
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "curated" / "trades"
OUT_DIR    = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))

from device import get_device
from logger import ExperimentLogger
from checkpoint import save_checkpoint, load_checkpoint, guard_nan

# ── hyper-parameters ─────────────────────────────────────────────────────────
N_ASSETS       = 3
FEATS_PER_ASSET = 6          # log_return, signed_volume, realized_vol, buy_pressure, trade_count, vwap_spread
N_FEATURES     = N_ASSETS * FEATS_PER_ASSET   # 18
SEQ_LEN        = 20          # lookback window (ticks = 1-second bins)
HORIZON        = 5           # prediction horizon
ALPHA          = 0.0015      # dead-zone for STATIONARY label
HIDDEN         = 64
LAMBDA_RATE    = 3e-3        # sparsity regularizer weight
LAMBDA_REGIME  = 1.0         # regime supervision weight
REGIME_SIGMA   = 2.0         # threshold: μ + k*σ
REGIME_WINDOW  = 200         # rolling window for vol stats
SAMPLE_TICKS   = 500_000     # ticks to sample per asset (cap for RAM)
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15        # remaining 15% = test
BATCH_SIZE     = 256
EPOCHS         = 25
LR             = 2e-3
SEED           = 0


# ── surrogate gradient ────────────────────────────────────────────────────────
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        return grad_out / (1.0 + (math.pi * v) ** 2)

spike_fn = SurrogateSpike.apply


# ── LIF layer ─────────────────────────────────────────────────────────────────
class LIFLayer(nn.Module):
    def __init__(self, n_in, n_out, beta=0.9, theta=1.0):
        super().__init__()
        self.fc       = nn.Linear(n_in, n_out)
        self.beta_raw = nn.Parameter(torch.logit(torch.full((n_out,), beta)))
        self.theta    = theta

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        beta = torch.sigmoid(self.beta_raw)
        U = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        S = torch.zeros_like(U)
        out = []
        for t in range(T):
            I  = self.fc(x_seq[:, t])
            U  = beta * U + I - self.theta * S
            S  = spike_fn(U - self.theta)
            out.append(S)
        return torch.stack(out, dim=1), U


class LeakyReadout(nn.Module):
    def __init__(self, n_in, n_out, beta=0.9):
        super().__init__()
        self.fc   = nn.Linear(n_in, n_out)
        self.beta = beta

    def forward(self, s_seq):
        U = torch.zeros(s_seq.shape[0], self.fc.out_features, device=s_seq.device)
        for t in range(s_seq.shape[1]):
            U = self.beta * U + self.fc(s_seq[:, t])
        return U


class ToySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lif1    = LIFLayer(2 * N_FEATURES, HIDDEN)
        self.lif2    = LIFLayer(HIDDEN, HIDDEN)
        self.readout = LeakyReadout(HIDDEN, 3)

    def forward(self, spikes):
        s1, _ = self.lif1(spikes)
        s2, _ = self.lif2(s1)
        return self.readout(s2), 0.5 * (s1.mean() + s2.mean())


# ── data engineering ──────────────────────────────────────────────────────────
_BATCH   = 2_000_000   # raw rows per streaming batch (~120-200 MB RAM)
_AGG_OPS = {
    "mid_sum": "sum", "mid_count": "sum", "signed_volume": "sum",
    "buy_vol": "sum", "sell_vol":  "sum", "trade_count": "sum",
    "total_pq": "sum", "total_qty": "sum",
}


def _batch_to_seconds(batch):
    """
    Convert one raw-trade batch (PyArrow RecordBatch) to per-second aggregates.
    Fully vectorised — no apply(), no lambda. Handles ~2M rows in < 1 second.
    """
    import pandas as pd
    df = batch.to_pandas()
    df["second_bin"] = (df["timestamp_us"] // 1_000_000).astype("int64")
    df["signed_qty"] = df["qty"].where(df["is_buyer_maker"], -df["qty"])
    df["qty_buy"]    = df["qty"].where(df["is_buyer_maker"],  0.0)
    df["qty_sell"]   = df["qty"].where(~df["is_buyer_maker"], 0.0)
    df["price_qty"]  = df["price"] * df["qty"]
    g = df.groupby("second_bin", sort=True)
    return pd.DataFrame({
        "mid_sum"    : g["price"].sum(),
        "mid_count"  : g["price"].count(),
        "signed_volume"        : g["signed_qty"].sum(),
        "buy_vol"    : g["qty_buy"].sum(),
        "sell_vol"   : g["qty_sell"].sum(),
        "trade_count": g["qty"].count(),
        "total_pq"   : g["price_qty"].sum(),
        "total_qty"  : g["qty"].sum(),
    })


def engineer_features_for_symbol(symbol_dir: Path, max_ticks: int) -> np.ndarray | None:
    """
    Stream Parquet files in 2M-row batches, aggregate each batch to 1-second
    bins on the fly, then merge the micro-aggregations.

    Peak RAM: ~500 MB regardless of symbol (BTC has ~1.15B raw rows).
    """
    import pandas as pd
    files = sorted(symbol_dir.rglob("*.parquet"))
    if not files:
        return None

    all_second_dfs = []
    ticks_so_far   = 0

    for f in files:
        pf          = pq.ParquetFile(f)
        batch_parts = []
        for batch in pf.iter_batches(
            batch_size=_BATCH,
            columns=["price", "qty", "timestamp_us", "is_buyer_maker"],
        ):
            batch_parts.append(_batch_to_seconds(batch))

        if not batch_parts:
            continue

        # Re-aggregate at second-bin level to merge cross-batch boundaries
        file_sec = (
            pd.concat(batch_parts)
            .groupby(level=0)
            .agg(_AGG_OPS)
        )
        all_second_dfs.append(file_sec)
        ticks_so_far += len(file_sec)
        if ticks_so_far >= max_ticks:
            break

    if not all_second_dfs:
        return None

    # Merge across files (second-bin boundaries can span files)
    combined = (
        pd.concat(all_second_dfs)
        .groupby(level=0)
        .agg(_AGG_OPS)
        .sort_index()
        .reset_index(drop=True)
    )

    combined["mid_price"]    = combined["mid_sum"]  / combined["mid_count"].clip(lower=1)
    combined["vwap"]         = combined["total_pq"] / combined["total_qty"].clip(lower=1e-8)
    combined["log_return"]   = np.log(combined["mid_price"] / combined["mid_price"].shift(1).bfill())
    combined["realized_vol"] = combined["log_return"].rolling(20, min_periods=1).std().fillna(0)
    combined["buy_pressure"] = combined["buy_vol"] / (combined["buy_vol"] + combined["sell_vol"]).clip(lower=1e-8)
    combined["vwap_spread"]  = (combined["vwap"] - combined["mid_price"]).abs() / combined["mid_price"].clip(lower=1e-8)

    feats = combined[
        ["log_return", "signed_volume", "realized_vol", "buy_pressure", "trade_count", "vwap_spread"]
    ].values.astype(np.float32)
    return feats[:max_ticks]


def build_regime_mask(feats_per_asset: list, window: int = REGIME_WINDOW) -> np.ndarray:
    """Construct the 2σ realized-volatility regime proxy mask (SWAP-15)."""
    import pandas as pd
    rv = feats_per_asset[0][:, 2]   # realized_vol from first asset (col index 2)
    s  = pd.Series(rv)
    mu = s.rolling(window, min_periods=1).mean()
    sg = s.rolling(window, min_periods=1).std().fillna(0)
    return ((s > mu + REGIME_SIGMA * sg)).astype(np.float32).values


def build_dataset(log):
    symbols  = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    all_feats = []

    for sym_dir in sorted(symbols):
        symbol = sym_dir.name.replace("symbol=", "")
        log.info(f"Processing {symbol}...")
        feats = engineer_features_for_symbol(sym_dir, SAMPLE_TICKS)
        if feats is not None:
            log.info(f"  {symbol}: {len(feats):,} 1-second ticks, {feats.shape[1]} features")
            all_feats.append(feats)

    min_len   = min(len(f) for f in all_feats)
    all_feats = [f[:min_len] for f in all_feats]

    feats_combined = np.concatenate(all_feats, axis=1)
    log.info(f"Combined feature matrix: {feats_combined.shape}")

    mid_returns = all_feats[0][:, 0]
    fut     = np.convolve(mid_returns, np.ones(HORIZON) / HORIZON, mode="full")[HORIZON - 1:]
    fut     = fut[:min_len]
    shifted = np.concatenate([fut[HORIZON:], np.repeat(fut[-1], HORIZON)])
    # Dynamically compute thresholds for balanced classes (33/33/33 split)
    alpha_down = np.percentile(shifted, 33.33)
    alpha_up   = np.percentile(shifted, 66.67)
    log.info(f"Dynamic return thresholds: DOWN < {alpha_down:.2e}, UP > {alpha_up:.2e}")

    labels  = np.ones(min_len, dtype=np.int64)
    labels[shifted > alpha_up] = 2
    labels[shifted < alpha_down] = 0

    regime = build_regime_mask(all_feats)

    log.metric("DOWN labels",    int((labels == 0).sum()))
    log.metric("STAT labels",    int((labels == 1).sum()))
    log.metric("UP labels",      int((labels == 2).sum()))
    log.metric("Regime %",       float(regime.mean() * 100), unit="%")
    return feats_combined, labels, regime




def delta_encode(x, threshold=0.15):
    d       = torch.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    on  = (d >  threshold).float()
    off = (d < -threshold).float()
    return torch.cat([on, off], dim=-1)


def make_windows(feats, labels, regime):
    # Normalize on TRAIN window only (fixes SWAP-02)
    n_total = len(feats) - SEQ_LEN - HORIZON
    n_train = int(n_total * TRAIN_FRAC)

    train_feats = feats[:n_train + SEQ_LEN]
    mu = train_feats.mean(0, keepdims=True)
    sd = train_feats.std(0,  keepdims=True) + 1e-8
    x  = (feats - mu) / sd

    idx = np.arange(n_total)
    X   = np.stack([x[i:i + SEQ_LEN] for i in idx]).astype(np.float32)
    y   = labels[idx + SEQ_LEN - 1]
    r   = np.stack([regime[i:i + SEQ_LEN].max() for i in idx]).astype(np.float32)

    n_val  = int(n_total * VAL_FRAC)
    tr_end = n_train
    va_end = n_train + n_val

    return (torch.tensor(X[:tr_end]),  torch.tensor(y[:tr_end]),  torch.tensor(r[:tr_end]),
            torch.tensor(X[tr_end:va_end]), torch.tensor(y[tr_end:va_end]), torch.tensor(r[tr_end:va_end]),
            torch.tensor(X[va_end:]),  torch.tensor(y[va_end:]),  torch.tensor(r[va_end:]))


# ── metrics ───────────────────────────────────────────────────────────────────
def macro_f1(pred, true, n=3):
    f1s = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        d  = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
    return sum(f1s) / len(f1s)


# ── training ──────────────────────────────────────────────────────────────────
CKPT_PATH = OUT_DIR / "checkpoint.pt"


def train(device, log: ExperimentLogger):
    torch.manual_seed(SEED); np.random.seed(SEED)

    log.section("Engineering features from Binance trade data")
    feats, labels, regime = build_dataset(log)
    Xtr, ytr, rtr, Xva, yva, rva, Xte, yte, rte = make_windows(feats, labels, regime)

    Str  = delta_encode(Xtr).to(device)
    Sva  = delta_encode(Xva).to(device)
    Ste  = delta_encode(Xte).to(device)
    ytr  = ytr.to(device);  yva  = yva.to(device);  yte  = yte.to(device)
    rtr  = rtr.to(device);  rva  = rva.to(device)

    majority = (yte == yte.mode().values).float().mean().item()
    log.section("Dataset summary")
    log.metric("train windows", len(Str))
    log.metric("val windows",   len(Sva))
    log.metric("test windows",  len(Ste))
    log.metric("majority baseline", majority)
    log.metric("input spike density", Str.mean().item())
    log.metric("regime % in train", rtr.mean().item() * 100, unit="%")

    model = ToySNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ce    = nn.CrossEntropyLoss()
    bce   = nn.BCEWithLogitsLoss()

    # ── resume from checkpoint if available ──────────────────────────────────
    start_epoch     = 0
    best_f1         = -1.0
    best_state      = None
    history         = []
    ckpt = load_checkpoint(CKPT_PATH, log)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_f1     = ckpt.get("best_f1", -1.0)
        best_state  = ckpt.get("best_state")
        history     = ckpt.get("history", [])
        log.info(f"Resuming from epoch {start_epoch + 1}/{EPOCHS}")
    else:
        log.section(f"Training SNN gate  ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})")
    t0 = time.time()

    for ep in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(len(Str), device=device)
        total_loss = 0.0
        nan_hit    = False

        for i in range(0, len(Str), BATCH_SIZE):
            b      = perm[i:i + BATCH_SIZE]
            logits, rate = model(Str[b])
            loss = (ce(logits, ytr[b])
                    + LAMBDA_RATE * rate
                    + LAMBDA_REGIME * bce(rate.expand(len(b)), rtr[b]))
            opt.zero_grad(); loss.backward(); opt.step()
            if guard_nan(loss, ep + 1, "A", log):
                nan_hit = True; break
            total_loss += loss.item() * len(b)

        if nan_hit:
            break

        model.eval()
        with torch.no_grad():
            vlogits, vrate = model(Sva)
            vf1 = macro_f1(vlogits.argmax(1), yva)

        history.append({"epoch": ep + 1, "loss": total_loss / len(Str), "val_f1": vf1})
        if vf1 > best_f1:
            best_f1    = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ── save checkpoint every epoch ───────────────────────────────────────
        save_checkpoint({
            "epoch": ep + 1, "stage": "A",
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_f1": best_f1,
            "best_state": best_state,
            "history": history,
        }, CKPT_PATH, log if (ep + 1) % 5 == 0 else None)  # log every 5 epochs

        if (ep + 1) % 5 == 0 or ep == start_epoch:
            elapsed = time.time() - t0
            eta     = elapsed / (ep - start_epoch + 1) * (EPOCHS - ep - 1)
            log.epoch(
                ep + 1, EPOCHS,
                loss=total_loss / len(Str),
                val_f1=vf1,
                extra={"spike": vrate.item()},
                eta_s=eta,
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tlogits, trate = model(Ste)
        pred = tlogits.argmax(1)

    results = {
        "experiment"      : "A — SNN Gate on Binance Trade Features",
        "timestamp"       : datetime.now().isoformat(),
        "device"          : str(device),
        "gpu_name"        : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "n_train"         : len(Str),
        "n_val"           : len(Sva),
        "n_test"          : len(Ste),
        "majority_baseline": majority,
        "accuracy"        : (pred == yte).float().mean().item(),
        "macro_f1"        : macro_f1(pred, yte),
        "spike_rate"      : trate.item(),
        "regime_proxy"    : f"realized_vol > mu + {REGIME_SIGMA}sigma (rolling {REGIME_WINDOW}-tick window)",
        "regime_frac_train": rtr.mean().item(),
        "swap_status"     : {
            "SWAP-01": "DONE — real Binance trade data",
            "SWAP-02": "DONE — scaler fitted on train window only",
            "SWAP-03": "PENDING — delta encoding, fixed threshold",
            "SWAP-15": "PARTIAL — 2sigma vol proxy, not verified with guide yet",
        }
    }

    return results, history, model


# ── results doc ───────────────────────────────────────────────────────────────
def write_results(results: dict, history: list):
    md = f"""# Experiment A — SNN Gate on Binance Trade Features

**Run date:** {results['timestamp']}  
**Device:** {results['gpu_name']}  
**Source code:** [`train.py`](./train.py)

---

## Dataset

| Property | Value |
|---|---|
| Source | Binance historical trades (`data/curated/trades/`) |
| Assets | BTCUSDT · ETHUSDT · SOLUSDT |
| Features per asset | 6 (log_return, OFI, realized_vol, buy_pressure, trade_count, vwap_spread) |
| Total features | {N_FEATURES} |
| Train windows | {results['n_train']:,} |
| Val windows | {results['n_val']:,} |
| Test windows | {results['n_test']:,} |
| Regime proxy | {results['regime_proxy']} |
| Regime % in train | {results['regime_frac_train']*100:.1f}% |

## Results

| Metric | Value |
|---|---|
| **Accuracy** | **{results['accuracy']:.3f}** |
| **Macro-F1** | **{results['macro_f1']:.3f}** |
| Majority-class baseline | {results['majority_baseline']:.3f} |
| Hidden spike rate | {results['spike_rate']:.4f} |
| Neurons silent per tick | {(1-results['spike_rate'])*100:.1f}% |

## Training History

| Epoch | Loss | Val Macro-F1 |
|---|---|---|
"""
    for h in history:
        md += f"| {h['epoch']} | {h['loss']:.4f} | {h['val_f1']:.3f} |\n"

    md += f"""
## Interpretation

This experiment validates that the SNN gate architecture can learn from **real Binance trade data**,
replacing the synthetic GBM generator used in the toy (SWAP-01 ✅).

The key improvements over the toy:

1. **Real OFI signal** — Order-flow imbalance computed from actual buy/sell classification
   rather than simulated AR(1) process.
2. **Train-window normalization** — Scaler fitted exclusively on the train split (SWAP-02 ✅),
   eliminating the future leakage present in the toy.
3. **Realized-volatility regime proxy** — The 2σ threshold replaces the synthetic ground-truth
   mask with a fully observable, reproducible label (SWAP-15 partial ✅).

## Known Limitations

- SWAP-03 still pending: delta encoding threshold is fixed. A Phase 2 ablation should compare
  delta vs. rate coding vs. latency-to-first-spike.
- SWAP-15 is a proxy, not ground truth. The regime supervision accuracy depends on whether
  volatility spikes correlate with the true structural shifts in BTC/ETH/SOL.
- Single seed, no walk-forward CV (SWAP-06/12 — deferred to Phase 3).

## SWAP Status

| # | Status | Note |
|---|---|---|
| SWAP-01 | ✅ Done | Real Binance trade data |
| SWAP-02 | ✅ Done | Scaler on train window only |
| SWAP-03 | ⏳ Pending | Fixed delta threshold |
| SWAP-15 | ⚠️ Partial | 2σ vol proxy — guide sign-off needed |
"""
    (OUT_DIR / "RESULTS.md").write_text(md, encoding="utf-8")
    print(f"\nResults saved to {OUT_DIR / 'RESULTS.md'}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    import os
    if os.environ.get("QSPINN_ALLOW_LEGACY_EXPA") != "1":
        raise SystemExit("Exp A is excluded from corrected v3.1 comparisons (legacy pipeline). "
                         "Set QSPINN_ALLOW_LEGACY_EXPA=1 to run it as a labelled legacy experiment.")
    log    = ExperimentLogger(OUT_DIR / "train.log",
                              experiment_name="Exp A — SNN Gate on Binance")
    device = get_device(verbose=True)

    try:
        results, history, model = train(device, log)
    except Exception as e:
        log.warn(f"Training crashed: {type(e).__name__}: {e}")
        log.warn("A checkpoint was saved before the crash — re-run to resume automatically.")
        raise

    # Save model
    log.section("Saving outputs")
    torch.save(model.state_dict(), OUT_DIR / "snn_binance.pt")
    log.info(f"Model weights  → {OUT_DIR / 'snn_binance.pt'}")

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON   → {OUT_DIR / 'results.json'}")

    write_results(results, history)
    log.info(f"RESULTS.md     → {OUT_DIR / 'RESULTS.md'}")

    # Clean up checkpoint on successful completion
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        log.info("Checkpoint deleted (run completed successfully)")

    log.done(summary={
        "Accuracy"          : results["accuracy"],
        "Macro-F1"          : results["macro_f1"],
        "Majority baseline" : results["majority_baseline"],
        "Neurons silent"    : f"{(1-results['spike_rate'])*100:.1f}%",
    })


if __name__ == "__main__":
    main()
