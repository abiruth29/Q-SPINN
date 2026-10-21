"""
Exp B — routing study (tasks 08-10).  RUN BY THE TEAMMATE after training finishes.

    python experiments/exp_B_hybrid_fi2010/routing_study.py --horizon 10 --seed 0 \
        --weights experiments/exp_B_hybrid_fi2010/qspinn_fi2010_k10_s0.pt

Produces routing_study_k{H}_s{S}.json with:
  * validation macro-F1 vs routing fraction for: learned gate, confidence cascade, rolling-vol,
    benefit head (and random); the fraction is chosen on VALIDATION only
  * the same routers evaluated on TEST at that fixed validation threshold
  * matched full-pipeline latency (raw windows in -> logits out, eval/no_grad, same chunking and warm-up):
    cheap-only, dense heavy, and the real conditional hybrid (learned gate and forced routing fractions)
No test label is used to choose any threshold.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train as T
import routing as R
from logger import ExperimentLogger
from device import get_device


@torch.no_grad()
def collect(model, X, dt, device, bs=1024):
    """Cheap logits, dense-heavy logits, gate logit and spike-rate features for every window."""
    model.eval(); C, Hh, G, F = [], [], [], []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device).contiguous(); db = dt[i:i + bs].to(device).contiguous()
        sp = model.encode(xb)
        cheap, g, rate, gl = model.gate(sp, db)
        heavy = model.heavy_infer(xb)
        C.append(cheap.cpu()); Hh.append(heavy.cpu()); G.append(gl.reshape(len(xb), -1)[:, 0].cpu())
        F.append(torch.stack([sp.float().mean((1, 2)), sp.float().abs().sum((1, 2))], 1).cpu())
    return torch.cat(C), torch.cat(Hh), torch.cat(G).numpy(), torch.cat(F).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", required=True); ap.add_argument("--legacy_split", action="store_true")
    a = ap.parse_args()
    out = HERE / f"routing_study_k{a.horizon}_s{a.seed}.json"
    log = ExperimentLogger(HERE / f"routing_study_k{a.horizon}_s{a.seed}.log", experiment_name="Exp B routing study")
    device = get_device(verbose=True)
    lc = f"label_k{a.horizon}"
    d = T.load_fi2010_setup2(log, lc)
    segs = {k: d[k]["seg"] for k in d}
    W = {}
    for k in ("train", "val", "test"):
        reg = np.zeros(len(d[k]["X"]), dtype=np.float32)
        W[k] = T.make_windows_from_array(d[k]["X"], d[k]["y"], reg, segs[k], a.horizon)
    model = T.QSPINN().to(device)
    model.encoder.fit(d["train"]["X"])
    T.load_state_compat(model, torch.load(a.weights, map_location=device)); model.eval()

    data = {}
    for k in ("train", "val", "test"):
        X, y, _, dt = W[k]
        c, h, g, f = collect(model, X, dt, device)
        data[k] = dict(c=c, h=h, g=g, f=f, y=y.numpy(), cp=c.argmax(1).numpy(), hp=h.argmax(1).numpy(), X=X)

    # routers: score higher => route to heavy
    head = R.fit_benefit_head(data["train"]["f"], R.benefit_targets(data["train"]["c"], data["train"]["h"],
                                                                    torch.tensor(data["train"]["y"])).numpy())
    from qspinn_core import window_starts
    def rv_at_windows(k):    # causal rolling vol per stock/day segment, sampled at each window's last row
        Xr, sg = d[k]["X"], d[k]["seg"]
        last = window_starts(len(Xr), T.SEQ_LEN, sg, a.horizon) + T.SEQ_LEN - 1
        return R.segmented_rolling_vol((Xr[:, 0] + Xr[:, 2]) / 2.0, sg)[last]
    rng = np.random.default_rng(a.seed)
    scores = {
        "learned_gate": lambda k: data[k]["g"],
        "confidence_cascade": lambda k: R.confidence_score(data[k]["c"]).numpy(),
        "rolling_vol": lambda k: rv_at_windows(k),
        "benefit_head": lambda k: R.apply_benefit_head(data[k]["f"], head),
        "random": lambda k: rng.random(len(data[k]["y"])),
    }
    fr = list(np.linspace(0, 1, 21)); report = {"routers": {}}
    for name, fn in scores.items():
        sv, st = fn("val"), fn("test")
        cv = R.f1_vs_fraction(sv, data["val"]["cp"], data["val"]["hp"], data["val"]["y"], fr)
        budget = R.choose_budget(cv, 0.99)
        thr = R.select_threshold(sv, budget)                       # validation only
        pred, sel = R.routed_predictions(st, data["test"]["cp"], data["test"]["hp"], thr)
        report["routers"][name] = dict(val_curve=cv, chosen_fraction=budget, val_threshold=thr,
                                       test_routed=float(sel.mean()), test_macro_f1=R._macro_f1(pred, data["test"]["y"]))
    report["test_cheap_f1"] = R._macro_f1(data["test"]["cp"], data["test"]["y"])
    report["test_dense_f1"] = R._macro_f1(data["test"]["hp"], data["test"]["y"])

    # Matched FULL-PIPELINE latency (review: same input prep, same chunking, eval + no_grad, warm-up, device sync).
    # Every pipeline starts from the same raw CPU windows and includes host->device copy, encoding and (where
    # applicable) the gate, so cheap / dense-heavy / hybrid are directly comparable.
    N = min(20000, len(data["test"]["X"])); BS = 1024
    Xraw = data["test"]["X"][:N]; DT = torch.ones(1, T.SEQ_LEN).expand(N, T.SEQ_LEN)
    def run(kind, frac=None):
        with torch.no_grad():
            for i in range(0, N, BS):
                xb = Xraw[i:i + BS].to(device).contiguous(); db = DT[i:i + BS].to(device).contiguous()
                if kind == "dense_heavy":
                    model.heavy_infer(xb)
                else:
                    sp = model.encode(xb)
                    if kind == "cheap_only":
                        model.gate(sp, db)
                    else:   # hybrid: real conditional execution; `frac` forces a routing fraction, None = learned gate
                        force = None if frac is None else (torch.arange(len(xb), device=device) < int(round(frac * len(xb))))
                        model.infer(sp, xb, db, chunk=BS, sel=force)
    lat = dict(device=str(device), n_windows=N, batch=BS, mode="eval+no_grad, cold-start excluded (2 warm-up runs)")
    lat["cheap_only_s"] = R.timed(lambda: run("cheap_only"), device=device)
    lat["dense_heavy_s"] = R.timed(lambda: run("dense_heavy"), device=device)
    lat["hybrid_learned_gate_s"] = R.timed(lambda: run("hybrid"), device=device)
    lat["hybrid_at_fraction_s"] = {str(f): R.timed(lambda f=f: run("hybrid", f), device=device) for f in (0.0, 0.1, 0.25, 0.5, 1.0)}
    report["latency"] = lat
    out.write_text(json.dumps(report, indent=2), encoding="utf-8"); log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
