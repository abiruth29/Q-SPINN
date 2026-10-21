"""
Exp B — routing study (tasks 08-10).  RUN BY THE TEAMMATE after training finishes.

    python experiments/exp_B_hybrid_fi2010/routing_study.py --horizon 10 --seed 0 \
        --weights experiments/exp_B_hybrid_fi2010/qspinn_fi2010_k10_s0.pt

Produces routing_study_k{H}_s{S}.json with:
  * validation macro-F1 vs routing fraction for: learned gate, confidence cascade, rolling-vol,
    benefit head (and random); the fraction is chosen on VALIDATION only
  * the same routers evaluated on TEST at that fixed validation threshold
  * true conditional-inference latency (heavy called only on selected rows) vs dense heavy
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
        heavy = model.heavy_path(xb)[0]
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
    def mid(k):  # last-step mid price for the rolling-vol baseline (causal)
        return ((data[k]["X"][:, -1, 0] + data[k]["X"][:, -1, 2]) / 2).numpy()
    rng = np.random.default_rng(a.seed)
    scores = {
        "learned_gate": lambda k: data[k]["g"],
        "confidence_cascade": lambda k: R.confidence_score(data[k]["c"]).numpy(),
        "rolling_vol": lambda k: R.rolling_vol_score(mid(k)),
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

    # conditional inference latency (task 09)
    Xte = data["test"]["X"][:20000].to(device); cheap = data["test"]["c"][:20000].to(device)
    sel = torch.from_numpy(scores["learned_gate"]("test")[:20000] >= report["routers"]["learned_gate"]["val_threshold"]).to(device)
    heavy_fn = lambda xb: model.heavy_path(xb)[0]
    report["latency_s"] = dict(
        dense_heavy=R.timed(lambda: heavy_fn(Xte), device=device),
        conditional=R.timed(lambda: R.conditional_infer(cheap, sel, heavy_fn, Xte), device=device),
        routed_fraction=float(sel.float().mean()), device=str(device))
    out.write_text(json.dumps(report, indent=2)); log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
