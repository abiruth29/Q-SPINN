"""
Exp B — forward early-warning study (task 10).  RUN BY THE TEAMMATE after training.

    python experiments/exp_B_hybrid_fi2010/early_warning_study.py --horizon 10 --seed 0 --weights <.pt>

Label: future realised-vol (t, t+tau] above the TRAIN 95th percentile, computed inside each stock-day
segment (never across joins).  Scores compared on the untouched test days:
    omega (numerical abscissa of S)  vs  backward rolling-vol  vs  cheap-expert (SNN) confidence-free baseline
Alert probability is Platt-calibrated on VALIDATION; alert threshold also chosen on validation (max F1).
Score metrics (AUROC/AUPRC/Brier) use the future-window label; lead time and false alerts use REALISED shock onsets
(|delta mid| above the train 99th pct) on the row clock, segment-safe (`routing.event_alert_metrics`).  omega stays a latent-flow
diagnostic unless it beats the rolling-vol baseline here.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train as T
import routing as R
from qspinn_core import window_starts
from logger import ExperimentLogger
from device import get_device


def seg_future_vol(X, seg, tau):
    mid = (X[:, 0] + X[:, 2]) / 2.0
    fv = np.full(len(X), np.nan)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        fv[idx] = R.future_vol(np.r_[0.0, np.diff(mid[idx])], tau)
    return fv


@torch.no_grad()
def omega_scores(model, X, device, bs=1024):
    model.eval(); out = []
    dt = torch.ones(1, T.SEQ_LEN)
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device).contiguous()
        S = model.heavy_path(xb)[3]
        out.append(torch.linalg.eigvalsh(S)[:, -1].cpu())
    return torch.cat(out).numpy()


def best_f1_threshold(p, y):
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.999, 60))):
        a = p >= t; tp = (a & (y == 1)).sum(); f = 2 * tp / max(a.sum() + (y == 1).sum(), 1)
        if f > best: best, thr = f, t
    return float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", required=True); ap.add_argument("--tau", type=int, default=20)
    a = ap.parse_args()
    log = ExperimentLogger(HERE / f"ews_k{a.horizon}_s{a.seed}.log", experiment_name="Exp B early warning")
    device = get_device(verbose=True)
    d = T.load_fi2010_setup2(log, f"label_k{a.horizon}")
    fv = {k: seg_future_vol(d[k]["X"], d[k]["seg"], a.tau) for k in d}
    thr = float(np.quantile(fv["train"][~np.isnan(fv["train"])], 0.95))      # TRAIN only

    model = T.QSPINN().to(device); model.encoder.fit(d["train"]["X"])
    T.load_state_compat(model, torch.load(a.weights, map_location=device)); model.eval()

    def abs_ret(X, seg):                     # observed per-row shock magnitude; zero across stock/day joins
        mid = (X[:, 0] + X[:, 2]) / 2.0
        r = np.r_[0.0, np.abs(np.diff(mid))]
        r[np.r_[True, seg[1:] != seg[:-1]]] = 0.0
        return r
    ev_thr = float(np.quantile(abs_ret(d["train"]["X"], d["train"]["seg"]), 0.99))   # TRAIN only

    res, P = {}, {}
    for k in ("val", "test"):
        st = window_starts(len(d[k]["X"]), T.SEQ_LEN, d[k]["seg"], a.tau)
        last = st + T.SEQ_LEN - 1
        y = (fv[k][last] > thr).astype(np.int64)
        Xw = torch.from_numpy(d[k]["X"]).unfold(0, T.SEQ_LEN, 1).transpose(1, 2)[st]
        mid = (d[k]["X"][:, 0] + d[k]["X"][:, 2]) / 2.0
        rv = np.zeros(len(mid))             # causal rolling vol per segment, sampled at the window end
        for s_ in np.unique(d[k]["seg"]):
            ii = np.flatnonzero(d[k]["seg"] == s_); rv[ii] = R.rolling_vol_score(mid[ii])
        P[k] = dict(y=y, last=last, omega=omega_scores(model, Xw, device), rv=rv[last])
    seg_te = d["test"]["seg"]
    onsets = R.realised_onsets(abs_ret(d["test"]["X"], seg_te), ev_thr, refractory=a.tau, seg=seg_te)
    for name, key in (("omega", "omega"), ("rolling_vol", "rv")):
        params = R.platt_fit(P["val"][key], P["val"]["y"].astype(float))
        pv, pt = R.platt_apply(P["val"][key], params), R.platt_apply(P["test"][key], params)
        at = best_f1_threshold(pv, P["val"]["y"])
        row_alert = np.zeros(len(d["test"]["X"]), dtype=bool); row_alert[P["test"]["last"]] = pt >= at
        res[name] = dict(R.alert_metrics(pt, P["test"]["y"]), alert_threshold=at,
                         test_label_prevalence=float(P["test"]["y"].mean()),
                         **R.event_alert_metrics(row_alert, onsets, a.tau, seg_te, rows_per_unit_time=1000.0))
    res["event_definition"] = dict(signal="|delta mid| per row (0 across joins)", threshold_train_q99=ev_thr,
                                   refractory_rows=a.tau, false_alerts_per_unit_time="per 1000 rows")
    out = HERE / f"early_warning_k{a.horizon}_s{a.seed}.json"
    out.write_text(json.dumps(res, indent=2)); log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
