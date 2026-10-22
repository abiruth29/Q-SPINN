"""
experiments/routing.py
======================
Phase 2 (tasks 08-10).  Pure torch/numpy so every piece is unit-testable on synthetic data.

08  Prediction-aware routing with a compute budget
    * `benefit_targets`     - per-sample improvement of heavy over cheap (loss gap / margin gap)
    * `fit_benefit_head`    - ridge regression of that benefit on cheap-side features (closed form)
    * `confidence_score`    - cascade baseline: 1 - max softmax of the cheap expert
    * `select_threshold`    - threshold chosen for a routing fraction on VALIDATION scores only
    * `f1_vs_fraction`      - validation macro-F1 curve over routing fractions 0..1
09  Real conditional inference
    * `conditional_infer`   - heavy expert is called ONLY on selected rows, outputs scattered back
10  Forward early warning
    * `forward_instability_labels`, `platt_fit/apply`, `alert_metrics`, `rolling_vol_score`
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch


# ───────────────────────────── 08: routing ─────────────────────────────────
def _macro_f1(pred: np.ndarray, true: np.ndarray, n: int = 3) -> float:
    f = []
    for c in range(n):
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c)); d = 2 * tp + fp + fn
        f.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f))


def benefit_targets(cheap: torch.Tensor, heavy: torch.Tensor, y: torch.Tensor, kind: str = "loss"):
    """
    benefit_i = how much better the heavy expert is for sample i  (positive => route to heavy).
    kind="loss":   CE(cheap) - CE(heavy)
    kind="margin": margin(heavy) - margin(cheap) with margin = z_y - max_{c!=y} z_c
    """
    if kind == "loss":
        ce = torch.nn.functional.cross_entropy
        return (ce(cheap, y, reduction="none") - ce(heavy, y, reduction="none")).detach()
    zy = lambda z: z.gather(1, y[:, None]).squeeze(1)
    other = lambda z: z.masked_fill(torch.nn.functional.one_hot(y, z.shape[1]).bool(), -1e30).max(1)[0]
    mh, mc = zy(heavy) - other(heavy), zy(cheap) - other(cheap)
    return (mh - mc).detach()


def fit_benefit_head(feats: np.ndarray, benefit: np.ndarray, l2: float = 1e-3):
    """Closed-form ridge fit on TRAIN features. Returns (w, b); score = feats @ w + b."""
    X = np.c_[feats, np.ones(len(feats))]
    A = X.T @ X + l2 * np.eye(X.shape[1])
    wb = np.linalg.solve(A, X.T @ benefit)
    return wb[:-1], float(wb[-1])


def apply_benefit_head(feats: np.ndarray, head) -> np.ndarray:
    w, b = head
    return feats @ w + b


def confidence_score(cheap_logits: torch.Tensor) -> torch.Tensor:
    """Cascade baseline: low cheap-expert confidence => high score."""
    return 1.0 - torch.softmax(cheap_logits, dim=1).max(1)[0]


def rolling_vol_score(mid: np.ndarray, window: int = 20) -> np.ndarray:
    """Backward rolling realised-vol router baseline (strictly causal)."""
    r = np.r_[0.0, np.diff(mid)]
    out = np.zeros(len(r))
    for i in range(len(r)):
        out[i] = r[max(0, i - window + 1): i + 1].std()
    return out


def segmented_rolling_vol(mid: np.ndarray, seg: np.ndarray, window: int = 20) -> np.ndarray:
    """Causal rolling vol computed independently inside each stock/day segment (never across a join)."""
    out = np.zeros(len(mid))
    for s_ in np.unique(seg):
        ii = np.flatnonzero(seg == s_)
        out[ii] = rolling_vol_score(mid[ii], window)
    return out


def select_threshold(val_scores: np.ndarray, fraction: float) -> float:
    """
    Budget-safe threshold chosen on VALIDATION scores only.  Policy for ties: the routed set is
    {score >= t}; t is the smallest value whose routed share does not exceed `fraction`
    (so tied scores are routed together or not at all, and the budget is never exceeded; with
    all-equal scores a fraction < 1 routes nothing).  fraction<=0 -> inf, fraction>=1 -> -inf.
    """
    if fraction <= 0: return float("inf")
    if fraction >= 1: return float("-inf")
    v = np.sort(np.asarray(val_scores))
    n = len(v)
    uniq = np.unique(v)[::-1]                                   # descending distinct values
    # share(t) = #{v >= t}/n via searchsorted on the sorted array
    share = (n - np.searchsorted(v, uniq, side="left")) / n
    ok = np.flatnonzero(share <= fraction + 1e-12)
    return float(uniq[ok[-1]]) if len(ok) else float("inf")


def routed_predictions(scores, cheap_pred, heavy_pred, thr):
    sel = scores >= thr
    return np.where(sel, heavy_pred, cheap_pred), sel


def f1_vs_fraction(scores, cheap_pred, heavy_pred, y, fractions: Sequence[float] = tuple(np.linspace(0, 1, 11))):
    """Macro-F1 and actual routed fraction as the budget varies (use on validation to choose)."""
    rows = []
    for f in fractions:
        thr = select_threshold(scores, f)
        pred, sel = routed_predictions(scores, cheap_pred, heavy_pred, thr)
        rows.append(dict(target_fraction=float(f), routed=float(sel.mean()), macro_f1=_macro_f1(pred, y)))
    return rows


def choose_budget(curve_val, min_f1_ratio_of_dense: float = 0.99):
    """Smallest validation fraction whose macro-F1 >= ratio * (all-heavy macro-F1)."""
    dense = curve_val[-1]["macro_f1"]
    for r in curve_val:
        if r["macro_f1"] >= min_f1_ratio_of_dense * dense:
            return r["target_fraction"]
    return 1.0


# ─────────────────────────── 09: conditional inference ─────────────────────
@torch.no_grad()
def conditional_infer(cheap_logits: torch.Tensor, sel: torch.Tensor,
                      heavy_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                      chunk: int = 4096) -> Dict[str, object]:
    """
    Run the heavy expert only on rows where `sel` is True and scatter its logits into a copy of the
    cheap logits.  Zero selected => `heavy_fn` is never called.  Diagnostics that need the heavy
    training path (covariance NLL, PDE residual, norm loss) are NOT produced here.
    """
    out = cheap_logits.clone()
    idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
    calls = 0
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        out[j] = heavy_fn(x[j])
        calls += len(j)
    return dict(logits=out, heavy_rows=calls, routed=len(idx) / max(len(sel), 1),
                diagnostics="unavailable (skipped in conditional inference)")


def timed(fn, *args, device: Optional[torch.device] = None, warmup: int = 2, reps: int = 5):
    """Median wall-clock seconds with warm-up and device synchronisation (CPU or CUDA)."""
    sync = (lambda: torch.cuda.synchronize()) if (device is not None and device.type == "cuda") else (lambda: None)
    for _ in range(warmup): fn(*args)
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(*args); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ─────────────────────────── 10: forward early warning ─────────────────────
def future_vol(returns: np.ndarray, tau: int) -> np.ndarray:
    """Realised vol over the strictly-future span (t, t+tau]; NaN where the span is incomplete."""
    n = len(returns)
    c1 = np.r_[0.0, np.cumsum(returns)]; c2 = np.r_[0.0, np.cumsum(np.square(returns))]
    fv = np.full(n, np.nan)
    i = np.arange(max(n - tau, 0))
    m = (c1[i + 1 + tau] - c1[i + 1]) / tau
    fv[i] = np.sqrt(np.maximum((c2[i + 1 + tau] - c2[i + 1]) / tau - m ** 2, 0.0))
    return fv


def forward_instability_labels(returns: np.ndarray, tau: int, train_end: int, q: float = 0.95):
    """
    label_t = 1 iff realised vol over the FUTURE span (t, t+tau] exceeds the q-quantile of future-vol
    computed on rows < train_end only.  The span starts strictly after t so it never overlaps a window
    ending at t.  Last `tau` rows are -1 (undefined).
    """
    fv = future_vol(returns, tau)
    ft = fv[: max(train_end - tau, 0)]                       # support (t, t+tau] must end inside training
    thr = np.quantile(ft[~np.isnan(ft)], q)
    lab = (fv > thr).astype(np.int64); lab[np.isnan(fv)] = -1
    return lab, float(thr)


def platt_fit(scores: np.ndarray, y: np.ndarray, iters: int = 200, lr: float = 0.5):
    """Logistic calibration p = sigmoid(a*s + b) on VALIDATION data (Newton-free GD on standardised s)."""
    mu, sd = scores.mean(), scores.std() + 1e-12
    s = (scores - mu) / sd
    a, b = 1.0, float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * s + b)))
        a -= lr * np.mean((p - y) * s); b -= lr * np.mean(p - y)
    return a, b, mu, sd


def platt_apply(scores, params):
    a, b, mu, sd = params
    return 1 / (1 + np.exp(-(a * (scores - mu) / sd + b)))


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def auprc(scores: np.ndarray, y: np.ndarray) -> float:
    """Average precision with equal-score groups collapsed to one threshold (order-invariant under ties)."""
    scores, y = np.asarray(scores), np.asarray(y)
    npos = y.sum()
    if npos == 0: return float("nan")
    o = np.argsort(-scores, kind="mergesort"); ss, ys = scores[o], y[o]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]                       # last index of each tie group
    tp_g, k_g = tp[last], k[last]
    prec, rec = tp_g / k_g, tp_g / npos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def alert_metrics(prob: np.ndarray, y: np.ndarray):
    """
    Score quality against the FUTURE-WINDOW label y (label_t = instability inside (t, t+tau]):
    AUROC, AUPRC (tie-grouped), Brier.  This says nothing about lead time or false alarms per unit
    time - those need a realised-event clock, see `event_alert_metrics`.
    """
    return dict(auroc=auroc(prob, y), auprc=auprc(prob, y), brier=float(np.mean((prob - y) ** 2)))


def realised_onsets(signal: np.ndarray, thr: float, refractory: int, seg: Optional[np.ndarray] = None):
    """
    Realised event onsets: rows where `signal` (an OBSERVED shock magnitude at that row, e.g. |return|)
    first exceeds `thr`; further exceedances within `refractory` rows of an onset, or in the same
    episode, are merged.  Onsets are found per segment (stock/day) so nothing crosses a join.
    """
    signal = np.asarray(signal); n = len(signal)
    seg = np.zeros(n, dtype=np.int64) if seg is None else np.asarray(seg)
    onsets, last = [], {}
    for t in np.flatnonzero(signal > thr):
        sg = seg[t]
        if sg in last and t - last[sg] <= refractory:
            last[sg] = t if t - last[sg] <= 1 else last[sg]       # extend only a contiguous exceedance run
            continue
        onsets.append(int(t)); last[sg] = int(t)
    return np.array(onsets, dtype=np.int64)


def _seg_bounds(seg: np.ndarray):
    """Per-row first and last row index of the (contiguous) segment each row belongs to."""
    n = len(seg)
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.r_[0, change]; ends = np.r_[change - 1, n - 1]
    lens = ends - starts + 1
    return np.repeat(starts, lens), np.repeat(ends, lens)


def alarms_from_alerts(alert: np.ndarray, seg: np.ndarray, refractory: int, renew_every: Optional[int]):
    """
    ONE alarm policy used for detection, lead time AND false alerts.
    An *episode* is a run of alerted rows inside one segment (gaps <= refractory rows merge).  An alarm is
    raised at the first row of each episode and, if `renew_every` is not None, renewed at the first alerted
    row at least `renew_every` rows after the previous alarm while the episode continues.  `renew_every=None`
    means starts only.
    """
    alarms = []
    prev, last_alarm = None, None
    for t in np.flatnonzero(alert):
        new_ep = prev is None or seg[t] != seg[prev] or t - prev > refractory + 1
        if new_ep or (renew_every is not None and t - last_alarm >= renew_every):
            alarms.append(int(t)); last_alarm = t
        prev = t
    return np.array(alarms, dtype=np.int64)


def event_alert_metrics(alert: np.ndarray, onsets: np.ndarray, tau: int, seg: Optional[np.ndarray] = None,
                        eligible: Optional[np.ndarray] = None, rows_per_unit_time: Optional[float] = None,
                        refractory: Optional[int] = None, renew_every="tau"):
    """
    Event-based warning evaluation on ONE row clock (concatenated rows + segment ids), one matching policy.

    alert     bool per row, True where the model raised/maintained an alert (must be False on rows without a
              prediction).
    eligible  bool per row: rows where the model COULD have predicted (the at-risk clock).  None = all rows.
    ALARMS    from `alarms_from_alerts` (episode starts, renewed every `renew_every` rows; "tau" -> tau,
              None -> starts only).
    * An event at row r is EVALUABLE only if the whole lead window [r-tau, r) lies in r's segment and every row
      in it is eligible; otherwise it is CENSORED (warm-up at segment start, scoring cut-off at segment end).
    * An evaluable event is DETECTED if an alarm lies in [r-tau, r); lead = r - earliest such alarm (rows).
    * An alarm is FALSE if its follow window (a, a+tau] lies inside its segment and contains no realised
      onset; if the window is cut by the segment end the alarm is CENSORED (not counted either way).
    * Exposure for rates = number of eligible rows (not all rows).  Rates are per `rows_per_unit_time`
      (e.g. 1000 rows) - they are NOT per-hour rates unless you convert with a real clock.
    """
    alert = np.asarray(alert, dtype=bool); n = len(alert)
    seg = np.zeros(n, dtype=np.int64) if seg is None else np.asarray(seg)
    elig = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    alert = alert & elig
    onsets = np.asarray(onsets, dtype=np.int64)
    refractory = tau if refractory is None else refractory
    renew = tau if renew_every == "tau" else renew_every
    sstart, send = _seg_bounds(seg)
    alarms = alarms_from_alerts(alert, seg, refractory, renew)

    leads, evaluable, censored_events = [], 0, 0
    for r in onsets:
        lo = r - tau
        if lo < sstart[r] or not elig[lo:r].all():
            censored_events += 1; continue
        evaluable += 1
        hit = alarms[(alarms >= lo) & (alarms < r)]
        if len(hit): leads.append(int(r - hit.min()))
    false_alarms = censored_alarms = 0
    for a in alarms:
        if a + tau > send[a]:
            censored_alarms += 1; continue
        if not np.any((onsets > a) & (onsets <= a + tau)):
            false_alarms += 1
    exposure = int(elig.sum())
    out = dict(events_total=int(len(onsets)), events_evaluable=evaluable, events_censored=censored_events,
               detected=len(leads),
               detection_rate=(len(leads) / evaluable) if evaluable else float("nan"),
               mean_lead_rows=float(np.mean(leads)) if leads else float("nan"),
               alarms=int(len(alarms)), false_alarms=false_alarms, alarms_censored=censored_alarms,
               exposure_rows=exposure,
               policy=dict(tau=tau, refractory=refractory, renew_every=renew))
    if rows_per_unit_time:
        out["false_alarms_per_unit_time"] = false_alarms / (exposure / rows_per_unit_time) if exposure else float("nan")
    return out
