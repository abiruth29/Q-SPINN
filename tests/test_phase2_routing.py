"""Tasks 08-10 verify criteria on synthetic data."""
import numpy as np, torch
import routing as R

rng = np.random.default_rng(0)


def _synthetic(n=3000):
    y = rng.integers(0, 3, n)
    onehot = np.eye(3)[y]
    cheap = onehot * 0.3 + rng.normal(0, 1.0, (n, 3))      # weak expert
    heavy = onehot * 3.0 + rng.normal(0, 0.5, (n, 3))      # strong expert
    return y, cheap, heavy


def test_benefit_positive_when_heavy_better():
    y, c, h = _synthetic()
    for kind in ("loss", "margin"):
        b = R.benefit_targets(torch.tensor(c), torch.tensor(h), torch.tensor(y), kind).numpy()
        assert b.mean() > 0


def test_threshold_from_validation_only_and_curve_monotone_endpoints():
    y, c, h = _synthetic()
    cheap_pred, heavy_pred = c.argmax(1), h.argmax(1)
    score = R.confidence_score(torch.tensor(c)).numpy()
    n = len(y) // 2
    thr = R.select_threshold(score[:n], 0.3)
    # changing the TEST half (labels & scores) cannot change a val-derived threshold
    score2 = score.copy(); score2[n:] += 5
    assert thr == R.select_threshold(score2[:n], 0.3)
    curve = R.f1_vs_fraction(score[:n], cheap_pred[:n], heavy_pred[:n], y[:n])
    assert curve[0]["routed"] == 0 and curve[-1]["routed"] == 1
    assert curve[-1]["macro_f1"] > curve[0]["macro_f1"]
    # a confidence cascade beats routing the same fraction at random
    rand = rng.random(n)
    mid = R.f1_vs_fraction(score[:n], cheap_pred[:n], heavy_pred[:n], y[:n], [0.5])[0]["macro_f1"]
    rnd = R.f1_vs_fraction(rand, cheap_pred[:n], heavy_pred[:n], y[:n], [0.5])[0]["macro_f1"]
    assert mid > rnd
    assert R.choose_budget(curve, 0.99) <= 1.0


def test_benefit_head_learns_signal():
    X = rng.normal(size=(2000, 4)); b = X[:, 0] * 2 - X[:, 1] + rng.normal(0, .1, 2000)
    head = R.fit_benefit_head(X, b)
    assert np.corrcoef(R.apply_benefit_head(X, head), b)[0, 1] > 0.99


def test_conditional_inference_exactness_and_call_counts():
    n = 50
    x = torch.randn(n, 7); cheap = torch.randn(n, 3)
    W = torch.randn(7, 3); calls = []
    def heavy(xb): calls.append(len(xb)); return xb @ W
    out0 = R.conditional_infer(cheap, torch.zeros(n, dtype=torch.bool), heavy, x)
    assert calls == [] and out0["heavy_rows"] == 0 and torch.equal(out0["logits"], cheap)
    out1 = R.conditional_infer(cheap, torch.ones(n, dtype=torch.bool), heavy, x)
    assert torch.allclose(out1["logits"], x @ W) and out1["heavy_rows"] == n
    sel = torch.zeros(n, dtype=torch.bool); sel[[3, 10, 41]] = True; calls.clear()
    out2 = R.conditional_infer(cheap, sel, heavy, x)
    assert sum(calls) == 3 and torch.equal(out2["logits"][~sel], cheap[~sel])
    assert torch.allclose(out2["logits"][sel], (x @ W)[sel])


def test_timed_returns_positive():
    assert R.timed(lambda: torch.randn(64, 64) @ torch.randn(64, 64)) > 0


def test_forward_labels_strictly_future_and_train_threshold():
    r = np.zeros(200); r[101:106] = np.array([5, -5, 5, -5, 5.0])
    lab, thr = R.forward_instability_labels(r, tau=5, train_end=80, q=0.9)
    assert lab[100] == 1 and lab[95] == 0 and (lab[-5:] == -1).all()
    r2 = r.copy(); r2[150:] += rng.normal(0, 50, 50)
    assert R.forward_instability_labels(r2, 5, 80, 0.9)[1] == thr


def test_alert_metrics_perfect_and_random():
    y = np.zeros(1000, dtype=int); y[300:320] = 1; y[700:710] = 1
    perfect = y.astype(float) * .9 + .05
    m = R.alert_metrics(perfect, y, 0.5, tau=10)
    assert m["auroc"] > 0.99 and m["auprc"] > 0.99 and m["brier"] < 0.01 and m["events"] == 2
    noise = rng.random(1000)
    assert abs(R.alert_metrics(noise, y, 0.5, 10)["auroc"] - 0.5) < 0.15
    # early alert gives positive lead time
    early = np.zeros(1000); early[290:320] = 1; early[690:710] = 1
    assert R.alert_metrics(early, y, 0.5, 20)["mean_lead"] >= 5


def test_platt_calibration_improves_brier():
    s = rng.normal(size=4000); y = (rng.random(4000) < 1 / (1 + np.exp(-(3 * s - 2)))).astype(float)
    p = R.platt_apply(s, R.platt_fit(s, y))
    raw = 1 / (1 + np.exp(-s))
    assert np.mean((p - y) ** 2) < np.mean((raw - y) ** 2)
