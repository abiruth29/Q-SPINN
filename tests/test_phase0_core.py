"""Focused unit tests for Phase 0 (tasks 01-06). Synthetic data only; no FI-2010 needed."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load_expB():
    spec = importlib.util.spec_from_file_location("expB", ROOT / "experiments/exp_B_hybrid_fi2010/train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def B():
    return _load_expB()


@pytest.fixture()
def tiny(B):
    torch.manual_seed(0)
    X = torch.randn(300, B.N_FEATURES).numpy()
    model = B.QSPINN()
    model.encoder.fit(X)
    xb = torch.randn(16, B.SEQ_LEN, B.N_FEATURES)
    dt = torch.ones(16, B.SEQ_LEN)
    return model, xb, dt


# ── 01 ────────────────────────────────────────────────────────────────────────
def test_bad_update_never_changes_params():
    from qspinn_core import StepGuard
    w = nn.Parameter(torch.zeros(3))
    opt = torch.optim.SGD([w], lr=1.0)
    g = StepGuard()
    before = w.detach().clone()
    # non-finite loss
    assert not g.step(w.sum() * float("nan"), [w], opt)
    # finite loss, non-finite gradient (sqrt at 0)
    assert not g.step(torch.sqrt((w ** 2).sum()), [w], opt)
    assert torch.equal(w.detach(), before)
    assert g.n_skipped == 2 and g.n_steps == 0
    # good step still works and is clipped
    assert g.step((w - 5).pow(2).sum() * 1e6, [w], opt)
    assert w.grad.norm() <= g.max_grad_norm + 1e-4
    assert not torch.equal(w.detach(), before)


def test_guard_halts_after_repeated_skips():
    from qspinn_core import StepGuard
    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([w], lr=1.0)
    g = StepGuard(max_consecutive_skips=3)
    for _ in range(3):
        g.step(w.sum() * float("inf"), [w], opt)
    assert g.halt


# ── 02 ────────────────────────────────────────────────────────────────────────
def test_covariance_not_executed_when_disabled(B, tiny):
    model, xb, _ = tiny
    def boom(*a, **k):
        raise AssertionError("van_loan executed")
    model.van_loan.forward = boom
    model.cov_head.forward = boom
    model.use_cov = False
    _, _, _, _, L_cov, _ = model.heavy_path(xb)       # must not raise
    assert float(L_cov) == 0.0
    model.use_cov = True
    with pytest.raises(AssertionError):
        model.heavy_path(xb)


# ── 03 ────────────────────────────────────────────────────────────────────────
def test_stage_c_teacher_deterministic(B, tiny):
    from qspinn_core import stage_c_mode
    model, xb, dt = tiny
    model.train()
    a = model.heavy_path(xb)[0]
    b = model.heavy_path(xb)[0]
    assert not torch.allclose(a, b), "sanity: training mode should be stochastic"
    params = stage_c_mode(model, list(model.gate.gate_head.parameters()))
    model.heavy_frozen = True
    assert not model.training
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if n.split(".")[0] in ("amp", "ham", "head"))
    assert {id(p) for p in params} == {id(p) for p in model.gate.gate_head.parameters()}
    s = model.encode(xb)
    l1 = model(s, xb, dt, mode="dense")[0]
    l2 = model(s, xb, dt, mode="dense")[0]
    assert torch.equal(l1, l2)
    assert not l1.requires_grad          # teacher logits carry no graph


# ── 04 ────────────────────────────────────────────────────────────────────────
def test_encoder_batch_order_invariant_and_requires_fit(B):
    from qspinn_core import SpikeEncoder
    enc = SpikeEncoder(5)
    with pytest.raises(RuntimeError):
        enc(torch.randn(2, 4, 5))
    enc.fit(np.random.RandomState(0).randn(500, 5).astype(np.float32))
    x = torch.randn(32, 6, 5)
    full = enc(x)
    perm = torch.randperm(32)
    assert torch.equal(enc(x[perm]), full[perm])          # order/batch composition irrelevant
    assert torch.equal(enc(x[:5]), full[:5])
    assert enc.threshold.shape == (1, 1, 5) and bool(enc.fitted)


def test_encoder_fitted_on_train_only():
    from qspinn_core import SpikeEncoder
    tr = np.random.RandomState(1).randn(1000, 3).astype(np.float32)
    thr_a = SpikeEncoder(3).fit(tr).threshold.clone()
    # Fitting must not depend on anything but its argument
    thr_b = SpikeEncoder(3).fit(tr.copy()).threshold
    assert torch.equal(thr_a, thr_b)


def test_encoder_state_in_checkpoint(B, tiny):
    model, _, _ = tiny
    assert "encoder.threshold" in model.state_dict() and "encoder.fitted" in model.state_dict()


# ── 05 ────────────────────────────────────────────────────────────────────────
def test_exact_handover_bit_for_bit(B, tiny):
    from qspinn_core import route_exact
    cheap, heavy = torch.randn(10, 3), torch.randn(10, 3)
    assert torch.equal(route_exact(cheap, heavy, torch.zeros(10)), cheap)
    assert torch.equal(route_exact(cheap, heavy, torch.ones(10)), heavy)
    g = torch.tensor([0., 1.] * 5)
    out = route_exact(cheap, heavy, g)
    assert torch.equal(out[g == 1], heavy[g == 1]) and torch.equal(out[g == 0], cheap[g == 0])


def test_route_exact_gives_gate_gradient():
    from qspinn_core import route_exact
    cheap, heavy = torch.zeros(4, 3), torch.ones(4, 3)
    g = torch.tensor([0., 1., 0., 1.], requires_grad=True)
    route_exact(cheap, heavy, g).sum().backward()
    assert torch.all(g.grad != 0)


def test_model_learned_equals_selected_expert(B, tiny):
    model, xb, dt = tiny
    model.eval()
    with torch.no_grad():
        s = model.encode(xb)
        out, g, *_ = model(s, xb, dt, mode="learned")
        cheap = model(s, xb, dt, mode="cheap")[0]
        heavy = model(s, xb, dt, mode="dense")[0]
    sel = g > 0.5
    assert torch.equal(out[sel], heavy[sel]) and torch.equal(out[~sel], cheap[~sel])


# ── 06 ────────────────────────────────────────────────────────────────────────
def test_rate_meter_is_global_not_mean_of_means():
    from qspinn_core import RateMeter
    m = RateMeter()
    m.update(torch.ones(1))          # batch of 1, all routed
    m.update(torch.zeros(99))        # batch of 99, none routed
    assert m.rate == pytest.approx(0.01)       # old mean-of-means would say 0.5


def test_windows_include_last_and_respect_segments(B):
    from qspinn_core import window_starts
    n, L = 50, B.SEQ_LEN
    X = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    X = np.pad(X, ((0, 0), (0, 0)))
    y = np.arange(n, dtype=np.int64)
    r = np.zeros(n, dtype=np.float32)
    Xw, yw, rw, dt = B.make_windows_from_array(X, y, r)
    assert len(Xw) == n - L + 1 and yw[-1] == n - 1       # last valid window kept
    assert torch.all(dt == 1.0)                           # event-index dt, no random gaps
    seg = np.r_[np.zeros(25), np.ones(25)].astype(int)
    st = window_starts(n, L, seg)
    assert all(seg[i] == seg[i + L - 1] for i in st)
    purged = window_starts(n, L, seg, horizon=10)
    assert all(seg[i + L - 1 + 10] == seg[i] for i in purged)


def test_load_state_compat_drops_retired_key(B):
    from qspinn_core import load_state_compat
    m = B.QSPINN()
    sd = m.state_dict()
    sd["log_var_gate"] = torch.tensor(0.0)
    del sd["encoder.threshold"], sd["encoder.fitted"]
    load_state_compat(m, sd)                              # must not raise
    assert not hasattr(m, "log_var_gate")


# ── smoke: stages run, Stage C leaves heavy weights bit-identical ─────────────
def test_stage_c_does_not_modify_heavy_weights(B, tmp_path):
    from logger import ExperimentLogger
    torch.manual_seed(0)
    n = 80
    X = torch.randn(n + B.SEQ_LEN, B.N_FEATURES).numpy()
    y = torch.randint(0, 3, (n + B.SEQ_LEN,)).numpy()
    r = (torch.rand(n + B.SEQ_LEN) > 0.9).float().numpy()
    Xw, yw, rw, dt = B.make_windows_from_array(X, y, r)
    model = B.QSPINN()
    model.encoder.fit(X)
    log = ExperimentLogger(tmp_path / "t.log", experiment_name="t")
    before = {k: v.clone() for k, v in model.state_dict().items() if k.split(".")[0] in ("amp", "ham", "head")}
    B.stage_C(model, Xw, yw, rw, dt, Xw, yw, rw, dt, torch.device("cpu"), log,
              epochs=1, ckpt_path=tmp_path / "c.pt", batch_size=40)
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k
