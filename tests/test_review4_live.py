"""Review 4 item 4: live inference only runs the heavy path when routed; omega status is truthful."""
import numpy as np, torch
import importlib
from live_contract import OmegaState, live_step

B = importlib.import_module("exp_B_hybrid_fi2010.train")


def _model():
    m = B.QSPINN(); m.encoder.fit(np.random.default_rng(0).normal(size=(200, B.N_FEATURES)).astype(np.float32)); return m.eval()


def _x(): return torch.randn(1, B.SEQ_LEN, B.N_FEATURES), torch.ones(1, B.SEQ_LEN)


class Count:
    def __init__(self, m): self.n = 0; self.orig = m.heavy_path; m.heavy_path = self
    def __call__(self, *a, **k): self.n += 1; return self.orig(*a, **k)


def _force_gate(m, on):
    with torch.no_grad():
        m.gate.gate_head.fc.weight.zero_(); m.gate.gate_head.fc.bias.fill_(8.0 if on else -8.0)


def test_gated_no_routed_rows_means_zero_heavy_calls_and_omega_unavailable():
    m = _model(); _force_gate(m, False); c = Count(m); st = OmegaState(); x, dt = _x()
    r = live_step(m, x, dt, st, "gated")
    assert c.n == 0 and r["heavy_calls"] == 0 and r["omega"] is None and r["omega_status"] == "unavailable"


def test_gated_routed_is_fresh_then_skipped_is_cached_with_age():
    m = _model(); st = OmegaState(); x, dt = _x()
    _force_gate(m, True); r1 = live_step(m, x, dt, st, "gated")
    assert r1["heavy_calls"] == 1 and r1["omega_status"] == "fresh" and r1["omega_age_ticks"] == 0
    _force_gate(m, False); r2 = live_step(m, x, dt, st, "gated"); r3 = live_step(m, x, dt, st, "gated")
    assert r2["omega_status"] == "cached" and r2["omega"] == r1["omega"] and r3["omega_age_ticks"] == 2


def test_always_mode_runs_heavy_every_tick():
    m = _model(); _force_gate(m, False); c = Count(m); st = OmegaState(); x, dt = _x()
    for _ in range(3): r = live_step(m, x, dt, st, "always")
    assert c.n == 3 and r["omega_status"] == "fresh" and r["omega_mode"] == "always" and r["latency_ms"] > 0
