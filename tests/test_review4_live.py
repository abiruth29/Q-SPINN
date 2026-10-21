"""Review 4 item 4: live inference only runs the heavy path when routed; omega status is truthful."""
import numpy as np, torch
import importlib
from live_contract import OmegaState, live_step

B = importlib.import_module("exp_B_hybrid_fi2010.train")


def _model():
    m = B.QSPINN(); m.encoder.fit(np.random.default_rng(0).normal(size=(200, B.N_FEATURES)).astype(np.float32)); return m.eval()


def _x(): return torch.randn(1, B.SEQ_LEN, B.N_FEATURES), torch.ones(1, B.SEQ_LEN)


class Count:
    def __init__(self, m): self.n = 0; self.orig = m.heavy_infer; m.heavy_infer = self
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


def test_heavy_infer_matches_heavy_path_logits_and_omega():
    m = _model(); x, _ = _x(); x = torch.randn(4, B.SEQ_LEN, B.N_FEATURES)
    with torch.no_grad():
        heavy, _, _, S, _, _ = m.heavy_path(x)
        lg, om = m.heavy_infer(x, return_omega=True)
    assert torch.allclose(lg, heavy, atol=1e-5) and torch.allclose(om, torch.linalg.eigvalsh(S)[:, -1], atol=1e-5)
    assert torch.equal(m.heavy_infer(x), lg)


def test_fresh_summary_is_sample_based_not_time_integral():
    from live_contract import FreshOmegaSummary
    a = FreshOmegaSummary(600); b = FreshOmegaSummary(600)
    for v in (1.0, 3.0): a.add(v)                                  # two readings, one second apart
    for v in (1.0, 3.0): b.add(v)                                  # same two readings, an hour apart
    assert a.mean == b.mean == 2.0 and a.n == 2                    # depends on samples only; no dt anywhere
    assert FreshOmegaSummary().mean is None


def test_dashboard_and_pipeline_no_longer_claim_integral():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in ("pipeline.py", "dashboard.py"):
        t = (root / f).read_text(encoding="utf-8")
        assert "Omega_T" not in t and "Integrated Abscissa" not in t, f


# ── review 6 item 2: event-driven freshness + faithful log record ─────────────
def test_trade_only_ticks_never_trigger_inference():
    from live_contract import DepthFreshness
    f = DepthFreshness(0.5)
    f.mark_depth(100.0)
    assert f.take(100.6)[0] is True                       # the new depth is consumed once
    for t in (101.0, 102.0, 105.0):                       # consecutive ticks without new depth (trade-only / gap)
        ok, reason, age = f.take(t)
        assert ok is False and reason == "no_new_depth" and age is None
    assert f.skipped["no_new_depth"] == 3


def test_throttled_depth_stays_pending_and_reports_age():
    from live_contract import DepthFreshness
    f = DepthFreshness(0.5)
    f.mark_depth(10.0); assert f.take(10.1)[0]
    f.mark_depth(10.2)
    assert f.take(10.3) == (False, "throttled", None)     # too soon, but the update is not lost
    ok, reason, age = f.take(10.7)
    assert ok and reason == "ok" and abs(age - 0.5) < 1e-9


def test_event_gap_then_new_depth_reports_old_age_not_fresh():
    from live_contract import DepthFreshness
    f = DepthFreshness(0.5)
    f.mark_depth(0.0); assert f.take(0.6)[0]
    f.mark_depth(30.0)                                    # feed returns after a gap
    ok, _, age = f.take(31.0)
    assert ok and abs(age - 1.0) < 1e-9                   # age is relative to THIS depth update


def test_log_record_matches_live_step():
    from live_contract import FreshOmegaSummary, make_log_record
    m = _model(); _force_gate(m, True); st = OmegaState(); x, dt = _x(); summ = FreshOmegaSummary()
    r = live_step(m, x, dt, st, "gated"); summ.add(r["omega"])
    rec = make_log_record(r, now_str="00:00:00", mode="DEMO", input_age_s=0.2,
                          prices=dict(btc=1.0, eth=2.0, sol=3.0), summary=summ)
    assert rec["pred"] == r["pred"] and rec["omega"] == r["omega"] and rec["routed"] == r["routed"]
    assert rec["omega_status"] == "fresh" and rec["input_age_s"] == 0.2 and rec["n_fresh_readings"] == 1


def test_contract_includes_cadence_and_mismatch_is_detected():
    from live_contract import make_contract, contracts_match
    a = make_contract(20, 144, "x", "y", "crypto"); b = make_contract(20, 144, "x", "y", "crypto", cadence="new_depth_throttled_2hz")
    ok, bad = contracts_match(a, b)
    assert not ok and bad == ["cadence"]
