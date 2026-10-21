"""Review-2 ranks 1, 3, 10: run provenance and aggregate-norm guard."""
import json, importlib, sys
from pathlib import Path
import torch
from qspinn_core import StepGuard, PROTOCOL_VERSION
from checkpoint import save_checkpoint, load_checkpoint
from live_contract import make_contract, contracts_match


def test_stale_completed_job_is_retrained(tmp_path, monkeypatch):
    R = importlib.import_module("run_unattended_suite")
    monkeypatch.setattr(R, "STATE_FILE", tmp_path / "state.json"); monkeypatch.setattr(R, "MASTER_MD", tmp_path / "m.md")
    (tmp_path / "state.json").write_text(json.dumps({"jobs": {"fi2010_k10_s0": {
        "status": "COMPLETED", "metrics": {"macro_f1": 0.745664}}}}))      # legacy: no protocol field
    called = []
    out = R.run_job("fi2010_k10_s0", "x", lambda: called.append(1) or {"macro_f1": 0.5, "accuracy": .5, "gate_rate": .1,
                    "hermiticity": 0., "omega": 0., "crash_p": 0., "majority": .5})
    assert called and out["macro_f1"] == 0.5
    assert json.loads((tmp_path / "state.json").read_text())["jobs"]["fi2010_k10_s0"]["protocol"] == PROTOCOL_VERSION
    # same protocol -> skipped
    called.clear(); R.run_job("fi2010_k10_s0", "x", lambda: called.append(1)); assert not called


def test_master_report_ignores_legacy_jobs(tmp_path, monkeypatch):
    R = importlib.import_module("run_unattended_suite")
    monkeypatch.setattr(R, "MASTER_MD", tmp_path / "m.md")
    R.generate_master_report({"jobs": {"fi2010_k10_s0": {"status": "COMPLETED", "metrics": {
        "accuracy": .8, "macro_f1": .745664, "gate_rate": .1, "hermiticity": 0, "omega": 0, "crash_p": 0, "majority": .6}}}})
    assert "0.746" not in (tmp_path / "m.md").read_text()


def test_legacy_checkpoint_rejected(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"epoch": 3, "stage": "A", "model": {}, "best_state": {}}, p)       # no protocol
    assert load_checkpoint(p) is None
    save_checkpoint({"epoch": 1, "stage": "A", "model": {}, "best_state": {}}, tmp_path / "d.pt")
    assert load_checkpoint(tmp_path / "d.pt")["protocol"] == PROTOCOL_VERSION


def test_contract_requires_protocol():
    live = make_contract(20, 144, "a", "b", "c")
    old = {k: v for k, v in live.items() if k != "protocol"}
    assert not contracts_match(old, live)[0]


def test_overflowing_aggregate_norm_is_clipped_not_discarded():
    """Review 3: finite entries whose naive norm overflows must be clipped (and the step taken), not zeroed/skipped."""
    wd = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = torch.optim.SGD([wd], lr=1.0)
    loss = (wd * 1e200).sum()
    g = StepGuard(max_grad_norm=1.0)
    assert g.step(loss, [wd], opt) is True and g.n_skipped == 0
    assert torch.allclose(wd.detach(), torch.full((3,), -1 / 3 ** .5, dtype=torch.float64), atol=1e-6)
