"""Teammate run: a smoke/test checkpoint must never be resumed by a real run; tests must not touch real outputs."""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
from checkpoint import save_checkpoint, load_checkpoint, fingerprint_of, set_run_fingerprint
import checkpoint as CK

ROOT = Path(__file__).resolve().parents[1]


def _ck(): return {"epoch": 1, "stage": "C", "model": {}, "best_state": {}, "optimizer": {}}


def test_checkpoint_resumes_only_on_matching_fingerprint(tmp_path):
    p = tmp_path / "c.pt"
    set_run_fingerprint(fingerprint_of(exp="C", epochs=[1, 1, 1]))
    save_checkpoint(_ck(), p)
    assert load_checkpoint(p) is not None                                       # same run -> resume
    set_run_fingerprint(fingerprint_of(exp="C", epochs=[50, 50, 50]))           # real run after a smoke run
    assert load_checkpoint(p) is None                                           # refused
    set_run_fingerprint("unset")


def test_unfingerprinted_legacy_checkpoint_is_refused(tmp_path):
    p = tmp_path / "c.pt"
    torch.save(dict(_ck(), protocol=CK.PROTOCOL_VERSION), p)                    # no fingerprint field
    set_run_fingerprint(fingerprint_of(exp="C"))
    assert load_checkpoint(p) is None
    set_run_fingerprint("unset")


def test_stage_default_checkpoint_path_resolves_at_call_time(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("expC_ckpt", ROOT / "experiments/exp_C_multi_asset_binance/train.py")
    C = importlib.util.module_from_spec(spec); sys.modules["expC_ckpt"] = C; spec.loader.exec_module(C)
    import inspect
    for fn in (C.stage_A, C.stage_B, C.stage_C):
        assert inspect.signature(fn).parameters["ckpt_path"].default is None    # not bound to the real path at import


def test_expc_smoke_run_leaves_real_output_folder_untouched(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    real = ROOT / "experiments" / "exp_C_multi_asset_binance"
    before = sorted(p.name for p in real.iterdir() if p.suffix in (".pt", ".md", ".log"))
    from test_review2_expc_smoke import test_expc_main_runs
    test_expc_main_runs(tmp_path, monkeypatch)
    after = sorted(p.name for p in real.iterdir() if p.suffix in (".pt", ".md", ".log"))
    assert before == after


def test_past_return_lookup_baseline_catches_mean_reversion():
    from baselines import past_return_lookup
    g = torch.Generator().manual_seed(0)
    n, T = 6000, 20
    X = torch.randn(n, T, 3, generator=g)
    s = X[:, -5:, 0].sum(1)
    y = torch.where(s > 0.6, 0, torch.where(s < -0.6, 2, 1))        # perfectly mean-reverting label
    acc, f1, table = past_return_lookup(X[:4000], y[:4000], X[4000:], y[4000:])
    assert acc > 0.8 and table[0] == 2 and table[-1] == 0


def test_suite_does_not_blame_jobs_it_did_not_run(tmp_path, monkeypatch):
    import run_unattended_suite as S
    monkeypatch.setattr(S, "STATE_FILE", tmp_path / "state.json", raising=False)
    monkeypatch.setattr(S, "LOG_FILE", tmp_path / "suite.log", raising=False)
    monkeypatch.setattr(S, "MASTER_MD", tmp_path / "m.md", raising=False)
    monkeypatch.setattr(S, "set_suite_files", lambda t: None)
    monkeypatch.setattr(S, "ATTEMPTED", set())
    (tmp_path / "state.json").write_text('{"jobs": {"fi2010_k10_s0": {"status": "FAILED", "error": "old"}}}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["suite", "--only-binance"])
    logs = []; monkeypatch.setattr(S, "log", lambda m: logs.append(str(m)))
    monkeypatch.setattr(S, "run_job", lambda jid, *a, **k: S.ATTEMPTED.add(jid))
    C = importlib.import_module("exp_C_multi_asset_binance.train")
    rc = S.main()
    text = "\n".join(logs)
    assert rc == 0                                                  # the old failure belongs to a job this run did not execute
    assert "Earlier FAILED job(s) NOT run this time: fi2010_k10_s0" in text


def test_completed_job_with_missing_or_different_signature_is_rerun_not_skipped(tmp_path, monkeypatch):
    import json, run_unattended_suite as S
    from qspinn_core import PROTOCOL_VERSION
    monkeypatch.setattr(S, "STATE_FILE", tmp_path / "state.json", raising=False)
    monkeypatch.setattr(S, "LOG_FILE", tmp_path / "suite.log", raising=False)
    monkeypatch.setattr(S, "MASTER_MD", tmp_path / "m.md", raising=False)
    monkeypatch.setattr(S, "ATTEMPTED", set()); monkeypatch.setattr(S, "SKIPPED", set())
    calls = []
    # an old COMPLETED record (e.g. the invalid first smoke run) has no signature -> must re-run
    (tmp_path / "state.json").write_text(json.dumps({"jobs": {"j": {"status": "COMPLETED", "protocol": PROTOCOL_VERSION,
                                                                    "metrics": {"x": 1}}}}), encoding="utf-8")
    S.run_job("j", "d", lambda **k: calls.append(1) or {"x": 2}, epochs_a=1)
    assert calls == [1] and "j" not in S.SKIPPED
    S.run_job("j", "d", lambda **k: calls.append(2) or {"x": 3}, epochs_a=1)       # identical config -> skipped
    assert calls == [1] and "j" in S.SKIPPED
    S.run_job("j", "d", lambda **k: calls.append(3) or {"x": 4}, epochs_a=50)      # different config -> re-run
    assert calls == [1, 3]


def test_release_check_does_not_write_tracked_files():
    t = (ROOT / "scripts" / "release_check.sh").read_text(encoding="utf-8")
    assert "release_logs/" in t and "docs/TEST_LOG_latest.txt" not in t
