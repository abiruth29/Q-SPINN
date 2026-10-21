"""Teammate-run findings: Windows encodings, honest suite summary, actionable missing-data errors."""
import ast, json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _py_files():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).parts
        if rel[0] in ("scratch", "tests", ".git") or "legacy_v3.0" in rel or "__pycache__" in rel:
            continue
        yield p


def test_every_text_io_call_has_explicit_encoding():
    """Windows defaults to cp1252: text-mode open/read_text/write_text without encoding= breaks on UTF-8 content."""
    bad = []
    for p in list(_py_files()) + [q for q in (ROOT / "tests").glob("*.py")]:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            kws = {k.arg for k in n.keywords}
            if name in ("read_text", "write_text") and "encoding" not in kws:
                bad.append(f"{p.relative_to(ROOT)}:{n.lineno} {name}")
            if name == "open" and isinstance(f, (ast.Name, ast.Attribute)):
                args = n.args
                mode = None
                if isinstance(f, ast.Name) and len(args) >= 2 and isinstance(args[1], ast.Constant): mode = args[1].value
                if isinstance(f, ast.Attribute) and len(args) >= 1 and isinstance(args[0], ast.Constant): mode = args[0].value
                for k in n.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant): mode = k.value.value
                if (mode is None or "b" not in str(mode)) and "encoding" not in kws:
                    bad.append(f"{p.relative_to(ROOT)}:{n.lineno} open")
    assert not bad, bad


def test_suite_reports_failures_and_nonzero_exit(tmp_path, monkeypatch):
    import run_unattended_suite as S
    monkeypatch.setattr(S, "STATE_FILE", tmp_path / "state.json", raising=False)
    monkeypatch.setattr(S, "LOG_FILE", tmp_path / "suite.log", raising=False)
    monkeypatch.setattr(S, "MASTER_MD", tmp_path / "m.md", raising=False)
    monkeypatch.setattr(sys, "argv", ["suite", "--test", "--skip-binance"])
    monkeypatch.setattr(S, "set_suite_files", lambda t: None)
    import importlib
    B = importlib.import_module("exp_B_hybrid_fi2010.train")
    monkeypatch.setattr(B, "main", lambda **k: (_ for _ in ()).throw(FileNotFoundError("no data")))
    logs = []
    monkeypatch.setattr(S, "log", lambda m: logs.append(str(m)))
    rc = S.main()
    assert rc == 1
    text = "\n".join(logs)
    assert "SUITE FINISHED WITH 1 FAILED JOB" in text and "SUCCEEDED" not in text.split("FAILED JOB")[-1].split("Succeeded")[0]
    assert "COMPLETED UNINTERRUPTED" not in text
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["n_failed"] == 1 and not state.get("completed_at")


def test_missing_binance_data_gives_actionable_error(tmp_path):
    import importlib
    C = importlib.import_module("exp_C_multi_asset_binance.train")
    with pytest.raises(FileNotFoundError, match="procure_binance_data"):
        C.engineer_features_for_symbol(tmp_path / "symbol=BTCUSDT", 1000)


def test_missing_segment_manifest_is_a_clear_exit(tmp_path):
    sys.path.insert(0, str(ROOT))
    import procure_fi2010 as P
    with pytest.raises(SystemExit) as e:
        P.main(setup2=True, segments=str(tmp_path / "nope.json"))
    assert "NOT shipped" in str(e.value) and "1705.03233" in str(e.value)
