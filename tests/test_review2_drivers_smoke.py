"""Both teammate-run drivers execute end to end on a tiny synthetic Setup-2 parquet with random weights (not a result)."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]
from test_phase3_main_smoke import _make_parquet


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def env(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    data = tmp_path / "fi2010"; data.mkdir()
    _make_parquet(data / "fi2010_setup2.parquet", np.random.default_rng(0))
    root = tmp_path / "root"; (root / "data" / "reference").mkdir(parents=True); (root / "data" / "reference" / "fi2010").symlink_to(data)
    exp_dir = ROOT / "experiments" / "exp_B_hybrid_fi2010"
    sys.path.insert(0, str(exp_dir)); sys.path.insert(0, str(ROOT / "experiments"))
    import importlib; T = importlib.import_module("train")
    monkeypatch.setattr(T, "ROOT", root)
    torch.manual_seed(0)
    model = T.QSPINN(); model.encoder.fit(np.random.default_rng(0).normal(size=(500, T.N_FEATURES)).astype(np.float32))  # trained weights always carry a fitted encoder
    w = tmp_path / "w.pt"; torch.save(model.state_dict(), w)
    return T, exp_dir, tmp_path, w


def test_routing_and_early_warning_drivers_run(env, monkeypatch):
    T, exp_dir, tmp, w = env
    for script, out in (("routing_study.py", "routing_study_k10_s0.json"), ("early_warning_study.py", "early_warning_k10_s0.json")):
        m = _load(exp_dir / script, "drv_" + script[:-3])
        monkeypatch.setattr(m, "HERE", tmp)
        monkeypatch.setattr(sys, "argv", [script, "--horizon", "10", "--seed", "0", "--weights", str(w)])
        m.main()
        rep = json.loads((tmp / out).read_text(encoding="utf-8"))
        assert rep
    r = json.loads((tmp / "routing_study_k10_s0.json").read_text(encoding="utf-8"))
    assert set(r["routers"]) == {"learned_gate", "confidence_cascade", "rolling_vol", "benefit_head", "random"}
    lat = r["latency"]
    assert lat["dense_heavy_s"] > 0 and lat["cheap_only_s"] > 0 and set(lat["hybrid_at_fraction_s"]) == {"0.0", "0.1", "0.25", "0.5", "1.0"}
    assert lat["hybrid_at_fraction_s"]["0.0"] < lat["hybrid_at_fraction_s"]["1.0"]      # skipping the heavy expert is cheaper
    e = json.loads((tmp / "early_warning_k10_s0.json").read_text(encoding="utf-8"))
    assert {"omega", "rolling_vol", "event_definition"} <= set(e) and "false_alarms" in e["omega"]
