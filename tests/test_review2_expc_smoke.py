"""Exp C main() end to end on tiny synthetic trades for BTC/ETH/SOL (1 epoch/stage). Exercises alignment, labels, baseline."""
import importlib, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]


def _trades(path, n_sec, rng, base):
    ts = np.repeat(np.arange(1_700_000_000, 1_700_000_000 + n_sec), 3)
    ts = ts[rng.random(len(ts)) > 0.15]                                  # ~15% of trades missing -> empty seconds
    n = len(ts)
    df = pd.DataFrame({"price": base + np.cumsum(rng.normal(0, .05, n)), "qty": rng.random(n) + .1,
                       "timestamp_us": ts * 1_000_000, "is_buyer_maker": rng.random(n) > .5})
    path.mkdir(parents=True); df.to_parquet(path / "part.parquet")


def test_expc_main_runs(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "experiments" / "exp_C_multi_asset_binance"))
    spec = importlib.util.spec_from_file_location("expC_main", ROOT / "experiments/exp_C_multi_asset_binance/train.py")
    C = importlib.util.module_from_spec(spec); sys.modules["expC_main"] = C; spec.loader.exec_module(C)
    rng = np.random.default_rng(0)
    for sym, base in (("BTCUSDT", 60000.), ("ETHUSDT", 3000.), ("SOLUSDT", 150.)):
        _trades(tmp_path / "trades" / f"symbol={sym}", 1500, rng, base)
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "trades"); monkeypatch.setattr(C, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(C, "CKPT_PATH", tmp_path / "out" / "ckpt.pt"); (tmp_path / "out").mkdir()
    res = C.main(epochs_a=1, epochs_b=1, epochs_c=1, max_ticks=5000, batch_size=64)
    assert 0 <= res["acc"] <= 1 and "loop_flux" in res
    assert "gauge-invariant" in (tmp_path / "out" / "RESULTS.md").read_text(encoding="utf-8")
