"""End-to-end smoke of Exp B main() on a TINY synthetic Setup-2 parquet (1 epoch/stage). Not a result."""
import sys, importlib
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]


def _make_parquet(path, rng):
    rows = []
    def part(name, n_seg, seg_len, off):
        for s in range(n_seg):
            m = np.cumsum(rng.normal(0, .01, seg_len)) + 3 * s
            X = rng.normal(size=(seg_len, 144)).astype(np.float32); X[:, 0] = m; X[:, 2] = m
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(144)])
            for c in ("label_k10", "label_k20", "label_k30", "label_k50", "label_k100"):
                df[c] = rng.integers(0, 3, seg_len)
            df["partition"] = name; df["provenance"] = "authoritative"; df["segment_id"] = off + s; df["day_in_partition"] = s // 5
            rows.append(df)
        return off + n_seg
    o = part("train", 10, 120, 0)
    for nm in ("test_d8", "test_d9", "test_d10"):
        o = part(nm, 5, 120, o)
    pd.concat(rows, ignore_index=True).to_parquet(path)


def test_main_runs_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "experiments" / "exp_B_hybrid_fi2010"))
    T = importlib.import_module("train")
    data = tmp_path / "fi2010"; data.mkdir()
    _make_parquet(data / "fi2010_setup2.parquet", np.random.default_rng(0))
    monkeypatch.setattr(T, "ROOT", tmp_path / "root")
    (tmp_path / "root" / "data" / "reference").mkdir(parents=True); (tmp_path / "root" / "data" / "reference" / "fi2010").symlink_to(data)
    monkeypatch.setattr(T, "OUT_DIR", tmp_path / "out"); (tmp_path / "out").mkdir()
    res = T.main(horizon=10, seed=0, epochs_a=1, epochs_b=1, epochs_c=1, batch_size=64, out_dir=tmp_path / "out")
    assert 0.0 <= res["macro_f1"] <= 1.0 and len(res["f1_ci95"]) == 2
    assert (tmp_path / "out" / "qspinn_fi2010_k10_s0_contract.json").exists()
