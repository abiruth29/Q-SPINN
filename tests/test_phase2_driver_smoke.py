"""Smoke: the routing-study helpers run end-to-end on a tiny untrained model (no training/results)."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp_B_hybrid_fi2010"))
import routing as R


def test_collect_and_conditional_equals_dense_on_selected():
    spec = importlib.util.spec_from_file_location("rs", ROOT / "experiments/exp_B_hybrid_fi2010/routing_study.py")
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    T = rs.T
    torch.manual_seed(0)
    model = T.QSPINN(); X = torch.randn(16, T.SEQ_LEN, T.N_FEATURES)
    model.encoder.fit(X.reshape(-1, T.N_FEATURES).numpy()); model.eval()
    dt = torch.ones(16, T.SEQ_LEN)
    c, h, g, f = rs.collect(model, X, dt, torch.device("cpu"))
    assert c.shape == h.shape == (16, 3) and g.shape == (16,) and np.isfinite(f).all()
    sel = torch.zeros(16, dtype=torch.bool); sel[[1, 5]] = True
    out = R.conditional_infer(c, sel, lambda xb: model.heavy_path(xb)[0], X)
    assert torch.allclose(out["logits"][sel], h[sel], atol=1e-5)
    assert torch.equal(out["logits"][~sel], c[~sel])
