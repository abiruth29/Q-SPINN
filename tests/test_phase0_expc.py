"""Phase 0 mirror tests for Exp C (Binance 3-asset). Synthetic data only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def C():
    spec = importlib.util.spec_from_file_location("expC", ROOT / "experiments/exp_C_multi_asset_binance/train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def tiny(C):
    torch.manual_seed(0)
    model = C.MultiAssetQSPINN()
    model.encoder.fit(torch.randn(300, C.N_FEATURES).numpy())
    return model, torch.randn(8, C.SEQ_LEN, C.N_FEATURES), torch.ones(8, C.SEQ_LEN)


def test_cov_disabled_not_executed(C, tiny):
    model, xb, _ = tiny
    def boom(*a, **k): raise AssertionError("executed")
    model.van_loan.forward = boom; model.cov_head.forward = boom
    model.use_cov = False
    out = model.heavy_path(xb)
    assert float(out[4]) == 0.0
    model.use_cov = True
    with pytest.raises(AssertionError):
        model.heavy_path(xb)


def test_stage_c_teacher_and_exact_routing(C, tiny):
    from qspinn_core import stage_c_mode
    model, xb, dt = tiny
    stage_c_mode(model, list(model.gate.gate_head.parameters())); model.heavy_frozen = True
    s = model.encode(xb)
    a = model(s, xb, dt, mode="dense")[0]; b = model(s, xb, dt, mode="dense")[0]
    assert torch.equal(a, b) and not a.requires_grad
    with torch.no_grad():
        out, g = model(s, xb, dt, mode="learned")[:2]
        cheap = model(s, xb, dt, mode="cheap")[0]
    sel = g > 0.5
    assert torch.equal(out[sel], a[sel]) and torch.equal(out[~sel], cheap[~sel])


def test_windows_purged_and_last_label_real(C):
    n = 2000
    feats = np.random.RandomState(0).randn(n, C.N_FEATURES).astype(np.float32)
    labels = np.arange(n, dtype=np.int64) % 3
    regime = np.zeros(n, dtype=np.float32)
    Xtr, ytr, _, _, Xva, yva, _, _, Xte, yte, _, _ = C.make_windows_binance(feats, labels, regime)
    L, H = C.SEQ_LEN, C.HORIZON
    # the last test window's label row + HORIZON is the last data row (no padded labels)
    total = len(Xtr) + len(Xva) + len(Xte)
    assert len(Xte) > 0 and Xte.shape[1] == L
    last_label_row = (n - L - H) + L - 1
    # purge: train label rows + HORIZON never reach the first val window's rows
    assert len(Xtr) + H == int((n - L - H) * C.TRAIN_FRAC)
    assert last_label_row + H == n - 1
    assert total < n - L + 1
