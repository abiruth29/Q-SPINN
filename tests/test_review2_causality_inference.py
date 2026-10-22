"""Review-2 ranks 11-12: residual generator is prefix-only (through heavy_path/ham); real conditional API."""
import importlib.util, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "experiments" / "exp_B_hybrid_fi2010"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


B = _load("expB_train", "experiments/exp_B_hybrid_fi2010/train.py")
C = _load("expC_train", "experiments/exp_C_multi_asset_binance/train.py")


class _ConstGate(torch.nn.Module):
    def __init__(self, real, c):
        super().__init__(); self.real, self.c = real, c
    def forward(self, s, d):
        return self.real(s, d)[0], torch.full((len(s),), self.c), torch.zeros(()), torch.zeros(len(s))


def _models():
    torch.manual_seed(0)
    out = []
    for M, mk in ((B, lambda: B.QSPINN()), (C, lambda: C.MultiAssetQSPINN(n_in=C.N_FEATURES, d=C.PSI_DIM, n_assets=C.N_ASSETS))):
        m = mk().eval(); n_in = M.N_FEATURES
        X = torch.randn(6, M.SEQ_LEN, n_in)
        if hasattr(m, "encoder"):
            m.encoder.fit(X.reshape(-1, n_in).numpy())
        out.append((M, m, X))
    return out


def test_residual_generator_ignores_transition_target():
    for M, m, X in _models():
        v = X[:, 1:] - X[:, :-1]
        (Hf, Sf, *_), (Hp, Sp, *_) = m.ham(v, with_prefix=True)
        X2 = X.clone(); X2[:, -1] += 3.0                          # change ONLY the observed transition target x[T-1]
        v2 = X2[:, 1:] - X2[:, :-1]
        (Hf2, Sf2, *_), (Hp2, Sp2, *_) = m.ham(v2, with_prefix=True)
        assert torch.allclose(Hp, Hp2, atol=1e-6) and torch.allclose(Sp, Sp2, atol=1e-6)   # residual generator fixed
        assert not torch.allclose(Hf, Hf2, atol=1e-4)                                      # forecast generator sees it
        # ... and the reconstruction error through the real heavy_path DOES respond to the target
        L1 = m.heavy_path(X)[1]; L2 = m.heavy_path(X2)[1]
        assert not torch.allclose(L1, L2)


def test_conditional_api_zero_and_all_selected():
    for M, m, X in _models():
        n = len(X); dt = torch.ones(n, M.SEQ_LEN)
        sp = m.encode(X)
        calls = []
        orig = m.heavy_infer
        m.heavy_infer = lambda xb: (calls.append(len(xb)), orig(xb))[1]   # plain function attr is fine
        real_gate = m.gate
        for const, tag in ((-1.0, "none"), (1.0, "all")):
            m.gate = _ConstGate(real_gate, const)
            calls.clear()
            out, sel, rows = m.infer(sp, X, dt)
            if tag == "none":
                assert calls == [] and rows == 0 and torch.equal(out, real_gate(sp, dt)[0])
            else:
                assert rows == n
                dense = m.heavy_path(X)[0]
                assert torch.allclose(out, dense, atol=1e-5)               # selected rows == dense heavy prediction
        m.gate = real_gate; m.heavy_infer = orig


def test_inference_path_skips_loss_only_work():
    for M, m, X in _models():
        called = []
        m.van_loan.forward = lambda *a, **k: called.append("vl")           # must never run on the inference path
        m.heavy_infer(X)
        assert called == []
