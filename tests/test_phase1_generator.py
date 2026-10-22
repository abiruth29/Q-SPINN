"""Phase 1 model-side tests (tasks 08-10): independent S/H, one generator, causal residual."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def B():
    return _load("expB1", "experiments/exp_B_hybrid_fi2010/train.py")


@pytest.fixture(scope="module")
def C():
    return _load("expC1", "experiments/exp_C_multi_asset_binance/train.py")


def _rand_HS(Bn=4, N=8, seed=0):
    from qspinn_core import hermitian_from_raw, symmetric_from_raw
    g = torch.Generator().manual_seed(seed)
    H = hermitian_from_raw(torch.randn(Bn, N, N, generator=g), torch.randn(Bn, N, N, generator=g))
    S = symmetric_from_raw(torch.randn(Bn, N, N, generator=g))
    return H, S


# ── 08 ────────────────────────────────────────────────────────────────────────
def test_symmetry_is_exact():
    H, S = _rand_HS()
    assert torch.equal(S, S.transpose(-1, -2))
    assert torch.equal(H, H.conj().transpose(-1, -2))


@pytest.mark.parametrize("mod,kw", [("B", {}), ("C", {})])
def test_S_and_H_vary_independently(request, mod, kw):
    M = request.getfixturevalue(mod)
    torch.manual_seed(0)
    ham = M.HamiltonianNet().eval()
    v = torch.randn(5, M.SEQ_LEN - 1, M.N_FEATURES)
    with torch.no_grad():
        H0, S0 = ham(v)[:2]
        d2 = ham.d * ham.d
        ham.net[-1].bias[:d2].add_(1.0)                 # channel 0 -> S only (single-asset layout)
        H1, S1 = ham(v)[:2]
    assert not torch.allclose(S0, S1)
    if M.N_ASSETS == 1:
        assert torch.equal(H0, H1)                      # H untouched when only S's channel moves


def test_S_is_no_longer_Re_H(B):
    torch.manual_seed(0)
    ham = B.HamiltonianNet().eval()
    with torch.no_grad():
        H, S = ham(torch.randn(3, B.SEQ_LEN - 1, B.N_FEATURES))
    assert not torch.allclose(H.real, S, atol=1e-4)     # v1 had S == Re(H) exactly


# ── 09 ────────────────────────────────────────────────────────────────────────
def test_pure_conservative_flow_preserves_norm_and_gain_changes_it():
    from qspinn_core import propagate
    H, S = _rand_HS()
    psi = torch.randn(4, 6, 8, dtype=torch.cfloat) * 3.0          # deliberately not unit norm
    U, _, _ = propagate(psi, H, torch.zeros_like(S))
    n0 = psi[:, -1].abs().pow(2).sum(-1)
    n1 = torch.einsum("bij,bj->bi", U, psi[:, -1]).abs().pow(2).sum(-1)
    assert torch.allclose(n0, n1, rtol=1e-4)
    U2, _, _ = propagate(psi, H, 0.3 * S)
    n2 = torch.einsum("bij,bj->bi", U2, psi[:, -1]).abs().pow(2).sum(-1)
    assert not torch.allclose(n0, n2, rtol=1e-3)


def test_norm_balance_zero_for_exact_dynamics():
    from qspinn_core import propagate
    H, S = _rand_HS(seed=3)
    S = 0.2 * S
    psi0 = torch.randn(4, 1, 8, dtype=torch.cfloat)
    A = S.to(torch.cfloat) - 1j * H
    psi1 = torch.einsum("bij,btj->bti", torch.matrix_exp(A), psi0)
    _, L_pde, L_norm = propagate(torch.cat([psi0, psi1], 1), H, S)
    assert L_pde < 1e-9 and L_norm < 1e-9


def test_dynamical_psi_unnormalised_readout_normalised(B):
    torch.manual_seed(0)
    amp = B.ComplexAmplitudeLayer()
    psi = amp(torch.randn(3, B.SEQ_LEN, B.N_FEATURES) * 5)
    assert not torch.allclose(psi.abs().pow(2).sum(-1), torch.ones(3, B.SEQ_LEN), atol=1e-3)
    nrm = B.ComplexAmplitudeLayer.normalise(psi).abs().pow(2).sum(-1)
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5)


@pytest.mark.parametrize("mod", ["B", "C"])
def test_gain_loss_influences_prediction_and_losses_finite(request, mod):
    M = request.getfixturevalue(mod)
    torch.manual_seed(0)
    model = M.QSPINN() if mod == "B" else M.MultiAssetQSPINN()
    model.eval()
    x = torch.randn(6, M.SEQ_LEN, M.N_FEATURES)
    with torch.no_grad():
        out0 = model.heavy_path(x)
        model.ham.net[-1].bias[: model.ham.d ** 2].add_(2.0)     # move only S's channel
        out1 = model.heavy_path(x)
    assert not torch.allclose(out0[0], out1[0])                  # logits depend on S
    for t in (out0[1], out0[4], out0[5]):
        assert torch.isfinite(t)


def test_spectral_control_bounds_generator(B):
    torch.manual_seed(0)
    ham = B.HamiltonianNet().eval()
    with torch.no_grad():
        ham.net[-1].bias.add_(1e3)                               # absurd raw outputs
        H, S = ham(torch.randn(3, B.SEQ_LEN - 1, B.N_FEATURES))
    assert torch.linalg.matrix_norm(S, "fro").max() <= B.S_FRO_MAX + 1e-5
    assert torch.linalg.matrix_norm(H, "fro").max() <= B.H_FRO_MAX + 1e-5


# ── 10 ────────────────────────────────────────────────────────────────────────
def test_last_pair_residual_ignores_earlier_states_but_legacy_does_not():
    from qspinn_core import propagate
    H, S = _rand_HS(seed=5)
    psi = torch.randn(4, 6, 8, dtype=torch.cfloat)
    psi_mod = psi.clone(); psi_mod[:, :-2] += 10.0               # change only the older history
    for mode, same in (("last", True), ("all", False)):
        _, a, an = propagate(psi, H, S, residual=mode)
        _, b, bn = propagate(psi_mod, H, S, residual=mode)
        assert (torch.allclose(a, b) and torch.allclose(an, bn)) == same
