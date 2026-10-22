"""Review-2 rank 15: shrinking the latent scale must not defeat the physics penalties; identification limit documented."""
import sys, torch
sys.path.insert(0, "tests")
from test_review2_causality_inference import _models
from qspinn_core import propagate, scale_anchor


def _objective(m, M, X, scale):
    psi_scale = scale
    orig = m.amp.forward
    m.amp.forward = lambda x: orig(x) * psi_scale
    try:
        logits, L_pde, H, S, L_cov, L_norm = m.heavy_path(X)[:6]
    finally:
        m.amp.forward = orig
    return logits, L_pde, L_norm                       # L_norm already includes the scale anchor


def test_shrinking_scale_does_not_reduce_total_physics_objective():
    for M, m, X in _models():
        l1, p1, n1 = _objective(m, M, X, 1.0)
        l2, p2, n2 = _objective(m, M, X, 1e-6)
        assert torch.allclose(l1, l2, atol=1e-4)                         # normalised read-out is blind to scale ...
        assert p2 > 0.5 * p1                                              # ... but the residual is now scale-invariant
        assert (p2 + n2) > (p1 + n1)                                      # ... and collapse is strictly penalised
        l3, p3, n3 = _objective(m, M, X, 1e-12)                          # far collapse also not profitable
        assert (p3 + n3) > (p1 + n1)


def test_anchor_zero_at_unit_mean_norm_and_free_per_step():
    psi = torch.randn(4, 6, 8, dtype=torch.complex64); psi = psi / psi.abs().pow(2).sum(-1, keepdim=True).sqrt()
    assert scale_anchor(psi) < 1e-8
    grow = psi.clone(); grow[:, 1::2] *= 1.3; grow[:, 0::2] *= (2 - 1.3 ** 2) ** 0.5   # per-step norms differ, mean ~1
    assert scale_anchor(grow) < 1e-2


def test_isotropic_growth_is_not_identified_by_classification():
    """exp((A+cI)dt)=exp(c dt)exp(A dt): a normalised read-out cannot see c.  omega shifts, logits do not."""
    for M, m, X in _models():
        (H, S, *_) = m.ham(X[:, 1:] - X[:, :-1])
        psi = m.amp(X)
        U0 = propagate(psi, H, S, 1.0)[0]
        U1 = propagate(psi, H, S + 0.3 * torch.eye(S.shape[-1]), 1.0)[0]
        n = lambda U: torch.nn.functional.normalize(torch.einsum("bij,bj->bi", U, psi[:, -1]), dim=-1)
        assert torch.allclose(n(U0).abs(), n(U1).abs(), atol=1e-5)
        w0 = torch.linalg.eigvalsh(S)[:, -1]; w1 = torch.linalg.eigvalsh(S + 0.3 * torch.eye(S.shape[-1]))[:, -1]
        assert torch.allclose(w1 - w0, torch.full_like(w0, 0.3), atol=1e-4)
