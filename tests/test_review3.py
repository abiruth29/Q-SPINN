"""Review 3: guard clips finite-entry overflow, outages are fail-closed."""
import numpy as np, pytest, torch, torch.nn as nn
from qspinn_core import StepGuard
from binance_align import align_on_grid, assert_no_long_outage
from test_phase1_binance_align import _frame


def _params(k=4, n=1):
    ps = [nn.Parameter(torch.zeros(n)) for _ in range(k)]
    return ps, torch.optim.SGD(ps, lr=1.0)


def _inject(ps, val, dtype):
    for p in ps: p.grad = torch.full_like(p, val, dtype=dtype)


def test_float32_overflowing_norm_is_clipped_not_zeroed():
    ps, opt = _params(4)
    g = StepGuard(max_grad_norm=1.0)
    loss = sum((p * 1e30).sum() for p in ps)          # grads exactly 1e30 (float32): naive float32 norm overflows
    assert g.step(loss, ps, opt)
    got = torch.cat([p.detach() for p in ps])
    assert torch.allclose(got, torch.full((4,), -0.5), atol=1e-3)   # clipped to unit norm, not zeroed
    assert g.n_steps == 1 and g.n_skipped == 0


def test_float64_overflow_case_is_clipped():
    ps = [nn.Parameter(torch.zeros(1, dtype=torch.float64)) for _ in range(4)]
    opt = torch.optim.SGD(ps, lr=1.0); g = StepGuard(max_grad_norm=1.0)
    loss = sum((p * 1e200).sum() for p in ps)
    assert g.step(loss, ps, opt)
    got = torch.cat([p.detach() for p in ps])
    assert torch.allclose(got, torch.full((4,), -0.5, dtype=torch.float64), atol=1e-6)


def test_nonfinite_entry_rejected_without_state_change():
    ps, opt = _params(2)
    g = StepGuard()
    before = [p.detach().clone() for p in ps]
    assert not g.step(sum(torch.sqrt((p ** 2).sum()) for p in ps), ps, opt)   # NaN grad at 0
    assert all(torch.equal(p.detach(), b) for p, b in zip(ps, before))
    assert len(opt.state) == 0 and g.n_skipped == 1


def test_short_outage_401s_raises():
    btc = _frame(np.arange(0, 20000), 100.0)
    sol = _frame(np.r_[np.arange(0, 9000), np.arange(9402, 20000)], 10.0)     # 402 s hole (> max_gap=300)
    frames, grid = align_on_grid([btc, sol], max_gap=300)
    with pytest.raises(RuntimeError, match="long data outage"):
        assert_no_long_outage(frames, grid)


def test_clean_series_passes():
    btc = _frame(np.arange(0, 2000), 100.0); sol = _frame(np.arange(0, 2000), 10.0)
    frames, grid = align_on_grid([btc, sol], max_gap=300)
    assert_no_long_outage(frames, grid)
