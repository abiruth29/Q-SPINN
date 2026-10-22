"""Review 4 item 3: gate trained on predictive benefit of the heavy expert, not on trailing volatility."""
import torch
import pytest
from qspinn_core import expert_benefit, benefit_gate_loss


def test_benefit_sign_and_target_independent_of_volatility():
    y = torch.tensor([0, 1])
    cheap = torch.tensor([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]])    # cheap wrong on row0, wrong on row1
    heavy = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])    # heavy right on row0 (helps), row1 right too? y=1 -> cheap right
    b = expert_benefit(cheap, heavy, y)
    assert b[0] > 1 and abs(float(b[1])) < 1e-3


def test_loss_pushes_gate_up_where_heavy_helps_and_down_otherwise():
    gl = torch.zeros(2, requires_grad=True)
    benefit = torch.tensor([2.0, -2.0])
    benefit_gate_loss(gl, benefit).backward()
    assert gl.grad[0] < 0 and gl.grad[1] > 0       # descent raises row0's logit, lowers row1's


def test_calm_row_with_benefit_is_positive_even_if_regime_flag_zero():
    gl = torch.zeros(1, requires_grad=True)
    benefit_gate_loss(gl, torch.tensor([1.5])).backward()   # no regime input exists at all
    assert gl.grad[0] < 0


def test_stage_c_benefit_objective_runs_and_keeps_heavy_frozen(tmp_path):
    import exp_B_hybrid_fi2010.train as B
    from logger import ExperimentLogger
    torch.manual_seed(0); n = 60
    X = torch.randn(n + B.SEQ_LEN, B.N_FEATURES).numpy(); y = torch.randint(0, 3, (n + B.SEQ_LEN,)).numpy()
    r = (torch.rand(n + B.SEQ_LEN) > 0.9).float().numpy()
    Xw, yw, rw, dt = B.make_windows_from_array(X, y, r)
    model = B.QSPINN(); model.encoder.fit(X)
    log = ExperimentLogger(tmp_path / "t.log", experiment_name="t")
    before = {k: v.clone() for k, v in model.state_dict().items() if k.split(".")[0] in ("amp", "ham", "head")}
    B.stage_C(model, Xw, yw, rw, dt, Xw, yw, rw, dt, torch.device("cpu"), log, epochs=1,
              ckpt_path=tmp_path / "c.pt", batch_size=30, gate_objective="benefit")
    for k, v in before.items():
        assert torch.equal(v, model.state_dict()[k]), k
