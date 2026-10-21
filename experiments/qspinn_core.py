"""
experiments/qspinn_core.py
==========================
Shared, dependency-light building blocks for the v3.1 corrections.
Used by both Exp B (FI-2010) and Exp C (Binance) so that a fix is made once.

Contents (task numbers refer to docs/FIX_PLAN.md / QSPINN_Required_Changes_In_Order.pdf)
  01  StepGuard         - reject non-finite loss/gradients BEFORE optimizer.step(), then clip
  03  stage_c_mode      - heavy expert frozen, eval mode, no-grad (deterministic teacher)
  04  SpikeEncoder      - train-only per-feature thresholds, deterministic per sample
  05  route_exact       - binary gate picks one expert's logits exactly (never attenuates)
  06  RateMeter         - global (sample-weighted) gate-rate accounting
  06  load_state_compat - drop retired parameters when loading older checkpoints
  06  window_starts     - sliding windows incl. the last valid one, optional segment safety
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


# Bump whenever model/data/protocol semantics change.  Run state, checkpoints, weights and live
# contracts carry this string and are REJECTED on mismatch (review item: stale COMPLETED jobs).
PROTOCOL_VERSION = "v3.1.1-setup2-commonclock"


# ──────────────────────────────────────────────────────────────────────────────
# 01  Pre-update numerical safety
# ──────────────────────────────────────────────────────────────────────────────
class StepGuard:
    """
    Safe optimizer step.

        guard = StepGuard(max_grad_norm=1.0, max_consecutive_skips=25)
        ok = guard.step(loss, params, opt)

    * Non-finite loss            -> no backward, no step, parameters untouched.
    * Non-finite gradient        -> no step, gradients zeroed, parameters untouched.
    * Otherwise                  -> clip_grad_norm_ then opt.step().

    `halt` becomes True after `max_consecutive_skips` skipped steps in a row so a
    genuinely diverged run still stops instead of silently doing nothing.
    """

    def __init__(self, max_grad_norm: float = 1.0, max_consecutive_skips: int = 25):
        self.max_grad_norm = max_grad_norm
        self.max_consecutive_skips = max_consecutive_skips
        self.n_steps = 0
        self.n_skipped = 0
        self.consecutive = 0
        self.last_grad_norm = float("nan")

    @property
    def halt(self) -> bool:
        return self.consecutive >= self.max_consecutive_skips

    def _skip(self, opt):
        opt.zero_grad(set_to_none=True)
        self.n_skipped += 1
        self.consecutive += 1
        return False

    def step(self, loss: torch.Tensor, params: Iterable[torch.nn.Parameter], opt) -> bool:
        params = [p for p in params if p.requires_grad]
        if not torch.isfinite(loss):
            return self._skip(opt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            return self._skip(opt)
        # aggregate norm can overflow to inf even when every entry is finite -> reject BEFORE the step
        gn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.double()) for g in grads])) if grads \
            else torch.zeros(())
        if not torch.isfinite(gn):
            return self._skip(opt)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.last_grad_norm = float(gn)
        opt.step()
        self.n_steps += 1
        self.consecutive = 0
        return True

    def summary(self) -> dict:
        return {"steps": self.n_steps, "skipped": self.n_skipped}


# ──────────────────────────────────────────────────────────────────────────────
# 03  Deterministic Stage-C teacher
# ──────────────────────────────────────────────────────────────────────────────
HEAVY_MODULES = ("amp", "ham", "head", "van_loan", "cov_head")


def stage_c_mode(model: nn.Module, trainable_gate_params: Iterable[nn.Parameter]):
    """
    Put `model` in the Stage-C configuration and return the gate params to optimise.

    * every parameter frozen except `trainable_gate_params`
    * whole model in eval() (no HamiltonianNet sampling noise, deterministic LIF dropout-free)
    * the SNN gate remains differentiable (its forward is NOT wrapped in no_grad)
    The heavy expert itself must be called under `torch.no_grad()` by the caller
    (see `heavy_teacher_ctx`) so its logits are a fixed teacher signal.
    """
    keep = {id(p) for p in trainable_gate_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    model.eval()
    return [p for p in model.parameters() if id(p) in keep]


def heavy_teacher_ctx():
    """Context manager for the heavy expert call in Stage C."""
    return torch.no_grad()


# ──────────────────────────────────────────────────────────────────────────────
# 04  Calibrated, deterministic spike encoder
# ──────────────────────────────────────────────────────────────────────────────
class SpikeEncoder(nn.Module):
    """
    Delta (change) encoder with *train-only* per-feature thresholds.

    Replaces the old `delta_encode`, whose threshold was the batch mean |delta|
    - so a sample's spikes depended on which other samples shared its batch and
    differed between train and test.  Here thresholds are fitted once on the
    training array and stored as a buffer (hence saved in every checkpoint).
    """

    def __init__(self, n_features: int, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("threshold", torch.zeros(1, 1, n_features))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, X: np.ndarray | torch.Tensor, valid_pair: Optional[np.ndarray] = None) -> "SpikeEncoder":
        """
        X: [N, F] time-ordered TRAIN array.  valid_pair[i]=True if rows i and i+1 lie
        in the same stock/day segment (so a boundary jump is not counted as a delta).
        """
        Xt = torch.as_tensor(np.asarray(X) if not torch.is_tensor(X) else X, dtype=torch.float32)
        d = (Xt[1:] - Xt[:-1]).abs()
        if valid_pair is not None:
            m = torch.as_tensor(valid_pair[: len(d)], dtype=torch.bool)
            d = d[m]
        if len(d) == 0:
            raise ValueError("SpikeEncoder.fit: no valid delta pairs")
        self.threshold.copy_(d.mean(dim=0).view(1, 1, -1) * self.alpha)
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("SpikeEncoder used before fit(); fit on the TRAIN split first.")
        d = torch.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        thr = self.threshold.to(x.dtype)
        return torch.cat([(d > thr).float(), (d < -thr).float()], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# 05  Exact expert hand-over
# ──────────────────────────────────────────────────────────────────────────────
def route_exact(cheap: torch.Tensor, heavy: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Binary routing: g==1 -> heavy logits, g==0 -> cheap logits, *bit-for-bit*.

    Forward value is `torch.where`; the extra `(g - g.detach())` term is exactly 0
    in the forward pass but gives the gate a straight-through gradient
    (d out/d g = heavy - cheap) whenever the caller keeps the graph.
    """
    sel = (g > 0.5).unsqueeze(-1)
    out = torch.where(sel, heavy, cheap)
    st = (g - g.detach()).unsqueeze(-1) * (heavy - cheap).detach()
    return out + st


# ──────────────────────────────────────────────────────────────────────────────
# 06  Bookkeeping
# ──────────────────────────────────────────────────────────────────────────────
class RateMeter:
    """Global rate = total routed / total samples  (not mean of per-batch means)."""

    def __init__(self):
        self.routed = 0.0
        self.total = 0

    def update(self, g: torch.Tensor):
        self.routed += float((g > 0.5).sum().item())
        self.total += int(g.numel())

    @property
    def rate(self) -> float:
        return self.routed / self.total if self.total else float("nan")


RETIRED_KEYS = ("log_var_gate",)


def load_state_compat(model: nn.Module, state: dict, log=None):
    """Load a state dict from an older checkpoint: drop retired keys, allow a missing encoder."""
    state = {k: v for k, v in state.items() if not k.endswith(RETIRED_KEYS)}
    res = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_missing} unexpected={res.unexpected_keys}")
    if log and any(k.startswith("encoder.") for k in res.missing_keys):
        log.warn("Legacy checkpoint has no SpikeEncoder thresholds; refit encoder on train data before use.")
    return res


def window_starts(n_rows: int, seq_len: int, seg_id: Optional[np.ndarray] = None,
                  horizon: int = 0) -> np.ndarray:
    """
    Indices i such that rows [i, i+seq_len) form a valid window.

    * includes the last window: there are n_rows - seq_len + 1 of them (old code lost one)
    * if seg_id (per-row stock/day id) is given, a window may not straddle two segments
    * `horizon` rows at the end of each segment are dropped (label looks `horizon` ahead
      and must not cross into the next segment) - purge for tasks 07/11.
    """
    if n_rows < seq_len:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(n_rows - seq_len + 1, dtype=np.int64)
    if seg_id is None:
        return starts
    seg_id = np.asarray(seg_id)
    first, last = seg_id[starts], seg_id[starts + seq_len - 1]
    ok = first == last
    if horizon:
        # last row index of the segment that window i belongs to
        ends = np.r_[np.flatnonzero(seg_id[1:] != seg_id[:-1]), n_rows - 1]
        seg_end = ends[np.searchsorted(ends, starts + seq_len - 1)]
        ok &= (starts + seq_len - 1 + horizon) <= seg_end
    return starts[ok]


# ──────────────────────────────────────────────────────────────────────────────
# 08-10  Single stable generator A = S - iH for the whole heavy path
# ──────────────────────────────────────────────────────────────────────────────
GENERATOR_VERSION = 2   # v1 = S tied to Re(H) and exp(-iH dt); old checkpoints are LEGACY


def bound_fro(M: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Spectral control: rescale so the Frobenius norm (>= spectral norm) is <= max_norm."""
    fro = torch.linalg.matrix_norm(M, ord="fro").clamp_min(1e-12)
    scale = (max_norm / fro).clamp(max=1.0)
    return M * scale.view(-1, *([1] * (M.dim() - 1))).to(M.dtype)


def hermitian_from_raw(h_re: torch.Tensor, h_im: torch.Tensor) -> torch.Tensor:
    M = torch.complex(h_re, h_im)
    return 0.5 * (M + M.conj().transpose(-1, -2))


def symmetric_from_raw(s_raw: torch.Tensor) -> torch.Tensor:
    return 0.5 * (s_raw + s_raw.transpose(-1, -2))


def propagate(psi: torch.Tensor, H: torch.Tensor, S: torch.Tensor, dt: float = 1.0,
              residual: str = "last", H_res: Optional[torch.Tensor] = None,
              S_res: Optional[torch.Tensor] = None):
    """
    One generator for the heavy path:  dpsi/dt = A psi,  A = S - iH,  U = exp(A dt).

    psi : [B, T, N] complex, UNNORMALISED dynamical state (normalise copies for readout).
    H   : [B, N, N] Hermitian,  S : [B, N, N] real symmetric.
    residual="last": regularise only the final pair (psi[T-2] -> psi[T-1]).  If (H_res, S_res) are
                     given they MUST be built from the prefix ending at T-2 (the transition target
                     x[T-1] must not influence the generator that is tested against it); the full-window
                     (H, S) is then used only to forecast beyond the window (returned U).
    residual="all" : legacy behaviour (end-of-window generator applied to every earlier pair,
                     which is NON-causal) - kept only as an ablation.

    Returns U, L_pde, L_norm.
      L_pde  : |psi_{t+1} - U psi_t|^2
      L_norm : finite-step norm balance, relative growth
               (|psi_{t+1}|^2 - |psi_t|^2)/|psi_t|^2  vs  psi_t^H (U^H U - I) psi_t / |psi_t|^2
               (exact for the same U; for S=0, U is unitary and the target growth is 0).
    """
    A = S.to(torch.complex64) - 1j * H
    U = torch.matrix_exp(A * dt)
    if H_res is not None:
        U_res = torch.matrix_exp((S_res.to(torch.complex64) - 1j * H_res) * dt)
    else:
        U_res = U
    if residual == "last":
        p0, p1 = psi[:, -2:-1], psi[:, -1:]
    elif residual == "all":
        p0, p1 = psi[:, :-1], psi[:, 1:]
    else:
        raise ValueError(f"residual must be 'last' or 'all', got {residual!r}")
    pred = torch.einsum("bij,btj->bti", U_res, p0)
    L_pde = (p1 - pred).abs().pow(2).mean()
    UhU = U_res.conj().transpose(-1, -2) @ U_res
    n0 = p0.abs().pow(2).sum(-1).clamp_min(1e-8)
    target = torch.einsum("bti,bij,btj->bt", p0.conj(), UhU, p0).real / n0 - 1.0
    obs = p1.abs().pow(2).sum(-1) / n0 - 1.0
    L_norm = (obs - target).pow(2).mean()
    return U, L_pde, L_norm
