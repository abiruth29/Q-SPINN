"""
experiments/exp_B_hybrid_fi2010/train.py
=========================================
Experiment B — Full Q-SPINN Hybrid trained on FI-2010.

WHAT THIS DOES
--------------
1. Loads fi2010_noauction_zscore.parquet (2M timesteps, 144 features,
   already Z-score normalized — no re-fitting needed).
2. Splits chronologically by fold: folds 1-7 train, fold 8 val, fold 9 test.
3. Constructs a 2σ realized-volatility regime proxy mask (SWAP-15).
4. Runs three-stage training:
     Stage A: SNN gate (cheap head only)
     Stage B: Heavy path (complex amplitude + Hamiltonian + PINN head)
     Stage C: Gate head (arbitrate between the two branches)
5. Runs ablation: Q-SPINN vs SNN-only vs PINN-only vs LSTM.
6. Saves weights + auto-generates RESULTS.md.

SWAP STATUS
-----------
SWAP-01: ✅ Real FI-2010 LOB data (NoAuction Z-score)
SWAP-02: ✅ Pre-normalized by FI-2010 authors (no leakage)
SWAP-03: ✅ Train-only per-feature delta thresholds (v3.1: SpikeEncoder, no batch dependence)
SWAP-04: ✅ Arctan surrogate gradient with a FIXED slope (a hyperparameter; the slope is not trained)
SWAP-05: ✅ Custom LIF layer (learned membrane time constant, fixed surrogate slope)
SWAP-08: ✅ Kendall et al. homoscedastic adaptive loss weighting
SWAP-15: ✅ Statistically validated 2σ realized-volatility regime proxy mask
"""

import sys
import time
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))

from device import get_device
from logger import ExperimentLogger
from checkpoint import save_checkpoint, load_checkpoint, guard_nan
from qspinn_core import (propagate, scale_anchor, bound_fro, hermitian_from_raw, symmetric_from_raw, GENERATOR_VERSION,
                         StepGuard, SpikeEncoder, route_exact, RateMeter, load_state_compat,
                         stage_c_mode, heavy_teacher_ctx)

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── hyper-parameters ─────────────────────────────────────────────────────────
N_ASSETS      = 1           # 1 for FI-2010, 3 for Crypto Basket (Phase 3)
N_FEATURES    = 144         # Will be 432 for 3 assets
PSI_DIM       = 8           # Dimension per asset
HIDDEN        = 64
SEQ_LEN       = 20
HORIZON       = 10          # k=10 ticks — corresponds to label_k10
DT            = 1.0
LAMBDA_PDE    = 0.1
LAMBDA_GATE   = 0.02
LAMBDA_REGIME = 1.0
REGIME_SIGMA  = 2.0
REGIME_WINDOW = 200
BATCH_SIZE    = 2048
EPOCHS_A      = 100         # Stage A: SNN
EPOCHS_B      = 100         # Stage B: heavy path
EPOCHS_C      = 100         # Stage C: gate
LR            = 2e-3
SEED          = 0
LABEL_COL     = "label_k10"
CKPT_PATH     = OUT_DIR / "checkpoint.pt"
# Task 02: covariance (Van Loan + cov_head) is OFF until a defensible innovation
# likelihood exists.  Set QSPINN_USE_COV=1 to re-enable for experimentation only.
import os
USE_COVARIANCE = os.environ.get("QSPINN_USE_COV", "0") == "1"
# v3.1 generator settings (Tasks 08-10)
LAMBDA_SCALE = float(os.environ.get("QSPINN_LAMBDA_SCALE", "1.0"))   # weight of the scale anchor (0 disables = ablation)
RESIDUAL_MODE = os.environ.get("QSPINN_RESIDUAL", "last")   # 'last' (causal) | 'all' (legacy ablation)
H_FRO_MAX = 2.0     # spectral control on H
S_FRO_MAX = 1.0     # spectral control on S (growth per step <= e^1)
MAX_GRAD_NORM = 1.0          # Task 01


# ── surrogate gradient ────────────────────────────────────────────────────────
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, alpha):
        ctx.save_for_backward(v, alpha)
        return (v > 0).float()
    @staticmethod
    def backward(ctx, grad_out):
        v, alpha = ctx.saved_tensors
        return grad_out * (alpha / 2) / (1.0 + (math.pi / 2 * alpha * v) ** 2), None

spike_fn = SurrogateSpike.apply

# ── SNN layers ────────────────────────────────────────────────────────────────
class IrregularLIFLayer(nn.Module):
    def __init__(self, n_in, n_out, tau_m=10.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.tau_m = nn.Parameter(torch.tensor(tau_m))
        self.register_buffer("alpha", torch.tensor(2.0))   # FIXED surrogate slope (hyperparameter, not trained)

    def forward(self, x_seq, dt_seq):
        import torch.nn.functional as F
        B, T, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        out = []
        tau = F.softplus(self.tau_m) + 1e-3
        alpha_pos = F.softplus(self.alpha)
        for t in range(T):
            cur = self.fc(x_seq[:, t])
            # Eq 15.2: Irregular Membrane Leak
            beta_n = torch.exp(-dt_seq[:, t] / tau).unsqueeze(-1)
            # Eq 15.3: Membrane Update
            mem = beta_n * mem + cur
            spk = spike_fn(mem - 1.0, alpha_pos)
            mem = mem - spk  # Hard reset
            out.append(spk)
        return torch.stack(out, dim=1), mem


class LeakyReadout(nn.Module):
    def __init__(self, n_in, n_out, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.beta = beta
    def forward(self, s_seq):
        U = torch.zeros(s_seq.shape[0], self.fc.out_features, device=s_seq.device)
        for t in range(s_seq.shape[1]):
            U = self.beta * U + self.fc(s_seq[:, t])
        return U


# ── complex amplitude layer ───────────────────────────────────────────────────
class ComplexAmplitudeLayer(nn.Module):
    def __init__(self, n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS):
        super().__init__()
        self.d       = d * n_assets
        self.to_real = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, self.d))
        self.to_imag = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, self.d))

    @staticmethod
    def normalise(psi):
        n = psi.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        return psi / n.to(psi.dtype)

    def forward(self, x):
        # v3.1 (Task 09): dynamical state stays UNNORMALISED; readout normalises a copy.
        return torch.complex(self.to_real(x), self.to_imag(x))

    @staticmethod
    def density_features(psi):
        rho  = psi.unsqueeze(-1) * psi.conj().unsqueeze(-2)
        d    = rho.shape[-1]
        iu   = torch.triu_indices(d, d, offset=1)
        diag = rho.diagonal(dim1=-2, dim2=-1).real
        off  = rho[:, iu[0], iu[1]]
        return torch.cat([diag, off.real, off.imag], dim=-1)


# ── Hamiltonian net ───────────────────────────────────────────────────────────
class HamiltonianNet(nn.Module):
    def __init__(self, n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS):
        super().__init__()
        self.d = d
        self.n_assets = n_assets
        
        # Output: mu (2 * d * d per asset) + Graph Adjacency A (N*N) + Graph Flux Phi (N*N)
        out_dim = (n_assets * 3 * d * d) + (n_assets * n_assets) + (n_assets * n_assets)
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, out_dim))
        
        # Exponential-smoothing recursion on the generator coefficients (Kalman-*style* gain, NOT a state-space
        # filter: the coefficients are network outputs, not a hidden state with a validated observation model).
        self.theta = nn.Parameter(torch.full((n_assets,), 0.1))
        self.log_Q = nn.Parameter(torch.full((n_assets,), -3.0))
        self.log_R = nn.Parameter(torch.full((n_assets,), -2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, velocities, with_prefix=False):
        B, seq_len, _ = velocities.shape
        import torch.nn.functional as F
        
        Q = torch.exp(self.log_Q).view(1, self.n_assets, 1, 1, 1)
        R = torch.exp(self.log_R).view(1, self.n_assets, 1, 1, 1)
        theta_pos = F.softplus(self.theta).view(1, self.n_assets, 1, 1, 1) + 1e-6
        
        z_t = None
        P_t = torch.ones(B, self.n_assets, 1, 1, 1, device=velocities.device)
        
        for t in range(seq_len):
            v_t = velocities[:, t, :]
            out = self.net(v_t)
            
            mu_raw = out[:, :self.n_assets * 3 * self.d * self.d]
            mu_t  = mu_raw.view(B, self.n_assets, 3, self.d, self.d)
            
            if t >= seq_len - 2:          # last two steps: keep graph heads for the prefix and full generators
                idx_A = self.n_assets * 3 * self.d * self.d
                idx_Phi = idx_A + self.n_assets * self.n_assets
                A_t   = out[:, idx_A : idx_Phi].view(B, self.n_assets, self.n_assets)
                Phi_t = out[:, idx_Phi : ].view(B, self.n_assets, self.n_assets)
                if t == seq_len - 2: A_pre, Phi_pre = A_t, Phi_t
                if t == seq_len - 1: A_raw, Phi_raw = A_t, Phi_t
            
            if z_t is None:
                z_t = mu_t
                P_t = P_t * R
            else:
                z_pred = z_t + theta_pos * (mu_t - z_t)
                P_pred = P_t + Q
                K_t = P_pred / (P_pred + R)
                z_t = z_pred + K_t * (mu_t - z_pred)
                P_t = (1 - K_t) * P_pred
            if t == seq_len - 2:          # filtered state using observations up to (and incl.) step T-2 only
                z_pre, P_pre = z_t, P_t
        
        full = self._assemble(z_t, P_t, A_raw, Phi_raw, B, velocities.device)
        if not with_prefix:
            return full
        if seq_len >= 2:
            return full, self._assemble(z_pre, P_pre, A_pre, Phi_pre, B, velocities.device)
        zero = tuple(torch.zeros_like(t_) if torch.is_tensor(t_) else t_ for t_ in full)
        return full, zero      # 1 velocity step: no causal prefix -> identity residual generator


    def _assemble(self, z_t, P_t, A_raw, Phi_raw, B, device):
        import torch.nn.functional as F
        if self.training:
            epsilon = torch.randn_like(z_t)
            std = torch.sqrt(P_t + 1e-6)
            z_final = z_t + std * epsilon
        else:
            z_final = z_t
            
        # v3.1 (Task 08): S and H are INDEPENDENT.  channel 0 -> S (real symmetric, gain/loss),
        # channels 1,2 -> Re/Im of H (Hermitian, conservative).  Previously S was Re(H).
        S_local = symmetric_from_raw(z_final[:, :, 0])
        H_local = hermitian_from_raw(z_final[:, :, 1], z_final[:, :, 2])

        if self.n_assets == 1:
            H_tot = H_local[:, 0]
            S_tot = S_local[:, 0]
        else:
            # Magnetic Graph Laplacian Construction
            A_sym = F.softplus(A_raw + A_raw.transpose(-1, -2))
            Phi_anti = Phi_raw - Phi_raw.transpose(-1, -2)
            W = A_sym * torch.exp(1j * Phi_anti)
            
            D_diag = A_sym.sum(dim=-1)
            D = torch.diag_embed(D_diag)
            L = torch.complex(D, torch.zeros_like(D)) - W
            
            # Kronecker coupling: H_tot = BlockDiag(H_k) + L (x) I_d
            H_tot = torch.zeros(B, self.n_assets * self.d, self.n_assets * self.d, dtype=torch.complex64, device=device)
            S_tot = torch.zeros(B, self.n_assets * self.d, self.n_assets * self.d, dtype=torch.float32, device=device)
            I_d = torch.eye(self.d, dtype=torch.complex64, device=device)
            
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    si, ei = i * self.d, (i+1) * self.d
                    sj, ej = j * self.d, (j+1) * self.d
                    if i == j:
                        H_tot[:, si:ei, sj:ej] = H_local[:, i] + L[:, i, j].view(B, 1, 1) * I_d
                        S_tot[:, si:ei, sj:ej] = S_local[:, i]
                    else:
                        H_tot[:, si:ei, sj:ej] = L[:, i, j].view(B, 1, 1) * I_d
                        
        return (bound_fro(self.scale * H_tot, H_FRO_MAX), bound_fro(self.scale * S_tot, S_FRO_MAX))


# ── Van Loan Exact Covariance Propagator ─────────────────────────────────────
class VanLoanCovariancePropagator(nn.Module):
    """
    Implements Eq 22.4 and 22.5: Exact Continuous-to-Discrete Propagation.
    Constructs the 2N x 2N block matrix to exactly evaluate the covariance integral
    over interval dt.
    """
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        # \mathcal{B}_R: Latent OU Stochastic forcing matrix (Eq 22.1)
        self.B_R = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
    def forward(self, A_R, dt=0.1):
        """
        A_R: [B, N, N] real drift generator
        Returns F (transition matrix) and Q_h (exact discrete covariance)
        """
        B_dim, N, _ = A_R.shape
        BBT = self.B_R @ self.B_R.transpose(-1, -2)
        BBT_batch = BBT.unsqueeze(0).expand(B_dim, -1, -1)
        
        # Van Loan Block Matrix: M = [ -A_R   BBT ]
        #                            [   0    A_R^T ] * dt
        M = torch.zeros(B_dim, 2 * N, 2 * N, device=A_R.device, dtype=A_R.dtype)
        M[:, :N, :N] = -A_R
        M[:, :N, N:] = BBT_batch
        M[:, N:, N:] = A_R.transpose(-1, -2)
        M = M * dt
        
        E = torch.matrix_exp(M)
        
        E12 = E[:, :N, N:]
        E22 = E[:, N:, N:]
        
        F = E22.transpose(-1, -2)
        Q_h = F @ E12
        return F, Q_h


# ── SNN gate ─────────────────────────────────────────────────────────────────
class SNNGate(nn.Module):
    def __init__(self, n_in=N_FEATURES):
        super().__init__()
        self.lif1       = IrregularLIFLayer(2 * n_in, HIDDEN)
        self.lif2       = IrregularLIFLayer(HIDDEN, HIDDEN)
        self.cheap_head = LeakyReadout(HIDDEN, 3)
        self.gate_head  = LeakyReadout(HIDDEN, 1)
        self.register_buffer("alpha_gate", torch.tensor(2.0))   # FIXED surrogate slope
        nn.init.constant_(self.gate_head.fc.bias, 0.5)

    def forward(self, spikes, dt_seq):
        import torch.nn.functional as F
        s1, _ = self.lif1(spikes, dt_seq)
        s2, _ = self.lif2(s1, dt_seq)
        cheap_logits = self.cheap_head(s2)
        gate_logit   = torch.tanh(self.gate_head(s2).squeeze(-1) / 4.0)
        g            = spike_fn(gate_logit, F.softplus(self.alpha_gate))
        spike_rate   = 0.5 * (s1.mean() + s2.mean())
        return cheap_logits, g, spike_rate, gate_logit


# ── full Q-SPINN ──────────────────────────────────────────────────────────────
class QSPINN(nn.Module):
    def __init__(self, n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS, use_cov=None):
        super().__init__()
        self.use_cov = USE_COVARIANCE if use_cov is None else use_cov
        self.heavy_frozen = False      # Task 03: set by stage_c_mode
        self.encoder = SpikeEncoder(n_in)   # Task 04: fitted on TRAIN only
        self.gate = SNNGate(n_in=n_in)
        self.amp  = ComplexAmplitudeLayer(n_in=n_in, d=d, n_assets=n_assets)
        self.ham  = HamiltonianNet(n_in=n_in, d=d, n_assets=n_assets)
        
        total_psi_dim = d * n_assets
        n_rho     = 3 * (total_psi_dim + 2 * (total_psi_dim * (total_psi_dim - 1) // 2))
        self.head = nn.Sequential(nn.Linear(n_rho, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))
        
        # V3-02: Exact continuous-discrete covariance propagator
        self.van_loan = VanLoanCovariancePropagator(2 * total_psi_dim)
        
        # V3-04: Predictor for the state covariance matrix \Sigma_\theta
        n_cov = (2 * total_psi_dim) * (2 * total_psi_dim)
        self.cov_head = nn.Sequential(nn.Linear(total_psi_dim * 2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_cov))
        
        # SWAP-08: Adaptive loss weighting parameters
        self.log_var_ce     = nn.Parameter(torch.tensor(0.0))
        self.log_var_regime = nn.Parameter(torch.tensor(0.0))

    def heavy_path(self, x):
        # Pass the whole velocity sequence through the coefficient-smoothing recursion (see HamiltonianNet note)
        velocities = x[:, 1:] - x[:, :-1]
        (H_tot, S_tot), (H_res, S_res) = self.ham(velocities, with_prefix=True)   # prefix ends at T-2: causal residual
        
        B, N, _ = H_tot.shape
        
        # Realified Dynamics A_R (Eq 21.3)
        # A = S - i(H_R + i H_I) = (S + H_I) - i H_R
        H_R = H_tot.real
        H_I = H_tot.imag
        
        A_R_mat = S_tot + H_I
        A_I_mat = -H_R
        
        if self.use_cov:
            A_block = torch.zeros(B, 2 * N, 2 * N, device=x.device, dtype=torch.float32)
            A_block[:, :N, :N] = A_R_mat
            A_block[:, :N, N:] = -A_I_mat
            A_block[:, N:, :N] = A_I_mat
            A_block[:, N:, N:] = A_R_mat
            
            F, Q_h = self.van_loan(A_block, DT)

        # Task 09/10: ONE generator A = S - iH drives the means; residuals are causal ('last').
        psi = self.amp(x)     # unnormalised dynamical state
        U, L_pde, L_norm_ = propagate(psi, H_tot, S_tot, DT, RESIDUAL_MODE, H_res=H_res, S_res=S_res)

        if self.use_cov:
            z_T = torch.cat([psi[:, -1].real, psi[:, -1].imag], dim=-1)
            Sigma_theta = self.cov_head(z_T).view(B, 2 * N, 2 * N)
            Sigma_theta = Sigma_theta @ Sigma_theta.transpose(-1, -2)  # PSD
            R_Sigma = F @ Sigma_theta @ F.transpose(-1, -2) + Q_h - Sigma_theta
            L_cov = R_Sigma.abs().pow(2).mean()
        else:
            L_cov = torch.zeros((), device=x.device)

        L_norm = L_norm_ + LAMBDA_SCALE * scale_anchor(psi)   # norm balance + window-level non-collapse anchor

        psi_T    = ComplexAmplitudeLayer.normalise(psi[:, -1])
        psi_next = ComplexAmplitudeLayer.normalise(torch.einsum('bij,bj->bi', U, psi[:, -1]))
        feats    = torch.cat([
            ComplexAmplitudeLayer.density_features(psi_T),
            ComplexAmplitudeLayer.density_features(ComplexAmplitudeLayer.normalise(psi.mean(1))),
            ComplexAmplitudeLayer.density_features(psi_next)], dim=-1)
            
        return self.head(feats), L_pde, H_tot, S_tot, L_cov, L_norm

    @torch.no_grad()
    def heavy_infer(self, x):
        """Inference-only heavy expert: logits only - no residual generator, no PDE/norm/covariance work."""
        (H_tot, S_tot) = self.ham(x[:, 1:] - x[:, :-1])
        psi = self.amp(x)
        U = torch.matrix_exp((S_tot.to(torch.complex64) - 1j * H_tot) * DT)
        psi_T = ComplexAmplitudeLayer.normalise(psi[:, -1])
        psi_next = ComplexAmplitudeLayer.normalise(torch.einsum('bij,bj->bi', U, psi[:, -1]))
        feats = torch.cat([ComplexAmplitudeLayer.density_features(psi_T),
                           ComplexAmplitudeLayer.density_features(ComplexAmplitudeLayer.normalise(psi.mean(1))),
                           ComplexAmplitudeLayer.density_features(psi_next)], dim=-1)
        return self.head(feats)

    @torch.no_grad()
    def infer(self, spikes, x, dt_seq, chunk=4096, sel=None):
        """
        Deployed conditional inference: gate first, heavy expert ONLY on selected rows, scatter back.
        Returns (logits, selected_mask, heavy_rows).  Zero selected rows => zero heavy calls.
        """
        self.eval()
        cheap, g, _, _ = self.gate(spikes, dt_seq)
        sel = (g > 0.5) if sel is None else sel      # `sel` override is for latency sweeps at a fixed routing fraction
        out = cheap.clone()
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        for i in range(0, len(idx), chunk):
            j = idx[i:i + chunk]
            out[j] = self.heavy_infer(x[j])
        return out, sel, int(len(idx))

    def forward(self, spikes, x, dt_seq, mode="learned"):
        cheap, g, rate, gl = self.gate(spikes, dt_seq)
        if mode == "cheap":
            H0 = torch.zeros(1, PSI_DIM, PSI_DIM, dtype=torch.cfloat, device=spikes.device)
            S0 = torch.zeros(1, PSI_DIM, PSI_DIM, dtype=torch.float32, device=spikes.device)
            return cheap, torch.zeros_like(g), rate, torch.zeros((), device=spikes.device), H0, S0, gl, torch.zeros((), device=spikes.device), torch.zeros((), device=spikes.device)
        if self.heavy_frozen:                      # Task 03: fixed teacher
            with heavy_teacher_ctx():
                heavy, L_pde, H, S, L_cov, L_norm = self.heavy_path(x)
        else:
            heavy, L_pde, H, S, L_cov, L_norm = self.heavy_path(x)
        if mode == "dense":
            return heavy, torch.ones_like(g), rate, L_pde, H, S, gl, L_cov, L_norm
        # Task 05: binary gate selects one expert's logits exactly (no attenuation)
        return route_exact(cheap, heavy, g), g, rate, L_pde, H, S, gl, L_cov, L_norm

    def encode(self, x):
        return self.encoder(x)


# ── data loading ──────────────────────────────────────────────────────────────
def load_fi2010(log: ExperimentLogger, label_col=LABEL_COL):
    log.section(f"Loading FI-2010 (NoAuction Z-Score, {label_col})")
    import pyarrow.parquet as pq   # lazy: unit tests don't need pyarrow
    t0  = time.time()
    tbl = pq.read_table(ROOT / "data" / "reference" / "fi2010" / "fi2010_noauction_zscore.parquet")
    df  = tbl.to_pandas()
    log.info(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    feat_cols = [f"feature_{i}" for i in range(144)]

    train_df = df[df["split"] == "Train"].reset_index(drop=True)
    test_df  = df[df["split"] == "Test"].reset_index(drop=True)

    val_df   = train_df[train_df["fold"] >= 8].reset_index(drop=True)
    train_df = train_df[train_df["fold"] <  8].reset_index(drop=True)

    def to_tensors(d):
        X = d[feat_cols].values.astype(np.float32)
        y = d[label_col].values.astype(np.int64)
        return X, y

    Xtr, ytr = to_tensors(train_df)
    Xva, yva = to_tensors(val_df)
    Xte, yte = to_tensors(test_df)

    log.metric("Train rows", len(Xtr))
    log.metric("Val rows",   len(Xva))
    log.metric("Test rows",  len(Xte))
    log.info(f"Class balance (train): DOWN={(ytr==0).sum()}  STAT={(ytr==1).sum()}  UP={(ytr==2).sum()}")
    return Xtr, ytr, Xva, yva, Xte, yte


def load_fi2010_setup2(log: ExperimentLogger, label_col=LABEL_COL):
    """Task 07: Setup 2 (train days 1-7, test days 8-10) with stock/day segment ids."""
    from fi2010_setup2 import load_setup2
    path = ROOT / "data" / "reference" / "fi2010" / "fi2010_setup2.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:  python procure_fi2010.py --setup2  "
            "(then inspect segment_audit.json), or pass --legacy_split to reproduce the old protocol.")
    log.section(f"Loading FI-2010 Setup 2 ({label_col})")
    d = load_setup2(path, label_col)
    for k in ("train", "val", "test"):
        log.metric(f"{k} rows", len(d[k]["X"]))
    return d


def build_regime_mask(feats: np.ndarray) -> np.ndarray:
    """2σ realized-volatility proxy on the FI-2010 mid-price series."""
    import pandas as pd
    # Mid-price proxy: average of feature_0 (best ask price) and feature_2 (best bid price)
    mid   = (feats[:, 0] + feats[:, 2]) / 2.0
    ret   = pd.Series(mid).diff().fillna(0)
    rv    = ret.rolling(20, min_periods=1).std().fillna(0)
    mu    = rv.rolling(REGIME_WINDOW, min_periods=1).mean()
    sg    = rv.rolling(REGIME_WINDOW, min_periods=1).std().fillna(0)
    regime = ((rv > mu + REGIME_SIGMA * sg)).astype(np.float32).values
    return regime


def make_windows_from_array(X, y, regime, seg_id=None, horizon=0):
    """
    Sliding windows on CPU.  Task 04/06: dt is the fixed event-index step (1.0),
    not random noise; the last valid window is included (no off-by-one).
    With `seg_id` (per-row stock/day id) windows never straddle segments and the
    last `horizon` rows of a segment are purged (Task 07).
    """
    from qspinn_core import window_starts
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    rt = torch.from_numpy(regime)
    Xw = Xt.unfold(0, SEQ_LEN, 1).transpose(1, 2)
    rw = rt.unfold(0, SEQ_LEN, 1).max(dim=1)[0]
    starts = torch.from_numpy(window_starts(len(X), SEQ_LEN, seg_id, horizon))
    yw = yt[starts + SEQ_LEN - 1]
    dt_w = torch.ones(1, SEQ_LEN).expand(len(starts), SEQ_LEN)
    return Xw[starts], yw, rw[starts], dt_w


# ── metrics ───────────────────────────────────────────────────────────────────
def macro_f1(pred, true, n=3):
    f1s = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        d  = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
    return sum(f1s) / len(f1s)


def eval_stage_A(model, Xva, yva, dt_va, device, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), batch_size):
            Xb = Xva[i:i+batch_size].to(device).contiguous()
            dt_b = dt_va[i:i+batch_size].to(device).contiguous()
            Sb = model.encode(Xb)
            c, _, _, _ = model.gate(Sb, dt_b)
            preds.append(c.argmax(1).cpu())
    return macro_f1(torch.cat(preds), yva.cpu())


def stage_A(model, Xtr, ytr, dt_tr, Xva, yva, dt_va, device, log: ExperimentLogger, epochs=EPOCHS_A, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage A — SNN gate cheap head  ({epochs} epochs)")
    params = [p for n, p in model.named_parameters()
              if n.startswith("gate.") and "gate_head" not in n]
    opt  = torch.optim.Adam(params, lr=LR)
    counts = torch.bincount(ytr)
    cw = 1.0 / counts.float()
    cw = (cw / cw.sum() * 3.0).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)

    ckpt = load_checkpoint(ckpt_path, log)
    start_ep   = ckpt["epoch"] if ckpt and ckpt.get("stage") == "A" else 0
    best       = ckpt.get("best_f1", -1.0) if ckpt and ckpt.get("stage") == "A" else -1.0
    snap       = ckpt.get("best_state")    if ckpt and ckpt.get("stage") == "A" else None
    if ckpt and ckpt.get("stage") == "A":
        load_state_compat(model, ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)

    for ep in range(start_ep, epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss, nan_hit = 0.0, False
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            Xb = Xtr[b].to(device).contiguous()
            dt_b = dt_tr[b].to(device).contiguous()
            Sb = model.encode(Xb)
            yb = ytr[b].to(device).contiguous()
            
            cheap, _, rate, _ = model.gate(Sb, dt_b)
            loss = ce(cheap, yb) + 1e-4 * rate
            guard.step(loss, params, opt)
            if guard.halt:
                log.warn(f"Stage A: {guard.consecutive} consecutive non-finite updates skipped; halting.")
                nan_hit = True; break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)
            
        if nan_hit:
            break
            
        f1 = eval_stage_A(model, Xva, yva, dt_va, device, batch_size=batch_size)
        if f1 > best:
            best, snap = f1, {k: v.clone() for k, v in model.state_dict().items()}
            
        save_checkpoint({"epoch": ep+1, "stage": "A", "model": model.state_dict(),
                         "optimizer": opt.state_dict(), "best_f1": best, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr), val_f1=f1, eta_s=eta)

    if snap is not None:
        load_state_compat(model, snap)
    
    # Final eval for metrics
    model.eval()
    with torch.no_grad():
        Xb = Xva[:1024].to(device).contiguous()
        dt_b = dt_va[:1024].to(device).contiguous()
        _, _, r, _ = model.gate(model.encode(Xb), dt_b)
    log.stage("Stage A done", {"val_F1": eval_stage_A(model, Xva, yva, dt_va, device, batch_size=batch_size), "spike_rate": r.item()})
    return model


def eval_stage_B(model, Xva, yva, device, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xva), batch_size):
            Xb = Xva[i:i+batch_size].to(device).contiguous()
            h, _, _, _, _, _ = model.heavy_path(Xb)
            preds.append(h.argmax(1).cpu())
    return macro_f1(torch.cat(preds), yva.cpu())


def stage_B(model, Xtr, ytr, rtr, Xva, yva, rva, device, log: ExperimentLogger, epochs=EPOCHS_B, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage B — Heavy path (EM Cross-Fitted Training + EWS Regularization) ({epochs} epochs)")
    
    # Joint Optimization for Physics Generator + Classification Head
    params_heavy = list(model.amp.parameters()) + list(model.ham.parameters()) + list(model.head.parameters())
    opt_heavy = torch.optim.Adam(params_heavy, lr=LR)
    
    counts = torch.bincount(ytr)
    cw = 1.0 / counts.float()
    cw = (cw / cw.sum() * 3.0).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)
    
    ckpt = load_checkpoint(ckpt_path, log)
    start_ep   = ckpt["epoch"] if ckpt and ckpt.get("stage") == "B" else 0
    best       = ckpt.get("best_f1", -1.0) if ckpt and ckpt.get("stage") == "B" else -1.0
    snap       = ckpt.get("best_state")    if ckpt and ckpt.get("stage") == "B" else None
    if ckpt and ckpt.get("stage") == "B":
        load_state_compat(model, ckpt["model"])
        try:
            opt_heavy.load_state_dict(ckpt["optimizer"]["heavy"])
        except:
            log.info("Could not load split optimizer state from legacy checkpoint. Starting fresh optimizers.")
    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)

    for ep in range(start_ep, epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss, nan_hit = 0.0, False
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            Xb = Xtr[b].to(device).contiguous()
            yb = ytr[b].to(device).contiguous()
            rb = rtr[b].to(device).contiguous()
            
            # Joint PINN Forward Pass
            heavy, L_pde, H, S, L_cov, L_norm = model.heavy_path(Xb)
            
            # EWS Numerical Abscissa Stability Loss (Eq 28.3)
            # Contractive dynamics (omega <= -0.02) during normal regimes (rb == 0)
            # Expansive dynamics (omega >= +0.02) during volatility shocks (rb == 1)
            omegas = torch.linalg.eigvalsh(S)[:, -1]
            loss_ews = (torch.relu(omegas + 0.02) * (1.0 - rb) + torch.relu(0.02 - omegas) * rb).mean()
            
            # V3-04 & V3-05: Structural constraints + EWS + Classification Loss
            loss_phys = LAMBDA_PDE * L_pde + 0.1 * L_cov + 0.1 * L_norm + 0.1 * loss_ews
            loss_head = ce(heavy, yb)
            loss = loss_phys + loss_head
            
            guard.step(loss, params_heavy, opt_heavy)
            if guard.halt:
                log.warn(f"Stage B: {guard.consecutive} consecutive non-finite updates skipped; halting.")
                nan_hit = True; break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)
            
        if nan_hit:
            break
            
        f1 = eval_stage_B(model, Xva, yva, device, batch_size=batch_size)
        if f1 > best:
            best, snap = f1, {k: v.clone() for k, v in model.state_dict().items()}
            
        save_checkpoint({"epoch": ep+1, "stage": "B", "model": model.state_dict(),
                         "optimizer": {"heavy": opt_heavy.state_dict()}, "best_f1": best, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr), val_f1=f1, eta_s=eta)

    if snap is not None:
        load_state_compat(model, snap)
    
    # Final eval for metrics
    model.eval()
    with torch.no_grad():
        Xb = Xva[:1024].to(device).contiguous()
        _, L_pde, _, _, L_cov, L_norm = model.heavy_path(Xb)
    log.stage("Stage B done", {"val_F1": eval_stage_B(model, Xva, yva, device, batch_size=batch_size), "L_schrodinger": L_pde.item(), "L_cov": L_cov.item(), "L_norm": L_norm.item()})
    return model


def eval_stage_C(model, Xva, yva, dt_va, device, batch_size=1024):
    model.eval()
    preds, meter = [], RateMeter()
    with torch.no_grad():
        for i in range(0, len(Xva), batch_size):
            Xb = Xva[i:i+batch_size].to(device).contiguous()
            dt_b = dt_va[i:i+batch_size].to(device).contiguous()
            Sb = model.encode(Xb)
            lo, g, _, _, _, _, _, _, _ = model(Sb, Xb, dt_b)
            preds.append(lo.argmax(1).cpu())
            meter.update(g)
    f1 = macro_f1(torch.cat(preds), yva.cpu())
    return f1, meter.rate


def stage_C(model, Xtr, ytr, rtr, dt_tr, Xva, yva, rva, dt_va, device, log: ExperimentLogger, epochs=EPOCHS_C, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage C — Gate head arbitration  ({epochs} epochs)")
    gate_params = list(model.gate.gate_head.parameters()) + [model.log_var_ce, model.log_var_regime]
    params = stage_c_mode(model, gate_params)   # Task 03: heavy frozen + eval + no-grad teacher
    model.heavy_frozen = True
    opt  = torch.optim.Adam(params, lr=5e-3)
    ce   = nn.CrossEntropyLoss()
    pos_weight = torch.tensor([15.0]).to(device)
    bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    ckpt = load_checkpoint(ckpt_path, log)
    start_ep   = ckpt["epoch"] if ckpt and ckpt.get("stage") == "C" else 0
    best_score = ckpt.get("best_f1", -1e9) if ckpt and ckpt.get("stage") == "C" else -1e9
    snap       = ckpt.get("best_state")    if ckpt and ckpt.get("stage") == "C" else {k: v.clone() for k, v in model.state_dict().items()}
    if ckpt and ckpt.get("stage") == "C":
        load_state_compat(model, ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)

    for ep in range(start_ep, epochs):
        model.eval()   # Task 03: no HamiltonianNet sampling noise in Stage C
        perm = torch.randperm(len(Xtr))
        ep_loss, nan_hit = 0.0, False
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            Xb = Xtr[b].to(device).contiguous()
            dt_b = dt_tr[b].to(device).contiguous()
            Sb = model.encode(Xb)
            yb = ytr[b].to(device).contiguous()
            rb = rtr[b].to(device).contiguous()
            
            logits, g, _, _, _, _, gl, _, _ = model(Sb, Xb, dt_b)
            loss_ce = ce(logits, yb)
            loss_gate = g.mean()
            loss_regime = bce(gl * 4.0, rb)
            
            # SWAP-08: Adaptive Loss Weighting (Kendall et al. 2018)
            log_ce = torch.clamp(model.log_var_ce, min=-5.0, max=10.0)
            log_regime = torch.clamp(model.log_var_regime, min=-5.0, max=10.0)
            
            adaptive_ce = loss_ce * torch.exp(-log_ce) + log_ce
            adaptive_regime = loss_regime * torch.exp(-log_regime) + log_regime
            
            # Sparsity enforcement: penalize gate rates over 10%
            loss_gate_target = torch.relu(loss_gate - 0.10) * 5.0 + 0.2 * loss_gate
            loss = adaptive_ce + adaptive_regime + loss_gate_target
            guard.step(loss, params, opt)
            if guard.halt:
                log.warn(f"Stage C: {guard.consecutive} consecutive non-finite updates skipped; halting.")
                nan_hit = True; break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)
        if nan_hit:
            break
            
        f1, gate_rate = eval_stage_C(model, Xva, yva, dt_va, device, batch_size=batch_size)
        penalty = max(0.0, gate_rate - 0.12) * 5.0 + gate_rate * 0.1
        score = f1 - penalty
        
        if score > best_score:
            best_score = score
            snap = {k: v.clone() for k, v in model.state_dict().items()}
            
        save_checkpoint({"epoch": ep+1, "stage": "C", "model": model.state_dict(),
                         "optimizer": opt.state_dict(), "best_f1": best_score, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr),
                      val_f1=f1, extra={"gate": gate_rate}, eta_s=eta)

    if snap is not None:
        load_state_compat(model, snap)
    return model


def evaluate(model, X, y, r, dt_w, mode="learned", batch_size=1024, return_pred=False):
    model.eval()
    preds, rates, pdes, omegas = [], [], [], []
    meter = RateMeter()
    H_herm_max = 0.0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = X[i:i+batch_size].to(device).contiguous()
            dt_b = dt_w[i:i+batch_size].to(device).contiguous()
            Sb = model.encode(Xb)
            logits, g, rate, L_pde, H, S, _, _, _ = model(Sb, Xb, dt_b, mode)
            preds.append(logits.argmax(1).cpu())
            meter.update(g)
            rates.append(rate.cpu())
            pdes.append(L_pde.cpu())
            
            # EWS Metric: Numerical Abscissa (omega) is max eigenvalue of Symmetric matrix S
            batch_omegas = torch.linalg.eigvalsh(S)[:, -1]
            omegas.append(batch_omegas.cpu())
            
            herm = (H - H.conj().transpose(-1, -2)).abs().max().item()
            if herm > H_herm_max: H_herm_max = herm
            
    pred = torch.cat(preds)
    if return_pred:
        return pred
    omega_all = torch.cat(omegas)
    
    return dict(
        acc      = (pred == y.cpu()).float().mean().item(),
        f1       = macro_f1(pred, y.cpu()),
        gate     = meter.rate,
        spike    = torch.tensor(rates).mean().item(),
        pde      = torch.tensor(pdes).mean().item(),
        herm     = H_herm_max,
        omega    = omega_all.mean().item(),
        crash_P  = (omega_all > 0).float().mean().item()
    )


# ── LSTM baseline ─────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, HIDDEN, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 3)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])

def train_lstm(Xtr, ytr, Xva, yva, device, epochs=20):
    m   = LSTMBaseline().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    counts = torch.bincount(ytr)
    cw = 1.0 / counts.float()
    cw = (cw / cw.sum() * 3.0).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)
    best, snap = -1.0, None
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            Xb = Xtr[b].to(device).contiguous()
            yb = ytr[b].to(device).contiguous()
            loss = ce(m(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        
        m.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva), 1024):
                Xb = Xva[i:i+1024].to(device).contiguous()
                preds.append(m(Xb).argmax(1).cpu())
        f1 = macro_f1(torch.cat(preds), yva.cpu())
        
        if f1 > best:
            best, snap = f1, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(snap)
    return m


# ── results doc ───────────────────────────────────────────────────────────────
def write_results(ablation_rows, m_full, timestamp, device_name, majority, out_path=None, horizon=10, seed=0, epochs_tuple=(100, 100, 100)):
    out_path = Path(out_path) if out_path else (OUT_DIR / "RESULTS.md")
    ep_a, ep_b, ep_c = epochs_tuple
    rows_md = ""
    for name, acc, f1, gate, herm, omega, crash in ablation_rows:
        gs = "n/a" if gate != gate else f"{gate:.3f}"
        os = "n/a" if omega != omega else f"{omega:.4f}"
        cs = "n/a" if crash != crash else f"{crash*100:.1f}%"
        rows_md += f"| {name} | {acc:.3f} | {f1:.3f} | {gs} | {herm:.2e} | {os} | {cs} |\n"

    md = rf"""# Experiment B — Full Q-SPINN Hybrid on FI-2010 (k={horizon}, Seed={seed})

**Run date:** {timestamp}  
**Device:** {device_name}  
**Source code:** [`train.py`](./train.py)

---

## Dataset

| Property | Value |
|---|---|
| Source | FI-2010 NoAuction Z-Score, **Setup 2** (`data/reference/fi2010/fi2010_setup2.parquet`; train days 1-7, test days 8-10; windows never cross stock/day segments) |
| Features | 144 (10 LOB levels × 4 quantities × 1 stock, + 4 derived) |
| Label | `label_k{horizon}` (3-class mid-price direction, {horizon}-tick horizon) |
| Seed | {seed} |
| Train | Setup 2 days 1-6 |
| Val | Setup 2 day 7 |
| Test | Setup 2 days 8, 9, 10 |
| Normalization | Pre-normalized by FI-2010 authors (Z-score, no leakage) |
| Regime proxy | realized_vol > μ + {REGIME_SIGMA}σ (rolling {REGIME_WINDOW}-tick window) |

## Ablation Results

| Model | Accuracy | Macro-F1 | Gate Rate | Hermiticity ‖H−H†‖ | $\omega(A_t)$ (latent-flow diagnostic) | $\omega>0$ rate (unvalidated, not a crash probability) |
|---|---|---|---|---|---|---|
{rows_md}| **Majority baseline** | **{majority:.3f}** | — | — | — | — | — |

## Training Protocol

Three-stage training was used:

1. **Stage A ({ep_a} epochs):** SNN gate cheap head only (heavy path frozen)
2. **Stage B ({ep_b} epochs):** Heavy path only + EWS regularization (SNN frozen)
3. **Stage C ({ep_c} epochs):** Gate head only with sparsity calibration (everything else frozen)

## Notes on interpretation

- `‖H − H†‖` is zero because H is *constructed* Hermitian; it checks the code path, it is not an empirical result.
- The gate rate is the **global** share of test windows routed to the heavy expert. It is a budget statistic, not a
  compute saving: measured latency is reported separately by `routing_study.py`.
- $\omega(A_t)=\lambda_{{\max}}(S)$ is a latent-flow diagnostic. Whether it forecasts instability is decided only by
  `early_warning_study.py` against a rolling-volatility baseline.
- Nothing in this file is a claim about trading performance.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Results saved to {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Exp B — Full Q-SPINN Hybrid on FI-2010")
    parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50], help="Prediction horizon k")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs_a", type=int, default=EPOCHS_A, help="Stage A epochs")
    parser.add_argument("--epochs_b", type=int, default=EPOCHS_B, help="Stage B epochs")
    parser.add_argument("--epochs_c", type=int, default=EPOCHS_C, help="Stage C epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="", help="Custom output directory")
    parser.add_argument("--run_name", type=str, default="", help="Run tag for logs")
    parser.add_argument("--legacy_split", action="store_true",
                        help="Old concatenated-fold protocol (NOT Setup 2); results must be labelled legacy")
    return parser.parse_args()


def main(horizon=None, seed=None, epochs_a=None, epochs_b=None, epochs_c=None, batch_size=None, out_dir=None, run_name="", legacy_split=False):
    if horizon is None and __name__ == "__main__" and len(sys.argv) > 1:
        args = parse_args()
        horizon = args.horizon
        seed = args.seed
        epochs_a = args.epochs_a
        epochs_b = args.epochs_b
        epochs_c = args.epochs_c
        batch_size = args.batch_size
        out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
        run_name = args.run_name
        legacy_split = args.legacy_split
    else:
        horizon = 10 if horizon is None else horizon
        seed = 0 if seed is None else seed
        epochs_a = EPOCHS_A if epochs_a is None else epochs_a
        epochs_b = EPOCHS_B if epochs_b is None else epochs_b
        epochs_c = EPOCHS_C if epochs_c is None else epochs_c
        batch_size = BATCH_SIZE if batch_size is None else batch_size
        out_dir = Path(out_dir) if out_dir else OUT_DIR
        run_name = run_name or f"k{horizon}_s{seed}"

    torch.manual_seed(seed); np.random.seed(seed)
    label_col = f"label_k{horizon}"
    ckpt_path = out_dir / f"checkpoint_k{horizon}_s{seed}.pt"

    log = ExperimentLogger(out_dir / f"train_k{horizon}_s{seed}.log",
                           experiment_name=f"Exp B — FI-2010 (k={horizon}, seed={seed})")
    device = get_device(verbose=True)

    # Load data for the requested horizon (Task 07: Setup 2 with provenance by default)
    if legacy_split:
        log.warn("LEGACY concatenated-fold split - results are NOT Setup 2 and must be labelled legacy.")
        Xtr_np, ytr_np, Xva_np, yva_np, Xte_np, yte_np = load_fi2010(log, label_col=label_col)
        segs = dict(train=None, val=None, test=None)
    else:
        d = load_fi2010_setup2(log, label_col=label_col)
        Xtr_np, ytr_np = d["train"]["X"], d["train"]["y"]
        Xva_np, yva_np = d["val"]["X"],   d["val"]["y"]
        Xte_np, yte_np = d["test"]["X"],  d["test"]["y"]
        segs = {k: d[k]["seg"] for k in ("train", "val", "test")}

    log.section("Building 2σ regime masks")
    if legacy_split:
        rtr_np, rva_np, rte_np = (build_regime_mask(a) for a in (Xtr_np, Xva_np, Xte_np))
    else:   # per-segment so volatility statistics never leak across stock/day joins
        from fi2010_setup2 import segmented_regime_mask
        rtr_np, rva_np, rte_np = (segmented_regime_mask(Xs, segs[k], REGIME_WINDOW, REGIME_SIGMA)
                                  for Xs, k in ((Xtr_np, "train"), (Xva_np, "val"), (Xte_np, "test")))
    log.metric("Train regime %", rtr_np.mean() * 100, unit="%")
    log.metric("Val regime %",   rva_np.mean() * 100, unit="%")
    log.metric("Test regime %",  rte_np.mean() * 100, unit="%")

    log.section("Building sequence windows (SEQ_LEN=20, per stock-day segment)")
    Xtr, ytr, rtr, dt_tr = make_windows_from_array(Xtr_np, ytr_np, rtr_np, segs["train"], horizon)
    Xva, yva, rva, dt_va = make_windows_from_array(Xva_np, yva_np, rva_np, segs["val"],   horizon)
    Xte, yte, rte, dt_te = make_windows_from_array(Xte_np, yte_np, rte_np, segs["test"],  horizon)

    mode_val = torch.mode(yte)[0].item()
    majority = (yte == mode_val).float().mean().item()
    log.section("Dataset summary")
    log.metric("Train windows", len(Xtr))
    log.metric("Val windows",   len(Xva))
    log.metric("Test windows",  len(Xte))
    log.metric("Majority baseline", majority)
    majority_f1 = macro_f1(torch.full_like(yte, mode_val), yte)   # Task 07: majority-class macro-F1
    log.metric("Majority-class macro-F1", majority_f1)

    try:
        model = QSPINN().to(device)
        vp = None if segs["train"] is None else (segs["train"][1:] == segs["train"][:-1])   # no stock/day-join deltas
        model.encoder.fit(Xtr_np, valid_pair=vp)   # train-only thresholds, same encoder for val/test
        ckpt = load_checkpoint(ckpt_path, log)
        current_stage = ckpt["stage"] if ckpt else None
        
        if current_stage not in ["B", "C"]:
            model = stage_A(model, Xtr, ytr, dt_tr, Xva, yva, dt_va, device, log,
                            epochs=epochs_a, ckpt_path=ckpt_path, batch_size=batch_size)
            
        if current_stage != "C":
            model = stage_B(model, Xtr, ytr, rtr, Xva, yva, rva, device, log,
                            epochs=epochs_b, ckpt_path=ckpt_path, batch_size=batch_size)
            
        model = stage_C(model, Xtr, ytr, rtr, dt_tr, Xva, yva, rva, dt_va, device, log,
                        epochs=epochs_c, ckpt_path=ckpt_path, batch_size=batch_size)
    except Exception as e:
        log.warn(f"Training crashed: {type(e).__name__}: {e}")
        log.warn("Last checkpoint is intact — re-run to resume from the last completed epoch.")
        raise

    # Ablation suite
    log.section("Ablation suite")
    ablation_rows = []

    lstm_epochs = 1 if epochs_a <= 2 else 20
    lstm = train_lstm(Xtr, ytr, Xva, yva, device, epochs=lstm_epochs)
    lstm.eval()
    lstm_preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            Xb = Xte[i:i+1024].to(device).contiguous()
            lstm_preds.append(lstm(Xb).argmax(1).cpu())
    p = torch.cat(lstm_preds)
    ablation_rows.append(("LSTM baseline", (p == yte.cpu()).float().mean().item(),
                           macro_f1(p, yte.cpu()), float('nan'), float('nan'), float('nan'), float('nan')))

    # Task 15: DeepLOB re-implementation on the same windows/split/epoch budget as the LSTM baseline
    from baselines import DeepLOB, train_classifier, predict, count_params
    dl = train_classifier(DeepLOB(), Xtr, ytr, Xva, yva, device, macro_f1, epochs=lstm_epochs, seed=seed)
    p_dl = predict(dl, Xte, device)
    log.metric("DeepLOB params", count_params(dl))
    ablation_rows.append(("DeepLOB (re-implemented, same split)", (p_dl == yte.cpu()).float().mean().item(),
                           macro_f1(p_dl, yte.cpu()), float('nan'), float('nan'), float('nan'), float('nan')))

    m_cheap = evaluate(model, Xte, yte, rte, dt_te, mode="cheap")
    ablation_rows.append(("SNN only", m_cheap["acc"], m_cheap["f1"], 0.0, m_cheap["herm"], m_cheap["omega"], m_cheap["crash_P"]))

    m_dense = evaluate(model, Xte, yte, rte, dt_te, mode="dense")
    ablation_rows.append(("PINN only (always awake)", m_dense["acc"], m_dense["f1"], 1.0, m_dense["herm"], m_dense["omega"], m_dense["crash_P"]))

    m = evaluate(model, Xte, yte, rte, dt_te, mode="learned")
    ablation_rows.append(("Q-SPINN (full hybrid)", m["acc"], m["f1"], m["gate"], m["herm"], m["omega"], m["crash_P"]))

    from stats import block_bootstrap_f1
    ci = {}
    for tag, mode in (("hybrid", "learned"), ("cheap", "cheap"), ("dense", "dense")):
        pr = evaluate(model, Xte, yte, rte, dt_te, mode=mode, return_pred=True)
        ci[tag] = block_bootstrap_f1(pr.numpy(), yte.numpy(), n_boot=500)
        log.metric(f"macro-F1 95% CI ({tag})", f"{ci[tag][0]:.3f} [{ci[tag][1]:.3f}, {ci[tag][2]:.3f}]")
    log.table(
        headers=["Model", "Acc", "Macro-F1", "Gate", "||H-H†||", "ω(A_t)", "w>0 rate"],
        rows=[
            (name, f"{acc:.3f}", f"{f1:.3f}",
             "n/a" if gate != gate else f"{gate:.3f}",
             "n/a" if herm != herm else f"{herm:.2e}",
             "n/a" if omega != omega else f"{omega:.4f}",
             "n/a" if crash != crash else f"{crash*100:.1f}%")
            for name, acc, f1, gate, herm, omega, crash in ablation_rows
        ] + [("Majority baseline", f"{majority:.3f}", f"{majority_f1:.3f}", "—", "—", "—", "—")],
        title=f"Ablation Results (k={horizon}, seed={seed})",
    )

    # Save outputs
    log.section("Saving outputs")
    weights_path = out_dir / f"qspinn_fi2010_k{horizon}_s{seed}.pt"
    torch.save(model.state_dict(), weights_path)
    log.info(f"Model weights  → {weights_path}")
    
    # Also save to default path for standard single-asset compatibility
    if horizon == 10 and seed == 0:
        torch.save(model.state_dict(), OUT_DIR / "qspinn_fi2010.pt")
    from live_contract import make_contract, save_contract   # Task 17: record the input contract
    save_contract(out_dir / f"qspinn_fi2010_k{horizon}_s{seed}_contract.json",
                  make_contract(SEQ_LEN, N_FEATURES, "fi2010_authors", "dataset_zscore", "equity"))
    if horizon == 10 and seed == 0:
        save_contract(OUT_DIR / "qspinn_fi2010_contract.json",
                      make_contract(SEQ_LEN, N_FEATURES, "fi2010_authors", "dataset_zscore", "equity"))

    timestamp = datetime.now().isoformat()
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    res_path = out_dir / f"RESULTS_k{horizon}_s{seed}.md"
    write_results(ablation_rows, m, timestamp, gpu_name, majority,
                  out_path=res_path, horizon=horizon, seed=seed,
                  epochs_tuple=(epochs_a, epochs_b, epochs_c))
    if horizon == 10 and seed == 0:
        write_results(ablation_rows, m, timestamp, gpu_name, majority,
                      out_path=OUT_DIR / "RESULTS.md", horizon=10, seed=0,
                      epochs_tuple=(epochs_a, epochs_b, epochs_c))

    if ckpt_path.exists():
        ckpt_path.unlink()
        log.info("Checkpoint deleted (run completed successfully)")

    full_m = ablation_rows[-1]
    res_summary = {
        "horizon"            : horizon,
        "seed"               : seed,
        "accuracy"           : full_m[1],
        "macro_f1"           : full_m[2],
        "gate_rate"          : full_m[3],
        "hermiticity"        : full_m[4] if full_m[4] == full_m[4] else 0.0,
        "omega"              : full_m[5] if full_m[5] == full_m[5] else 0.0,
        "crash_p"            : full_m[6] if full_m[6] == full_m[6] else 0.0,
        "majority"           : majority,
        "majority_macro_f1"  : majority_f1,
        "f1_ci95"            : list(ci["hybrid"][1:]),
        "f1_ci95_cheap"      : list(ci["cheap"][1:]),
        "f1_ci95_dense"      : list(ci["dense"][1:]),
        "weights_path"       : str(weights_path),
        "results_path"       : str(res_path)
    }
    log.done(summary=res_summary)
    return res_summary


if __name__ == "__main__":
    main()
