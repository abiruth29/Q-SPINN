"""
experiments/exp_C_multi_asset_binance/train.py
================================================
Experiment C — Full 3-Asset Q-SPINN Hybrid on Binance Crypto Basket.
(BTC/USDT, ETH/USDT, SOL/USDT)

WHAT THIS DOES
--------------
1. Ingests and aggregates real tick trades from data/curated/trades across
   BTCUSDT, ETHUSDT, and SOLUSDT into 1-second microstructure feature vectors.
2. Constructs the 18-feature multi-asset tensor (6 features x 3 assets).
3. Trains the full 3-asset Q-SPINN incorporating:
   - Continuous-time Magnetic Graph Laplacian (directed phase flux W_ij = A_ij * exp(i*Phi_ij))
   - Van Loan Continuous-to-Discrete Covariance Propagation
   - Early Warning System (Numerical Abscissa contractive regularization)
   - Event-driven SNN Gating
4. Evaluates lead-lag relationships (BTC phase lead over ETH and SOL).
5. Compares against SNN-only, PINN-only, and LSTM baselines.
6. Saves model weights and generates RESULTS.md.
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
import torch.nn.functional as F

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "curated" / "trades"
OUT_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))

from device import get_device
from logger import ExperimentLogger
from checkpoint import save_checkpoint, load_checkpoint, guard_nan
from qspinn_core import (propagate, bound_fro, hermitian_from_raw, symmetric_from_raw, GENERATOR_VERSION,
                         StepGuard, SpikeEncoder, route_exact, RateMeter, load_state_compat,
                         stage_c_mode, heavy_teacher_ctx)
import os
USE_COVARIANCE = os.environ.get("QSPINN_USE_COV", "0") == "1"   # Task 02: off by default
# v3.1 generator settings (Tasks 08-10)
RESIDUAL_MODE = os.environ.get("QSPINN_RESIDUAL", "last")   # 'last' (causal) | 'all' (legacy ablation)
H_FRO_MAX = 2.0     # spectral control on H
S_FRO_MAX = 1.0     # spectral control on S (growth per step <= e^1)
MAX_GRAD_NORM = 1.0                                              # Task 01

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── hyper-parameters ─────────────────────────────────────────────────────────
N_ASSETS        = 3
FEATS_PER_ASSET = 6           # log_return, ofi, realized_vol, buy_pressure, trade_count, vwap_spread
N_FEATURES      = N_ASSETS * FEATS_PER_ASSET   # 18
PSI_DIM         = 8           # Dimension per asset -> Total Psi Dim = 24
HIDDEN          = 64
SEQ_LEN         = 20
HORIZON         = 5           # 5-second prediction horizon
DT              = 1.0
LAMBDA_PDE      = 0.1
LAMBDA_GATE     = 0.02
LAMBDA_REGIME   = 1.0
REGIME_SIGMA    = 2.0
REGIME_WINDOW   = 200
SAMPLE_TICKS    = 350_000
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
BATCH_SIZE      = 2048
EPOCHS_A        = 50
EPOCHS_B        = 50
EPOCHS_C        = 50
LR              = 2e-3
SEED            = 0
CKPT_PATH       = OUT_DIR / "checkpoint.pt"


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
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, x_seq, dt_seq):
        B, T, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        out = []
        tau = F.softplus(self.tau_m) + 1e-3
        alpha_pos = F.softplus(self.alpha)
        for t in range(T):
            cur = self.fc(x_seq[:, t])
            beta_n = torch.exp(-dt_seq[:, t] / tau).unsqueeze(-1)
            mem = beta_n * mem + cur
            spk = spike_fn(mem - 1.0, alpha_pos)
            mem = mem - spk
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
        self.d = d * n_assets
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


# ── Hamiltonian net with Magnetic Graph Laplacian ─────────────────────────────
class HamiltonianNet(nn.Module):
    def __init__(self, n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS):
        super().__init__()
        self.d = d
        self.n_assets = n_assets
        
        out_dim = (n_assets * 3 * d * d) + (n_assets * n_assets) + (n_assets * n_assets)
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, out_dim))
        
        self.theta = nn.Parameter(torch.full((n_assets,), 0.1))
        self.log_Q = nn.Parameter(torch.full((n_assets,), -3.0))
        self.log_R = nn.Parameter(torch.full((n_assets,), -2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, velocities):
        B, seq_len, _ = velocities.shape
        
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
            
            if t == seq_len - 1:
                idx_A = self.n_assets * 3 * self.d * self.d
                idx_Phi = idx_A + self.n_assets * self.n_assets
                A_raw   = out[:, idx_A : idx_Phi].view(B, self.n_assets, self.n_assets)
                Phi_raw = out[:, idx_Phi : ].view(B, self.n_assets, self.n_assets)
            
            if z_t is None:
                z_t = mu_t
                P_t = P_t * R
            else:
                z_pred = z_t + theta_pos * (mu_t - z_t)
                P_pred = P_t + Q
                K_t = P_pred / (P_pred + R)
                z_t = z_pred + K_t * (mu_t - z_pred)
                P_t = (1 - K_t) * P_pred
        
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

        # Magnetic Graph Laplacian Construction
        A_sym = F.softplus(A_raw + A_raw.transpose(-1, -2))
        Phi_anti = Phi_raw - Phi_raw.transpose(-1, -2)
        W = A_sym * torch.exp(1j * Phi_anti)
        
        D_diag = A_sym.sum(dim=-1)
        D = torch.diag_embed(D_diag)
        L = torch.complex(D, torch.zeros_like(D)) - W
        
        # Kronecker coupling: H_tot = BlockDiag(H_k) + L (x) I_d
        H_tot = torch.zeros(B, self.n_assets * self.d, self.n_assets * self.d, dtype=torch.complex64, device=velocities.device)
        S_tot = torch.zeros(B, self.n_assets * self.d, self.n_assets * self.d, dtype=torch.float32, device=velocities.device)
        I_d = torch.eye(self.d, dtype=torch.complex64, device=velocities.device)
        
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                si, ei = i * self.d, (i+1) * self.d
                sj, ej = j * self.d, (j+1) * self.d
                if i == j:
                    H_tot[:, si:ei, sj:ej] = H_local[:, i] + L[:, i, j].view(B, 1, 1) * I_d
                    S_tot[:, si:ei, sj:ej] = S_local[:, i]
                else:
                    H_tot[:, si:ei, sj:ej] = L[:, i, j].view(B, 1, 1) * I_d
                    
        return (bound_fro(self.scale * H_tot, H_FRO_MAX), bound_fro(self.scale * S_tot, S_FRO_MAX), A_sym, Phi_anti)


# ── Van Loan Propagator ───────────────────────────────────────────────────────
class VanLoanCovariancePropagator(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.B_R = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        
    def forward(self, A_R, dt=0.1):
        B_dim, N, _ = A_R.shape
        BBT = self.B_R @ self.B_R.transpose(-1, -2)
        BBT_batch = BBT.unsqueeze(0).expand(B_dim, -1, -1)
        
        M = torch.zeros(B_dim, 2 * N, 2 * N, device=A_R.device, dtype=A_R.dtype)
        M[:, :N, :N] = -A_R
        M[:, :N, N:] = BBT_batch
        M[:, N:, N:] = A_R.transpose(-1, -2)
        M = M * dt
        
        E = torch.matrix_exp(M)
        E12 = E[:, :N, N:]
        E22 = E[:, N:, N:]
        
        F_mat = E22.transpose(-1, -2)
        Q_h = F_mat @ E12
        return F_mat, Q_h


# ── SNN Gate ─────────────────────────────────────────────────────────────────
class SNNGate(nn.Module):
    def __init__(self, n_in=N_FEATURES):
        super().__init__()
        self.lif1       = IrregularLIFLayer(2 * n_in, HIDDEN)
        self.lif2       = IrregularLIFLayer(HIDDEN, HIDDEN)
        self.cheap_head = LeakyReadout(HIDDEN, 3)
        self.gate_head  = LeakyReadout(HIDDEN, 1)
        self.alpha_gate = nn.Parameter(torch.tensor(2.0))
        nn.init.constant_(self.gate_head.fc.bias, 0.5)

    def forward(self, spikes, dt_seq):
        s1, _ = self.lif1(spikes, dt_seq)
        s2, _ = self.lif2(s1, dt_seq)
        cheap_logits = self.cheap_head(s2)
        gate_logit   = torch.tanh(self.gate_head(s2).squeeze(-1) / 4.0)
        g            = spike_fn(gate_logit, F.softplus(self.alpha_gate))
        spike_rate   = 0.5 * (s1.mean() + s2.mean())
        return cheap_logits, g, spike_rate, gate_logit


# ── Multi-Asset Q-SPINN ──────────────────────────────────────────────────────
class MultiAssetQSPINN(nn.Module):
    def __init__(self, n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS, use_cov=None):
        super().__init__()
        self.use_cov = USE_COVARIANCE if use_cov is None else use_cov
        self.heavy_frozen = False
        self.encoder = SpikeEncoder(n_in)
        self.gate = SNNGate(n_in=n_in)
        self.amp  = ComplexAmplitudeLayer(n_in=n_in, d=d, n_assets=n_assets)
        self.ham  = HamiltonianNet(n_in=n_in, d=d, n_assets=n_assets)
        
        total_psi_dim = d * n_assets
        n_rho = 3 * (total_psi_dim + 2 * (total_psi_dim * (total_psi_dim - 1) // 2))
        self.head = nn.Sequential(nn.Linear(n_rho, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))
        
        self.van_loan = VanLoanCovariancePropagator(2 * total_psi_dim)
        n_cov = (2 * total_psi_dim) * (2 * total_psi_dim)
        self.cov_head = nn.Sequential(nn.Linear(total_psi_dim * 2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_cov))
        
        self.log_var_ce     = nn.Parameter(torch.tensor(0.0))
        self.log_var_regime = nn.Parameter(torch.tensor(0.0))

    def heavy_path(self, x):
        velocities = x[:, 1:] - x[:, :-1]
        H_tot, S_tot, A_sym, Phi_anti = self.ham(velocities)
        
        B, N, _ = H_tot.shape
        H_R = H_tot.real
        H_I = H_tot.imag
        
        A_R_mat = S_tot + H_I
        A_I_mat = -H_R
        
        if self.use_cov:   # Task 02: Van Loan never executed when disabled
            A_block = torch.zeros(B, 2 * N, 2 * N, device=x.device, dtype=torch.float32)
            A_block[:, :N, :N] = A_R_mat
            A_block[:, :N, N:] = -A_I_mat
            A_block[:, N:, :N] = A_I_mat
            A_block[:, N:, N:] = A_R_mat
        
            F_mat, Q_h = self.van_loan(A_block, DT)
        # Task 09/10: ONE generator A = S - iH drives the means; residuals are causal ('last').
        psi = self.amp(x)     # unnormalised dynamical state
        U, L_pde, L_norm_ = propagate(psi, H_tot, S_tot, DT, RESIDUAL_MODE)

        if self.use_cov:
            z_T = torch.cat([psi[:, -1].real, psi[:, -1].imag], dim=-1)
            Sigma_theta_flat = self.cov_head(z_T)
            Sigma_theta = Sigma_theta_flat.view(B, 2 * N, 2 * N)
            Sigma_theta = Sigma_theta @ Sigma_theta.transpose(-1, -2)
        
            R_Sigma = F_mat @ Sigma_theta @ F_mat.transpose(-1, -2) + Q_h - Sigma_theta
            L_cov = R_Sigma.abs().pow(2).mean()
        
        else:
            L_cov = torch.zeros((), device=x.device)

        L_norm = L_norm_

        psi_T    = ComplexAmplitudeLayer.normalise(psi[:, -1])
        psi_next = ComplexAmplitudeLayer.normalise(torch.einsum('bij,bj->bi', U, psi[:, -1]))
        feats    = torch.cat([
            ComplexAmplitudeLayer.density_features(psi_T),
            ComplexAmplitudeLayer.density_features(ComplexAmplitudeLayer.normalise(psi.mean(1))),
            ComplexAmplitudeLayer.density_features(psi_next)], dim=-1)
            
        return self.head(feats), L_pde, H_tot, S_tot, L_cov, L_norm, A_sym, Phi_anti

    def forward(self, spikes, x, dt_seq, mode="learned"):
        cheap, g, rate, gl = self.gate(spikes, dt_seq)
        total_dim = PSI_DIM * N_ASSETS
        if mode == "cheap":
            H0 = torch.zeros(1, total_dim, total_dim, dtype=torch.cfloat, device=spikes.device)
            S0 = torch.zeros(1, total_dim, total_dim, dtype=torch.float32, device=spikes.device)
            A0 = torch.zeros(1, N_ASSETS, N_ASSETS, device=spikes.device)
            P0 = torch.zeros(1, N_ASSETS, N_ASSETS, device=spikes.device)
            return cheap, torch.zeros_like(g), rate, torch.zeros((), device=spikes.device), H0, S0, gl, torch.zeros((), device=spikes.device), torch.zeros((), device=spikes.device), A0, P0
            
        if self.heavy_frozen:                      # Task 03
            with heavy_teacher_ctx():
                heavy, L_pde, H, S, L_cov, L_norm, A_sym, Phi_anti = self.heavy_path(x)
        else:
            heavy, L_pde, H, S, L_cov, L_norm, A_sym, Phi_anti = self.heavy_path(x)
        if mode == "dense":
            return heavy, torch.ones_like(g), rate, L_pde, H, S, gl, L_cov, L_norm, A_sym, Phi_anti
            
        # Task 05: exact binary hand-over (no logit attenuation)
        return route_exact(cheap, heavy, g), g, rate, L_pde, H, S, gl, L_cov, L_norm, A_sym, Phi_anti

    def encode(self, x):
        return self.encoder(x)


# ── data engineering ──────────────────────────────────────────────────────────
_BATCH = 2_000_000
_AGG_OPS = {
    "mid_sum": "sum", "mid_count": "sum", "ofi": "sum",
    "buy_vol": "sum", "sell_vol":  "sum", "trade_count": "sum",
    "total_pq": "sum", "total_qty": "sum",
}

def _batch_to_seconds(batch, time_col="time"):
    import pandas as pd
    df = batch.to_pandas()
    t = df[time_col]
    sample_val = t.iloc[0] if len(t) > 0 else 0
    if sample_val > 1e14:      # microseconds (~1e15)
        df["second_bin"] = (t // 1_000_000).astype("int64")
    elif sample_val > 1e11:    # milliseconds (~1e12)
        df["second_bin"] = (t // 1_000).astype("int64")
    else:                      # seconds (~1e9)
        df["second_bin"] = t.astype("int64")

    # In Binance, is_buyer_maker=True means seller was aggressive taker (sell trade)
    df["signed_qty"] = df["qty"].where(~df["is_buyer_maker"], -df["qty"])
    df["qty_buy"]    = df["qty"].where(~df["is_buyer_maker"], 0.0)
    df["qty_sell"]   = df["qty"].where(df["is_buyer_maker"], 0.0)
    df["price_qty"]  = df["price"] * df["qty"]
    g = df.groupby("second_bin", sort=True)
    return pd.DataFrame({
        "mid_sum"    : g["price"].sum(),
        "mid_count"  : g["price"].count(),
        "ofi"        : g["signed_qty"].sum(),
        "buy_vol"    : g["qty_buy"].sum(),
        "sell_vol"   : g["qty_sell"].sum(),
        "trade_count": g["qty"].count(),
        "total_pq"   : g["price_qty"].sum(),
        "total_qty"  : g["qty"].sum(),
    })


def engineer_features_for_symbol(symbol_dir: Path, max_ticks: int):
    import pyarrow.parquet as pq   # lazy: unit tests don't need pyarrow
    import pandas as pd
    files = sorted(symbol_dir.rglob("*.parquet"))
    if not files: return None
    all_second_dfs, ticks_so_far = [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        schema_cols = pf.schema.names
        time_col = "timestamp_us" if "timestamp_us" in schema_cols else ("time" if "time" in schema_cols else "timestamp")
        batch_parts = []
        for batch in pf.iter_batches(batch_size=_BATCH, columns=["price", "qty", time_col, "is_buyer_maker"]):
            batch_parts.append(_batch_to_seconds(batch, time_col=time_col))
        if not batch_parts: continue
        file_sec = pd.concat(batch_parts).groupby(level=0).agg(_AGG_OPS)
        all_second_dfs.append(file_sec)
        ticks_so_far += len(file_sec)
        if ticks_so_far >= max_ticks: break

    if not all_second_dfs: return None
    combined = pd.concat(all_second_dfs).groupby(level=0).agg(_AGG_OPS).sort_index().reset_index(drop=True)
    combined["mid_price"]    = combined["mid_sum"]  / combined["mid_count"].clip(lower=1)
    combined["vwap"]         = combined["total_pq"] / combined["total_qty"].clip(lower=1e-8)
    combined["log_return"]   = np.log(combined["mid_price"] / combined["mid_price"].shift(1).bfill())
    combined["realized_vol"] = combined["log_return"].rolling(20, min_periods=1).std().fillna(0)
    combined["buy_pressure"] = combined["buy_vol"] / (combined["buy_vol"] + combined["sell_vol"]).clip(lower=1e-8)
    combined["vwap_spread"]  = (combined["vwap"] - combined["mid_price"]).abs() / combined["mid_price"].clip(lower=1e-8)
    feats = combined[["log_return", "ofi", "realized_vol", "buy_pressure", "trade_count", "vwap_spread"]].values.astype(np.float32)
    return feats[:max_ticks]


def build_binance_multi_asset_data(log: ExperimentLogger, max_ticks=SAMPLE_TICKS):
    log.section("Loading & Engineering 3-Asset Binance Trade Streams")
    syms = ["symbol=BTCUSDT", "symbol=ETHUSDT", "symbol=SOLUSDT"]
    all_feats = []
    for s in syms:
        s_dir = DATA_DIR / s
        name = s.replace("symbol=", "")
        log.info(f"Streaming parquet chunks for {name}...")
        f = engineer_features_for_symbol(s_dir, max_ticks)
        all_feats.append(f)
        log.info(f"  {name}: {len(f):,} 1s bins")

    min_len = min(len(f) for f in all_feats)
    all_feats = [f[:min_len] for f in all_feats]
    combined_feats = np.concatenate(all_feats, axis=1) # [min_len, 18]

    # Primary predictive label: 5s forward return of BTCUSDT (3 classes)
    mid_returns = all_feats[0][:, 0]
    fut = np.convolve(mid_returns, np.ones(HORIZON) / HORIZON, mode="full")[HORIZON - 1:][:min_len]
    shifted = np.concatenate([fut[HORIZON:], np.repeat(fut[-1], HORIZON)])
    alpha_down = np.percentile(shifted, 33.33)
    alpha_up   = np.percentile(shifted, 66.67)
    labels = np.ones(min_len, dtype=np.int64)
    labels[shifted > alpha_up] = 2
    labels[shifted < alpha_down] = 0

    # 2σ realized-volatility regime mask on BTC
    import pandas as pd
    s = pd.Series(all_feats[0][:, 2])
    mu = s.rolling(REGIME_WINDOW, min_periods=1).mean()
    sg = s.rolling(REGIME_WINDOW, min_periods=1).std().fillna(0)
    regime = ((s > mu + REGIME_SIGMA * sg)).astype(np.float32).values

    log.metric("Total 1s bins", min_len)
    log.metric("Features", combined_feats.shape[1])
    log.metric("Regime %", float(regime.mean() * 100), unit="%")
    return combined_feats, labels, regime


def make_windows_binance(feats, labels, regime):
    n_total = len(feats) - SEQ_LEN - HORIZON
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)

    train_feats = feats[:n_train + SEQ_LEN]
    mu = train_feats.mean(0, keepdims=True)
    sd = train_feats.std(0, keepdims=True) + 1e-8
    x_norm = (feats - mu) / sd

    Xt = torch.from_numpy(x_norm)
    yt = torch.from_numpy(labels)
    rt = torch.from_numpy(regime)
    dt = torch.ones(len(feats), dtype=torch.float32) * 1.0 # 1s regular bins

    Xw = Xt.unfold(0, SEQ_LEN, 1).transpose(1, 2)
    dt_w = dt.unfold(0, SEQ_LEN, 1)
    rw = rt.unfold(0, SEQ_LEN, 1).max(dim=1)[0]
    yw = yt[SEQ_LEN - 1:]

    # Task 06/11: keep every window whose label is real (last HORIZON labels are padding)
    # and purge HORIZON windows before each split boundary so labels never peek across it.
    n = len(feats) - SEQ_LEN - HORIZON + 1
    tr_end = n_train
    va_end = n_train + n_val
    tr_hi, va_hi = tr_end - HORIZON, va_end - HORIZON

    return (Xw[:tr_hi], yw[:tr_hi], rw[:tr_hi], dt_w[:tr_hi],
            Xw[tr_end:va_hi], yw[tr_end:va_hi], rw[tr_end:va_hi], dt_w[tr_end:va_hi],
            Xw[va_end:n], yw[va_end:n], rw[va_end:n], dt_w[va_end:n])


def macro_f1(pred, true, n=3):
    f1s = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        d  = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
    return sum(f1s) / len(f1s)


# ── training stages ───────────────────────────────────────────────────────────
def stage_A(model, Xtr, ytr, dt_tr, Xva, yva, dt_va, device, log: ExperimentLogger, epochs=EPOCHS_A, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage A — SNN Gate Cheap Head ({epochs} epochs)")
    params = [p for n, p in model.named_parameters() if n.startswith("gate.") and "gate_head" not in n]
    opt = torch.optim.Adam(params, lr=LR)
    counts = torch.bincount(ytr)
    cw = (1.0 / counts.float())
    cw = (cw / cw.sum() * 3.0).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)

    ckpt = load_checkpoint(ckpt_path, log)
    start_ep = ckpt["epoch"] if ckpt and ckpt.get("stage") == "A" else 0
    best = ckpt.get("best_f1", -1.0) if ckpt and ckpt.get("stage") == "A" else -1.0
    snap = ckpt.get("best_state") if ckpt and ckpt.get("stage") == "A" else None
    if ckpt and ckpt.get("stage") == "A":
        load_state_compat(model, ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])

    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)
    for ep in range(start_ep, epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss = 0.0
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
                log.warn(f"Stage A: {guard.consecutive} consecutive non-finite updates; halting."); break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva), batch_size):
                Xb = Xva[i:i+batch_size].to(device).contiguous()
                dt_b = dt_va[i:i+batch_size].to(device).contiguous()
                c, _, _, _ = model.gate(model.encode(Xb), dt_b)
                preds.append(c.argmax(1).cpu())
        f1 = macro_f1(torch.cat(preds), yva.cpu())
        if f1 > best:
            best, snap = f1, {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint({"epoch": ep+1, "stage": "A", "model": model.state_dict(),
                         "optimizer": opt.state_dict(), "best_f1": best, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr), val_f1=f1, eta_s=eta)

    if snap is not None: load_state_compat(model, snap)
    return model


def stage_B(model, Xtr, ytr, rtr, Xva, yva, rva, device, log: ExperimentLogger, epochs=EPOCHS_B, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage B — Heavy Path (Magnetic Graph Laplacian + EWS) ({epochs} epochs)")
    params_heavy = list(model.amp.parameters()) + list(model.ham.parameters()) + list(model.head.parameters())
    opt_heavy = torch.optim.Adam(params_heavy, lr=LR)
    
    counts = torch.bincount(ytr)
    cw = (1.0 / counts.float())
    cw = (cw / cw.sum() * 3.0).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)

    ckpt = load_checkpoint(ckpt_path, log)
    start_ep = ckpt["epoch"] if ckpt and ckpt.get("stage") == "B" else 0
    best = ckpt.get("best_f1", -1.0) if ckpt and ckpt.get("stage") == "B" else -1.0
    snap = ckpt.get("best_state") if ckpt and ckpt.get("stage") == "B" else None
    if ckpt and ckpt.get("stage") == "B":
        load_state_compat(model, ckpt["model"])

    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)
    for ep in range(start_ep, epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            Xb = Xtr[b].to(device).contiguous()
            yb = ytr[b].to(device).contiguous()
            rb = rtr[b].to(device).contiguous()
            
            heavy, L_pde, H, S, L_cov, L_norm, _, _ = model.heavy_path(Xb)
            omegas = torch.linalg.eigvalsh(S)[:, -1]
            loss_ews = (torch.relu(omegas + 0.02) * (1.0 - rb) + torch.relu(0.02 - omegas) * rb).mean()
            
            loss_phys = LAMBDA_PDE * L_pde + 0.1 * L_cov + 0.1 * L_norm + 0.1 * loss_ews
            loss_head = ce(heavy, yb)
            loss = loss_phys + loss_head
            
            guard.step(loss, params_heavy, opt_heavy)
            if guard.halt:
                log.warn(f"Stage B: {guard.consecutive} consecutive non-finite updates; halting."); break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva), batch_size):
                Xb = Xva[i:i+batch_size].to(device).contiguous()
                h, _, _, _, _, _, _, _ = model.heavy_path(Xb)
                preds.append(h.argmax(1).cpu())
        f1 = macro_f1(torch.cat(preds), yva.cpu())
        if f1 > best:
            best, snap = f1, {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint({"epoch": ep+1, "stage": "B", "model": model.state_dict(),
                         "optimizer": {"heavy": opt_heavy.state_dict()}, "best_f1": best, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr), val_f1=f1, eta_s=eta)

    if snap is not None: load_state_compat(model, snap)
    return model


def stage_C(model, Xtr, ytr, rtr, dt_tr, Xva, yva, rva, dt_va, device, log: ExperimentLogger, epochs=EPOCHS_C, ckpt_path=CKPT_PATH, batch_size=BATCH_SIZE):
    log.section(f"Stage C — Multi-Asset Gate Arbitration ({epochs} epochs)")
    gate_params = list(model.gate.gate_head.parameters()) + [model.log_var_ce, model.log_var_regime]
    params = stage_c_mode(model, gate_params)   # Task 03
    model.heavy_frozen = True
    opt = torch.optim.Adam(params, lr=5e-3)
    ce  = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))

    ckpt = load_checkpoint(ckpt_path, log)
    start_ep = ckpt["epoch"] if ckpt and ckpt.get("stage") == "C" else 0
    best_score = ckpt.get("best_f1", -1e9) if ckpt and ckpt.get("stage") == "C" else -1e9
    snap = ckpt.get("best_state") if ckpt and ckpt.get("stage") == "C" else None
    if ckpt and ckpt.get("stage") == "C":
        load_state_compat(model, ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])

    t0 = time.time()
    guard = StepGuard(MAX_GRAD_NORM)
    for ep in range(start_ep, epochs):
        model.eval()   # Task 03: no sampling noise
        perm = torch.randperm(len(Xtr))
        ep_loss = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            Xb = Xtr[b].to(device).contiguous()
            dt_b = dt_tr[b].to(device).contiguous()
            Sb = model.encode(Xb)
            yb = ytr[b].to(device).contiguous()
            rb = rtr[b].to(device).contiguous()
            
            logits, g, _, _, _, _, gl, _, _, _, _ = model(Sb, Xb, dt_b)
            loss_ce = ce(logits, yb)
            loss_gate = g.mean()
            loss_regime = bce(gl * 4.0, rb)
            
            log_ce = torch.clamp(model.log_var_ce, min=-5.0, max=10.0)
            log_regime = torch.clamp(model.log_var_regime, min=-5.0, max=10.0)
            adaptive_ce = loss_ce * torch.exp(-log_ce) + log_ce
            adaptive_regime = loss_regime * torch.exp(-log_regime) + log_regime
            
            loss_gate_target = torch.relu(loss_gate - 0.10) * 5.0 + 0.2 * loss_gate
            loss = adaptive_ce + adaptive_regime + loss_gate_target
            guard.step(loss, params, opt)
            if guard.halt:
                log.warn(f"Stage C: {guard.consecutive} consecutive non-finite updates; halting."); break
            if not torch.isfinite(loss): continue
            ep_loss += loss.item() * len(b)

        model.eval()
        preds, meter = [], RateMeter()
        with torch.no_grad():
            for i in range(0, len(Xva), batch_size):
                Xb = Xva[i:i+batch_size].to(device).contiguous()
                dt_b = dt_va[i:i+batch_size].to(device).contiguous()
                lo, g, _, _, _, _, _, _, _, _, _ = model(model.encode(Xb), Xb, dt_b)
                preds.append(lo.argmax(1).cpu())
                meter.update(g)
        f1 = macro_f1(torch.cat(preds), yva.cpu())
        gate_rate = meter.rate
        score = f1 - (max(0.0, gate_rate - 0.12) * 5.0 + gate_rate * 0.1)

        if score > best_score:
            best_score = score
            snap = {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint({"epoch": ep+1, "stage": "C", "model": model.state_dict(),
                         "optimizer": opt.state_dict(), "best_f1": best_score, "best_state": snap},
                        ckpt_path, log if (ep+1) % 5 == 0 else None)
        if (ep + 1) % 5 == 0 or ep == start_ep or ep == epochs - 1:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (epochs - ep - 1)
            log.epoch(ep + 1, epochs, loss=ep_loss/len(Xtr), val_f1=f1, extra={"gate": gate_rate}, eta_s=eta)

    if snap is not None: load_state_compat(model, snap)
    return model


def evaluate_binance(model, X, y, r, dt_w, mode="learned", batch_size=1024):
    model.eval()
    preds, rates, pdes, omegas = [], [], [], []
    meter = RateMeter()
    H_herm_max = 0.0
    A_sym_accum, Phi_accum = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = X[i:i+batch_size].to(device).contiguous()
            dt_b = dt_w[i:i+batch_size].to(device).contiguous()
            Sb = model.encode(Xb)
            logits, g, rate, L_pde, H, S, _, _, _, A_sym, Phi_anti = model(Sb, Xb, dt_b, mode)
            preds.append(logits.argmax(1).cpu())
            meter.update(g)
            rates.append(rate.cpu())
            pdes.append(L_pde.cpu())
            omegas.append(torch.linalg.eigvalsh(S)[:, -1].cpu())
            
            herm = (H - H.conj().transpose(-1, -2)).abs().max().item()
            if herm > H_herm_max: H_herm_max = herm
            if mode != "cheap":
                A_sym_accum.append(A_sym.mean(dim=0).cpu())
                Phi_accum.append(Phi_anti.mean(dim=0).cpu())

    pred = torch.cat(preds)
    omega_all = torch.cat(omegas)
    mean_A = torch.stack(A_sym_accum).mean(dim=0) if A_sym_accum else torch.zeros(3, 3)
    mean_Phi = torch.stack(Phi_accum).mean(dim=0) if Phi_accum else torch.zeros(3, 3)

    return dict(
        acc      = (pred == y.cpu()).float().mean().item(),
        f1       = macro_f1(pred, y.cpu()),
        gate     = meter.rate,
        spike    = torch.tensor(rates).mean().item(),
        pde      = torch.tensor(pdes).mean().item(),
        herm     = H_herm_max,
        omega    = omega_all.mean().item(),
        crash_P  = (omega_all > 0).float().mean().item(),
        A_sym    = mean_A.numpy(),
        Phi_anti = mean_Phi.numpy()
    )


# ── LSTM baseline ─────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, HIDDEN, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 3)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])

def train_lstm(Xtr, ytr, Xva, yva, device, epochs=15):
    m = LSTMBaseline().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    counts = torch.bincount(ytr)
    cw = (1.0 / counts.float())
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
    if snap is not None:
        m.load_state_dict(snap)
    return m


# ── main entry point ──────────────────────────────────────────────────────────
def main(epochs_a=EPOCHS_A, epochs_b=EPOCHS_B, epochs_c=EPOCHS_C, max_ticks=SAMPLE_TICKS, batch_size=BATCH_SIZE):
    torch.manual_seed(SEED); np.random.seed(SEED)
    log = ExperimentLogger(OUT_DIR / "train.log", experiment_name="Exp C — 3-Asset Binance Basket")
    device = get_device(verbose=True)

    feats, labels, regime = build_binance_multi_asset_data(log, max_ticks=max_ticks)
    log.section("Partitioning Trailing Windows")
    Xtr, ytr, rtr, dt_tr, Xva, yva, rva, dt_va, Xte, yte, rte, dt_te = make_windows_binance(feats, labels, regime)

    mode_val = torch.mode(yte)[0].item()
    majority = (yte == mode_val).float().mean().item()
    log.metric("Train windows", len(Xtr))
    log.metric("Val windows",   len(Xva))
    log.metric("Test windows",  len(Xte))
    log.metric("Majority baseline", majority)

    model = MultiAssetQSPINN(n_in=N_FEATURES, d=PSI_DIM, n_assets=N_ASSETS).to(device)
    # Task 04: encoder thresholds from TRAIN rows only (windows have stride 1 -> rebuild the row series)
    model.encoder.fit(torch.cat([Xtr[:, 0], Xtr[-1, 1:]]).numpy())
    ckpt = load_checkpoint(CKPT_PATH, log)
    current_stage = ckpt["stage"] if ckpt else None

    if current_stage not in ["B", "C"]:
        model = stage_A(model, Xtr, ytr, dt_tr, Xva, yva, dt_va, device, log, epochs=epochs_a, batch_size=batch_size)
    if current_stage != "C":
        model = stage_B(model, Xtr, ytr, rtr, Xva, yva, rva, device, log, epochs=epochs_b, batch_size=batch_size)
    model = stage_C(model, Xtr, ytr, rtr, dt_tr, Xva, yva, rva, dt_va, device, log, epochs=epochs_c, batch_size=batch_size)

    log.section("Multi-Asset Ablation Suite")
    ablation_rows = []

    lstm_epochs = 1 if epochs_a <= 2 else 15
    lstm = train_lstm(Xtr, ytr, Xva, yva, device, epochs=lstm_epochs)
    lstm_preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 1024):
            Xb = Xte[i:i+1024].to(device).contiguous()
            lstm_preds.append(lstm(Xb).argmax(1).cpu())
    p = torch.cat(lstm_preds)
    ablation_rows.append(("LSTM baseline", (p == yte.cpu()).float().mean().item(), macro_f1(p, yte.cpu()), float('nan'), float('nan'), float('nan'), float('nan')))

    m_cheap = evaluate_binance(model, Xte, yte, rte, dt_te, mode="cheap")
    ablation_rows.append(("SNN only (cheap)", m_cheap["acc"], m_cheap["f1"], 0.0, m_cheap["herm"], m_cheap["omega"], m_cheap["crash_P"]))

    m_dense = evaluate_binance(model, Xte, yte, rte, dt_te, mode="dense")
    ablation_rows.append(("PINN only (always awake)", m_dense["acc"], m_dense["f1"], 1.0, m_dense["herm"], m_dense["omega"], m_dense["crash_P"]))

    m = evaluate_binance(model, Xte, yte, rte, dt_te, mode="learned")
    ablation_rows.append(("Q-SPINN Multi-Asset Hybrid", m["acc"], m["f1"], m["gate"], m["herm"], m["omega"], m["crash_P"]))

    log.table(
        headers=["Model", "Acc", "Macro-F1", "Gate", "||H-H†||", "ω(A_t)", "Crash P"],
        rows=[
            (name, f"{acc:.3f}", f"{f1:.3f}",
             "n/a" if gate != gate else f"{gate:.3f}",
             "n/a" if herm != herm else f"{herm:.2e}",
             "n/a" if omega != omega else f"{omega:.4f}",
             "n/a" if crash != crash else f"{crash*100:.1f}%")
            for name, acc, f1, gate, herm, omega, crash in ablation_rows
        ] + [("Majority baseline", f"{majority:.3f}", "—", "—", "—", "—", "—")],
        title="3-Asset Crypto Basket Ablation Results",
    )

    # Lead-lag analysis
    log.section("Magnetic Graph Lead-Lag Latency Analysis")
    phi = m["Phi_anti"]
    btc_eth_lead = phi[0, 1]
    btc_sol_lead = phi[0, 2]
    log.info(f"Learned Phase Flux BTC -> ETH: {btc_eth_lead:+.4f} rad")
    log.info(f"Learned Phase Flux BTC -> SOL: {btc_sol_lead:+.4f} rad")

    torch.save(model.state_dict(), OUT_DIR / "qspinn_binance_multi_asset.pt")
    log.info(f"Weights saved → {OUT_DIR / 'qspinn_binance_multi_asset.pt'}")

    rows_md = "".join([f"| {name} | {acc:.3f} | {f1:.3f} | {'n/a' if gate!=gate else f'{gate:.3f}'} | {'n/a' if herm!=herm else f'{herm:.2e}'} | {'n/a' if omega!=omega else f'{omega:.4f}'} | {'n/a' if crash!=crash else f'{crash*100:.1f}%'} |\n" for name, acc, f1, gate, herm, omega, crash in ablation_rows])
    md = f"""# Experiment C — Multi-Asset Q-SPINN on Binance Crypto Basket

**Date:** {datetime.now().isoformat()}  
**Assets:** BTCUSDT, ETHUSDT, SOLUSDT (18 features total)  
**Prediction Horizon:** 5 seconds  

## Ablation Results

| Model | Accuracy | Macro-F1 | Gate Rate | Hermiticity ‖H−H†‖ | $\\omega(A_t)$ (EWS Margin) | Crash Prob |
|---|---|---|---|---|---|---|
{rows_md}| **Majority baseline** | **{majority:.3f}** | — | — | — | — | — |

## Magnetic Graph Laplacian & Lead-Lag Phase Analysis

The directed phase flux matrix $\\Phi_{{ij}}$ captures empirical lead-lag latency:
- **BTC $\\to$ ETH Phase Flux:** `{btc_eth_lead:+.4f}` rad
- **BTC $\\to$ SOL Phase Flux:** `{btc_sol_lead:+.4f}` rad

A positive phase indicates that BTC microstructure features lead altcoin quote adjustments, confirming Objective O2.
"""
    (OUT_DIR / "RESULTS.md").write_text(md, encoding="utf-8")
    if CKPT_PATH.exists(): CKPT_PATH.unlink()

    summary = {
        "acc": m["acc"],
        "f1": m["f1"],
        "gate": m["gate"],
        "btc_eth_lead": float(btc_eth_lead),
        "btc_sol_lead": float(btc_sol_lead),
        "weights": str(OUT_DIR / "qspinn_binance_multi_asset.pt")
    }
    log.done(summary=summary)
    return summary


if __name__ == "__main__":
    main()
