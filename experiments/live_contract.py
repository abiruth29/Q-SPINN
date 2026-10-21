"""
experiments/live_contract.py
============================
Task 17: one explicit input contract shared by training and live inference.

A model may be used for *product* inference only when the contract it was trained under equals the
contract the live pipeline produces.  The old pipeline fed FI-2010 equity weights (20-step windows,
144 dataset-normalised features) with a 2-step window of online-normalised crypto features; that run
is allowed only as a clearly labelled DEMONSTRATION.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

CONTRACT_KEYS = ("seq_len", "n_features", "feature_source", "normalisation", "market", "protocol")


def make_contract(seq_len: int, n_features: int, feature_source: str, normalisation: str, market: str) -> dict:
    from qspinn_core import PROTOCOL_VERSION
    return dict(seq_len=seq_len, n_features=n_features, feature_source=feature_source,
                normalisation=normalisation, market=market, protocol=PROTOCOL_VERSION)


def save_contract(path: Path, contract: dict) -> None:
    Path(path).write_text(json.dumps(contract, indent=2))


def load_contract(path: Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def contracts_match(train: dict | None, live: dict) -> tuple[bool, list]:
    """(ok, list of mismatching keys). A missing training contract never matches."""
    if train is None:
        return False, ["<no training contract found>"]
    bad = [k for k in CONTRACT_KEYS if train.get(k) != live.get(k)]
    return (not bad), bad


class LiveWindow:
    """Rolling window of exactly `seq_len` feature vectors; `ready` only once full (no duplicate padding)."""

    def __init__(self, seq_len: int, n_features: int):
        self.seq_len, self.n_features = seq_len, n_features
        self.buf: deque = deque(maxlen=seq_len)

    def push(self, feats: np.ndarray) -> None:
        feats = np.asarray(feats, dtype=np.float32)
        if feats.shape != (self.n_features,):
            raise ValueError(f"expected {self.n_features} features, got {feats.shape}")
        self.buf.append(feats)

    @property
    def ready(self) -> bool:
        return len(self.buf) == self.seq_len

    def tensor(self):
        import torch
        if not self.ready:
            raise RuntimeError("window not full")
        return torch.from_numpy(np.stack(self.buf)).unsqueeze(0)


# ── truthful live compute / omega contract (review 4, item 4) ─────────────────
class OmegaState:
    """Last FRESHLY computed omega and how many ticks ago it was computed."""
    def __init__(self):
        self.value = None; self.age = 0

    def fresh(self, v: float):
        self.value, self.age = float(v), 0

    def stale(self):
        if self.value is not None:
            self.age += 1


def live_step(model, x, dt, state: OmegaState, omega_mode: str = "gated"):
    """
    One live tick.
      omega_mode="gated"  : gate first.  Heavy path (and therefore omega) is computed ONLY when the gate routes the
                            row; otherwise the cheap prediction is used and omega is reported as 'cached' (last fresh
                            value + age) or 'unavailable' (never computed).  No routed row => zero heavy calls.
      omega_mode="always" : heavy path on every tick, omega always fresh.  NO compute-saving claim may be made.
    Returns a dict including `heavy_calls`, `omega`, `omega_status` in {fresh, cached, unavailable} and
    `latency_ms` measured end-to-end (encode + gate + routed heavy work).
    """
    import time as _t
    import torch
    if omega_mode not in ("gated", "always"):
        raise ValueError(omega_mode)
    t0 = _t.perf_counter()
    with torch.no_grad():
        model.eval()
        sp = model.encode(x)
        cheap, g, _, _ = model.gate(sp, dt)
        routed = True if omega_mode == "always" else bool(g[0] > 0.5)
        heavy_calls = 0
        if routed:
            heavy, _, _, S, _, _ = model.heavy_path(x)
            state.fresh(torch.linalg.eigvalsh(S)[:, -1].item())
            logits = heavy; heavy_calls = 1; status = "fresh"
        else:
            state.stale()
            logits = cheap
            status = "cached" if state.value is not None else "unavailable"
    return dict(pred=int(logits.argmax(1)[0]), routed=routed, heavy_calls=heavy_calls,
                omega=state.value, omega_status=status, omega_age_ticks=state.age if status != "fresh" else 0,
                omega_mode=omega_mode, latency_ms=(_t.perf_counter() - t0) * 1e3)
