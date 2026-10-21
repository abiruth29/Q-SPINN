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
