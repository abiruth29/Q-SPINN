"""
experiments/checkpoint.py
==========================
Checkpoint save / load / validate utilities shared by all experiment scripts.

Saves after every epoch so a crash or OOM can always be resumed from the
last completed epoch rather than restarting from scratch.

Checkpoint format (a plain dict, stored as a .pt file):
  {
    "epoch":      int,          # last completed epoch (1-indexed)
    "stage":      str,          # "A", "B", or "C"
    "model":      state_dict,
    "optimizer":  state_dict,
    "best_f1":    float,
    "best_state": state_dict,
    "history":    list[dict],
  }
"""

import hashlib
import json
from pathlib import Path
import torch
from qspinn_core import PROTOCOL_VERSION


_FINGERPRINT = "unset"


def fingerprint_of(**fields) -> str:
    """Stable short hash of everything that defines a run: data summary, split, horizon, seed, epochs, objective..."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def set_run_fingerprint(fp: str) -> None:
    """Drivers call this once the run configuration + data are known.  Checkpoints are only resumed on an exact match."""
    global _FINGERPRINT
    _FINGERPRINT = fp


def save_checkpoint(ckpt: dict, path: Path, log=None) -> None:
    """
    Atomically save a checkpoint.

    Writes to a .tmp file first, then renames it, so a crash mid-write
    never produces a half-written (corrupted) checkpoint.
    """
    ckpt = dict(ckpt, protocol=PROTOCOL_VERSION, fingerprint=_FINGERPRINT)
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)          # atomic on all major OS
    if log:
        log.info(f"Checkpoint saved  → {path.name}  (epoch {ckpt['epoch']}, stage {ckpt['stage']})")


def load_checkpoint(path: Path, log=None) -> dict | None:
    """
    Load a checkpoint with corruption detection.

    Returns None if the file doesn't exist or is corrupted, allowing the
    caller to start fresh rather than crash.
    """
    if not path.exists():
        return None

    try:
        ckpt = torch.load(path, weights_only=False)

        # Validate expected keys
        required = {"epoch", "stage", "model", "best_state"}
        missing  = required - ckpt.keys()
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")

        if ckpt.get("protocol") != PROTOCOL_VERSION:
            msg = (f"Checkpoint '{path.name}' has protocol {ckpt.get('protocol')!r} != {PROTOCOL_VERSION!r}; "
                   "ignoring it and starting fresh (never load legacy tensors into the new architecture).")
            (log.warn(msg) if log else print(f"\n⚠  {msg}\n"))
            return None
        if ckpt.get("fingerprint") != _FINGERPRINT:
            msg = (f"Checkpoint '{path.name}' was written by a different run configuration/data "
                   f"(fingerprint {ckpt.get('fingerprint')!r} != {_FINGERPRINT!r}); ignoring it and starting fresh. "
                   "(Typical cause: a smoke test, another seed/horizon/epoch budget, or different data.)")
            (log.warn(msg) if log else print(f"\n⚠  {msg}\n"))
            return None
        if log:
            log.info(f"Resuming from checkpoint: epoch {ckpt['epoch']}, stage {ckpt['stage']}")
        return ckpt

    except Exception as e:
        msg = (
            f"Checkpoint at '{path}' appears corrupted ({type(e).__name__}: {e}).\n"
            f"  → Rename or delete it and re-run to start from scratch:\n"
            f"      del \"{path}\"\n"
            f"  → Or restore the previous backup if one exists:\n"
            f"      copy \"{path.with_suffix('.bak')}\" \"{path}\""
        )
        if log:
            log.warn(msg)
        else:
            print(f"\n⚠  {msg}\n")
        return None


def guard_nan(loss: torch.Tensor, epoch: int, stage: str, log=None) -> bool:
    """
    Return True (and warn) if the loss is NaN or Inf — the caller should break.

    NaN loss almost always means one of:
      1. Learning rate too high — try halving LR and resuming from the last checkpoint.
      2. A dead gate logit that wasn't squashed — tanh squashing in gate_head fixes this.
      3. Complex arithmetic overflow in the Hamiltonian — reduce LAMBDA_PDE.
    """
    if torch.isnan(loss) or torch.isinf(loss):
        msg = (
            f"Loss is {loss.item()} at epoch {epoch}, stage {stage}. "
            f"Training halted. Last checkpoint is intact.\n"
            f"  Suggestions:\n"
            f"    1. Halve the learning rate (LR) and resume.\n"
            f"    2. Reduce LAMBDA_PDE (currently controls Schrödinger residual weight).\n"
            f"    3. Check GPU memory — OOM sometimes silently corrupts tensors on some drivers."
        )
        if log:
            log.warn(msg)
        else:
            print(f"\n⚠  {msg}\n")
        return True
    return False
