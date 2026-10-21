"""
experiments/fi2010_setup2.py
============================
FI-2010 "Setup 2" with explicit provenance (Task 07).

Setup 2 = train on days 1-7, test on days 8, 9, 10 (anchored).  In the public NoAuction
Z-score release this is  Train_..._CF_7  (once!)  and  Test_..._CF_7 / CF_8 / CF_9  (one
distinct day each).  The previous pipeline concatenated Train CF1..CF9 - which contain the
SAME days again and again - and Test CF1..CF9, and dropped every stock/day boundary.

Each file concatenates  (days x 5 stocks)  contiguous segments with no id column.  We recover
the boundaries by the only robust cue available - the largest jumps of the (Z-scored) mid price
at segment joins - choosing exactly K = n_expected_segments - 1 jumps, and we RECORD how
confident that choice is (ratio of the K-th to the (K+1)-th jump).  The teammate must look at
`segment_audit.json`; a low ratio means boundaries must be supplied from the dataset authors'
per-stock/day counts instead (see docs/RUNBOOK_TEAMMATE.md).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

N_STOCKS = 5
TRAIN_DAYS = 7
TEST_DAYS = {"test_d8": 7, "test_d9": 8, "test_d10": 9}      # partition -> Test CF index
MIN_CONFIDENCE = 1.5


def detect_segments(mid: np.ndarray, n_segments: int):
    """
    Return (segment_id[int64, len(mid)], confidence).  Boundaries are the K=n_segments-1
    largest |delta mid| positions.  confidence = K-th largest jump / (K+1)-th largest jump.
    """
    mid = np.asarray(mid, dtype=np.float64)
    if n_segments <= 1:
        return np.zeros(len(mid), dtype=np.int64), float("inf")
    jumps = np.abs(np.diff(mid))
    order = np.argsort(-jumps)
    K = n_segments - 1
    if K >= len(jumps):
        raise ValueError("more segments requested than rows")
    cut = np.sort(order[:K]) + 1                       # first row of every new segment
    kth, next_ = jumps[order[K - 1]], jumps[order[K]] if K < len(order) else 0.0
    conf = float(kth / max(next_, 1e-12))
    seg = np.zeros(len(mid), dtype=np.int64)
    for c in cut:
        seg[c:] += 1
    return seg, conf


def segments_from_lengths(lengths: List[int], n_rows: int, n_expected: int):
    """
    AUTHORITATIVE segment ids from verified per-stock/day row counts (in file order).
    Validates: right number of segments, all positive, and exact coverage of the file.
    """
    lengths = [int(x) for x in lengths]
    if len(lengths) != n_expected:
        raise ValueError(f"expected {n_expected} segments, got {len(lengths)}")
    if min(lengths) <= 0:
        raise ValueError("segment lengths must be positive")
    if sum(lengths) != n_rows:
        raise ValueError(f"segment lengths sum to {sum(lengths)} but the file has {n_rows} rows")
    return np.repeat(np.arange(n_expected, dtype=np.int64), lengths)


def build_setup2_frames(train_cf7: np.ndarray, tests: Dict[str, np.ndarray],
                        segment_lengths: Optional[Dict[str, List[int]]] = None,
                        allow_heuristic: bool = False):
    """
    train_cf7 / tests[name]: arrays [rows, 149] (144 features + 5 labels) for Train CF7 and
    Test CF7/8/9.  Returns dict(partition -> dict(X, y_all, seg, day, conf, provenance)).

    FAIL CLOSED: stock/day boundaries come from `segment_lengths` (verified per-segment row counts) or
    the build refuses.  The jump heuristic is only used with allow_heuristic=True and every partition is
    then stamped provenance="heuristic" (a large in-session move can be mistaken for a join and a small
    true join missed - the review reproduced this with confidence ~2000).
    """
    if segment_lengths is None and not allow_heuristic:
        raise RuntimeError(
            "Authoritative FI-2010 segment lengths are required (segment_lengths.json: partition -> list of "
            "per-stock/day row counts). Pass allow_heuristic=True / --allow-heuristic ONLY for exploration; "
            "heuristic datasets are refused by the training loader.")
    out = {}
    plan = [("train", train_cf7, TRAIN_DAYS * N_STOCKS)] + \
           [(name, arr, N_STOCKS) for name, arr in tests.items()]
    offset = 0
    for name, arr, n_seg in plan:
        if segment_lengths is not None and name in segment_lengths:
            seg = segments_from_lengths(segment_lengths[name], len(arr), n_seg)
            conf, prov = float("nan"), "authoritative"
        elif allow_heuristic:
            mid = (arr[:, 0] + arr[:, 2]) / 2.0            # z-scored best ask / best bid price
            seg, conf = detect_segments(mid, n_seg)
            prov = "heuristic"
        else:
            raise RuntimeError(f"segment lengths missing for partition {name!r}")
        out[name] = dict(X=arr[:, :144].astype(np.float32), y_all=arr[:, 144:149].astype(np.int64) - 1,
                         seg=seg + offset, day=seg // N_STOCKS, conf=conf, n_segments=n_seg, provenance=prov)
        offset += n_seg                                  # globally unique ids
    return out


def audit(frames) -> dict:
    rep = {}
    for name, f in frames.items():
        counts = np.bincount(f["seg"] - f["seg"].min())
        rep[name] = dict(rows=int(len(f["X"])), n_segments=int(f["n_segments"]),
                         provenance=f["provenance"],
                         boundary_confidence=None if f["provenance"] == "authoritative" else f["conf"],
                         confident=bool(f["provenance"] == "authoritative"),   # heuristic never counts as confident
                         segment_lengths=counts.tolist())
    return rep


def save_setup2(frames, out_dir: Path):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, f in frames.items():
        df = pd.DataFrame(f["X"], columns=[f"feature_{i}" for i in range(144)])
        for j, c in enumerate(["label_k10", "label_k20", "label_k30", "label_k50", "label_k100"]):
            df[c] = f["y_all"][:, j]
        df["partition"] = name
        df["provenance"] = f["provenance"]
        df["segment_id"] = f["seg"]
        df["day_in_partition"] = f["day"]
        df["row_in_partition"] = np.arange(len(df))
        parts.append(df)
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / "fi2010_setup2.parquet", compression="zstd")
    (out_dir / "segment_audit.json").write_text(json.dumps(audit(frames), indent=2), encoding="utf-8")


def load_setup2(path: Path, label_col: str, val_days: int = 1, allow_heuristic: bool = False):
    """
    Returns dict(train/val/test -> dict(X, y, seg)).  Validation = the last `val_days` training
    day(s) (segment-aware, so no window crosses into train); test = the three distinct test days.
    """
    import pandas as pd
    df = pd.read_parquet(path)
    if not allow_heuristic and (df["provenance"] != "authoritative").any():
        raise RuntimeError("Setup 2 parquet has heuristic (unverified) stock/day boundaries; rebuild with "
                           "authoritative segment lengths (procure_fi2010.py --setup2 --segments segment_lengths.json).")
    feat = [f"feature_{i}" for i in range(144)]
    tr = df[df.partition == "train"]
    last_day = tr.day_in_partition.max()
    is_val = tr.day_in_partition > last_day - val_days
    te = df[df.partition.isin(list(TEST_DAYS))]

    def pack(d):
        return dict(X=d[feat].values.astype(np.float32), y=d[label_col].values.astype(np.int64),
                    seg=d.segment_id.values.astype(np.int64))
    return dict(train=pack(tr[~is_val]), val=pack(tr[is_val]), test=pack(te))


def segmented_regime_mask(X: np.ndarray, seg: np.ndarray, window: int, sigma: float) -> np.ndarray:
    """Backward realised-vol flag computed independently inside every segment (no leakage across joins)."""
    import pandas as pd
    out = np.zeros(len(X), dtype=np.float32)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        mid = (X[idx, 0] + X[idx, 2]) / 2.0
        ret = pd.Series(mid).diff().fillna(0)
        rv = ret.rolling(20, min_periods=1).std().fillna(0)
        mu = rv.rolling(window, min_periods=1).mean()
        sg = rv.rolling(window, min_periods=1).std().fillna(0)
        out[idx] = (rv > mu + sigma * sg).astype(np.float32).values
    return out
