"""Task 07: Setup 2 provenance, segment-aware windows, no cross-segment leakage."""
import numpy as np
from fi2010_setup2 import detect_segments, build_setup2_frames, segmented_regime_mask, audit
from qspinn_core import window_starts


def _synthetic(n_seg, seg_len, rng):
    # each segment is a random walk starting from a very different level -> big jump at joins
    parts = [np.cumsum(rng.normal(0, .01, seg_len)) + 10 * i for i in range(n_seg)]
    return np.concatenate(parts)


def test_detect_segments_recovers_boundaries():
    rng = np.random.default_rng(0)
    mid = _synthetic(35, 200, rng)
    seg, conf = detect_segments(mid, 35)
    assert seg.max() == 34
    assert (np.bincount(seg) == 200).all()
    assert conf > 1.5


def test_low_confidence_flagged_when_no_joins():
    rng = np.random.default_rng(1)
    mid = np.cumsum(rng.normal(0, 1, 5000))             # no real boundaries
    _, conf = detect_segments(mid, 35)
    assert conf < 1.5


def test_frames_have_unique_ids_and_day_index():
    rng = np.random.default_rng(2)
    def arr(n_seg):
        m = _synthetic(n_seg, 120, rng)
        a = np.zeros((len(m), 149)); a[:, 0] = m; a[:, 2] = m; a[:, 144:] = 1
        return a
    fr = build_setup2_frames(arr(35), {"test_d8": arr(5), "test_d9": arr(5), "test_d10": arr(5)})
    ids = [set(np.unique(f["seg"])) for f in fr.values()]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    assert fr["train"]["day"].max() == 6
    assert all(r["confident"] for r in audit(fr).values())


def test_windows_never_cross_segments_and_purge_horizon():
    seg = np.repeat(np.arange(4), 100)
    st = window_starts(400, 20, seg, horizon=10)
    assert (seg[st] == seg[st + 19]).all()
    ends = np.array([99, 199, 299, 399])
    last = st + 19
    seg_end = ends[seg[last]]
    assert (last + 10 <= seg_end).all()


def test_regime_mask_has_no_cross_segment_leakage():
    rng = np.random.default_rng(3)
    X = np.zeros((400, 144), dtype=np.float32)
    X[:200, 0] = X[:200, 2] = np.cumsum(rng.normal(0, .01, 200))
    X[200:, 0] = X[200:, 2] = 500 + np.cumsum(rng.normal(0, .01, 200))   # huge join jump
    seg = np.repeat([0, 1], 200)
    r = segmented_regime_mask(X, seg, 50, 2.0)
    assert r[200] == 0.0        # the jump itself must not be flagged as a volatility regime
