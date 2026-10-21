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


def _arr(n_seg, rng, seg_len=120):
    m = _synthetic(n_seg, seg_len, rng)
    a = np.zeros((len(m), 149)); a[:, 0] = m; a[:, 2] = m; a[:, 144:] = 1
    return a


def test_frames_require_authoritative_lengths_and_have_unique_ids():
    import pytest
    from fi2010_setup2 import segments_from_lengths
    rng = np.random.default_rng(2)
    tr, t8, t9, t10 = _arr(35, rng), _arr(5, rng), _arr(5, rng), _arr(5, rng)
    tests = {"test_d8": t8, "test_d9": t9, "test_d10": t10}
    with pytest.raises(RuntimeError):                                  # fail closed without provenance
        build_setup2_frames(tr, tests)
    lens = {"train": [120] * 35, "test_d8": [120] * 5, "test_d9": [120] * 5, "test_d10": [120] * 5}
    fr = build_setup2_frames(tr, tests, segment_lengths=lens)
    ids = [set(np.unique(f["seg"])) for f in fr.values()]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    assert fr["train"]["day"].max() == 6 and all(f["provenance"] == "authoritative" for f in fr.values())
    assert all(r["confident"] for r in audit(fr).values())
    for bad in ([120] * 34, [120] * 34 + [119], [120] * 34 + [0]):    # wrong count / wrong coverage / non-positive
        with pytest.raises(ValueError):
            segments_from_lengths(bad, 35 * 120, 35)


def test_heuristic_is_stamped_and_loader_refuses_it(tmp_path):
    import pytest
    pytest.importorskip("pyarrow")
    from fi2010_setup2 import save_setup2, load_setup2
    rng = np.random.default_rng(3)
    tests = {"test_d8": _arr(5, rng), "test_d9": _arr(5, rng), "test_d10": _arr(5, rng)}
    fr = build_setup2_frames(_arr(35, rng), tests, allow_heuristic=True)
    assert all(f["provenance"] == "heuristic" for f in fr.values())
    assert not any(r["confident"] for r in audit(fr).values())          # never "confident" without provenance
    save_setup2(fr, tmp_path)
    with pytest.raises(RuntimeError):
        load_setup2(tmp_path / "fi2010_setup2.parquet", "label_k10")
    assert len(load_setup2(tmp_path / "fi2010_setup2.parquet", "label_k10", allow_heuristic=True)["train"]["X"]) > 0


def test_heuristic_misplaces_boundary_when_shock_exceeds_true_join():
    """Review counterexample: in-session shock larger than the true (small) join -> wrong cut, high 'confidence'."""
    rng = np.random.default_rng(0)
    mid = np.cumsum(rng.normal(0, .001, 400))
    mid[200:] += 0.05            # small TRUE join at row 200
    mid[100:] += 5.0             # big in-session move at row 100
    seg, conf = detect_segments(mid, 2)
    assert np.flatnonzero(np.diff(seg))[0] + 1 == 100 and conf > 50     # confidently wrong -> why provenance is required


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
