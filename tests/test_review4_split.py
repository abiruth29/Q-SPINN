"""Review 4 item 1: leak-safe crypto split and scaler support on a 500-row synthetic fixture."""
import numpy as np
import exp_C_multi_asset_binance.train as C
from binance_align import forward_labels


def _fixture(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 18)).astype(np.float32)


def test_plan_label_support_is_disjoint_across_splits():
    n_len = 500; sp = C.split_plan(n_len); H, L = C.HORIZON, C.SEQ_LEN
    sup = lambda i: (i + L - 1 + 1, i + L - 1 + H)            # returns rows (t, t+H]
    tr_max = sup(sp["tr_hi"] - 1)[1]; va_min = sup(sp["tr_end"])[0]
    va_max = sup(sp["va_hi"] - 1)[1]; te_min = sup(sp["va_end"])[0]
    assert tr_max < va_min and va_max < te_min
    assert tr_max < sp["label_support_end"] <= va_min
    assert sp["scaler_end"] <= sp["tr_hi"] - 1 + L            # exactly the last train input row + 1
    assert sp["scaler_end"] <= sp["tr_end"] + L - H - 1


def test_scaler_never_reads_validation_only_rows():
    f = _fixture(); lab = np.ones(len(f), dtype=np.int64); reg = np.zeros(len(f), dtype=np.float32)
    sp = C.split_plan(len(f))
    a = C.make_windows_binance(f, lab, reg)[0]
    f2 = f.copy(); f2[sp["scaler_end"]:] += 1e3                # rows no train window reads
    b = C.make_windows_binance(f2, lab, reg)[0]
    assert np.allclose(a.numpy(), b.numpy())                    # train windows (and thus mu/sd) unchanged


def test_thresholds_ignore_returns_no_train_label_reads():
    r = np.random.default_rng(1).normal(size=500); sp = C.split_plan(500)
    base = forward_labels(r, C.HORIZON, sp["label_support_end"])[1]
    r2 = r.copy(); r2[sp["label_support_end"]:] *= 100
    assert forward_labels(r2, C.HORIZON, sp["label_support_end"])[1] == base


def test_train_windows_count_matches_plan():
    f = _fixture(); lab = np.ones(len(f), dtype=np.int64); reg = np.zeros(len(f), dtype=np.float32)
    out = C.make_windows_binance(f, lab, reg); sp = C.split_plan(len(f))
    assert len(out[0]) == sp["tr_hi"] and len(out[4]) == sp["va_hi"] - sp["tr_end"] and len(out[8]) == sp["n"] - sp["va_end"]
