import numpy as np
import importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ld", ROOT / "experiments/exp_C_multi_asset_binance/label_diagnostics.py")
LD = importlib.util.module_from_spec(spec); spec.loader.exec_module(LD)


def test_random_walk_has_no_predictability_and_no_flat_class():
    rng = np.random.default_rng(0); n = 60000
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    d = LD.diagnose(mid, np.zeros(n))
    assert abs(d["lookup_acc_test"] - 1 / 3) < 0.03                      # nothing to exploit
    assert d["share_windows_exact_zero_price_change"] == 0.0
    assert d["label_threshold_usd_over_horizon"][1] > 1e-3 and all(abs(x) < 0.05 for x in d["return_autocorr_lag1_to_5"])


def test_stale_flat_price_dominates_and_is_detected():
    rng = np.random.default_rng(1); n = 60000
    steps = rng.normal(0, 1e-4, n); stale = rng.random(n) < 0.7           # 70% stale seconds
    steps[stale] = 0.0
    mid = 100 * np.exp(np.cumsum(steps))
    d = LD.diagnose(mid, stale.astype(float))
    assert d["share_windows_exact_zero_price_change"] > 0.15
    assert d["label_threshold_usd_over_horizon"][1] < 0.05               # "flat" band ~ zero dollars
    assert d["zero_return_share_all"] > 0.6 and d["share_label_windows_with_any_stale"] > 0.9
    assert abs(d["lookup_acc_test"] - 1 / 3) < 0.03                      # tertile labels are balanced by construction
