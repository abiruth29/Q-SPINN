import numpy as np, pytest
from live_contract import make_contract, contracts_match, LiveWindow, save_contract, load_contract

TRAIN = make_contract(20, 144, "fi2010_authors", "dataset_zscore", "equity")


def test_matching_and_mismatching_contracts():
    assert contracts_match(TRAIN, dict(TRAIN))[0]
    live = make_contract(2, 88, "binance_depth_extractor", "online_zscore", "crypto")
    ok, bad = contracts_match(TRAIN, live)
    assert not ok and set(bad) == {"seq_len", "n_features", "feature_source", "normalisation", "market"}
    assert contracts_match(None, TRAIN)[0] is False


def test_live_window_only_ready_when_full_and_checks_width():
    w = LiveWindow(20, 144)
    for i in range(19):
        w.push(np.zeros(144)); assert not w.ready
    with pytest.raises(RuntimeError): w.tensor()
    w.push(np.ones(144)); assert w.ready and tuple(w.tensor().shape) == (1, 20, 144)
    with pytest.raises(ValueError): w.push(np.zeros(88))


def test_contract_roundtrip(tmp_path):
    save_contract(tmp_path / "c.json", TRAIN); assert load_contract(tmp_path / "c.json") == TRAIN
    assert load_contract(tmp_path / "none.json") is None
