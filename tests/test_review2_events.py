"""Review-2 rank 16: event-based warning evaluation on a realised-onset clock, segment-safe."""
import numpy as np
import routing as R


def test_known_shock_and_alert_ten_steps_earlier_gives_ten_step_lead():
    n = 300; ret = np.zeros(n); ret[100] = 5.0
    onsets = R.realised_onsets(np.abs(ret), thr=1.0, refractory=20)
    assert list(onsets) == [100]
    alert = np.zeros(n, bool); alert[90] = True
    m = R.event_alert_metrics(alert, onsets, tau=20)
    assert m["events"] == 1 and m["detected"] == 1 and m["mean_lead_rows"] == 10.0 and m["false_alerts"] == 0


def test_future_label_alert_is_not_mistaken_for_event_clock():
    """Alert matching future-window labels (positive from row 80..99) detects the row-100 shock with lead >= 1."""
    n = 300; ret = np.zeros(n); ret[100] = 5.0
    onsets = R.realised_onsets(np.abs(ret), 1.0, 20)
    alert = np.zeros(n, bool); alert[80:100] = True
    assert R.event_alert_metrics(alert, onsets, tau=20)["detected"] == 1


def test_alert_cannot_cross_segment_seam():
    seg = np.r_[np.zeros(100, int), np.ones(100, int)]
    ret = np.zeros(200); ret[105] = 5.0                       # shock early in segment 1
    onsets = R.realised_onsets(np.abs(ret), 1.0, 10, seg)
    alert = np.zeros(200, bool); alert[98] = True             # alert in segment 0, 7 rows before the shock (across the seam)
    m = R.event_alert_metrics(alert, onsets, tau=20, seg=seg)
    assert m["detected"] == 0 and m["false_alerts"] == 1      # not credited, and counted as false


def test_false_alert_episodes_and_rate():
    n = 1000; ret = np.zeros(n); ret[500] = 5.0
    onsets = R.realised_onsets(np.abs(ret), 1.0, 20)
    alert = np.zeros(n, bool); alert[100:110] = True; alert[300] = True; alert[490:495] = True
    m = R.event_alert_metrics(alert, onsets, tau=20, rows_per_unit_time=100)
    assert m["alert_episodes"] == 3 and m["false_alerts"] == 2 and m["detected"] == 1
    assert abs(m["false_alerts_per_unit_time"] - 2 / (n / 100)) < 1e-12


def test_refractory_merges_runs_of_one_episode():
    ret = np.zeros(100); ret[40:45] = 5.0; ret[80] = 5.0
    assert list(R.realised_onsets(np.abs(ret), 1.0, 10)) == [40, 80]
