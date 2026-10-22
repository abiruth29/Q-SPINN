"""Event-based warning evaluation: one alarm policy, an explicit at-risk clock, segment-safe, censoring explicit."""
import numpy as np
import routing as R


def _ret(n, *rows, mag=5.0):
    r = np.zeros(n)
    for x in rows: r[x] = mag
    return r


def test_known_shock_and_alarm_ten_steps_earlier_gives_ten_step_lead():
    onsets = R.realised_onsets(np.abs(_ret(300, 100)), 1.0, 20)
    assert list(onsets) == [100]
    alert = np.zeros(300, bool); alert[90] = True
    m = R.event_alert_metrics(alert, onsets, tau=20)
    assert m["events_evaluable"] == 1 and m["detected"] == 1 and m["mean_lead_rows"] == 10.0 and m["false_alarms"] == 0


def test_sustained_alarm_is_one_coherent_policy():
    """Review 3: alert on 10..50, event at 50, tau=20.  Starts-only: alarm@10 misses AND is false (coherent).
    With renewal every tau: alarm@30 detects (lead 20); alarm@10 false; alarm@50 follow window censored/false-free."""
    onsets = np.array([50]); alert = np.zeros(100, bool); alert[10:51] = True
    starts = R.event_alert_metrics(alert, onsets, tau=20, renew_every=None)
    assert starts["alarms"] == 1 and starts["detected"] == 0 and starts["false_alarms"] == 1
    renew = R.event_alert_metrics(alert, onsets, tau=20)                       # default renewal every tau
    assert renew["detected"] == 1 and renew["mean_lead_rows"] == 20.0
    assert renew["alarms"] == 3                                                # alarms at 10, 30, 50
    assert renew["false_alarms"] == 2                                          # @10 (no event in (10,30]) and @50 (post-event); @30 is the hit


def test_renewed_alarm_after_refractory_and_multiple_events():
    n = 400; onsets = np.array([100, 300])
    alert = np.zeros(n, bool); alert[85:95] = True; alert[280:290] = True; alert[200] = True
    m = R.event_alert_metrics(alert, onsets, tau=20)
    assert m["events_evaluable"] == 2 and m["detected"] == 2 and m["false_alarms"] == 1
    assert m["mean_lead_rows"] == np.mean([100 - 85, 300 - 280])


def test_warmup_event_is_censored_not_a_miss():
    n = 200; onsets = np.array([5, 100])
    elig = np.zeros(n, bool); elig[20:150] = True                              # predictions only from row 20
    alert = np.zeros(n, bool); alert[90] = True
    m = R.event_alert_metrics(alert, onsets, tau=20, eligible=elig)
    assert m["events_total"] == 2 and m["events_evaluable"] == 1 and m["events_censored"] == 1 and m["detected"] == 1
    assert m["exposure_rows"] == 130                                           # exposure = eligible rows only


def test_event_near_segment_end_is_censored_and_late_alarm_not_called_false():
    n = 100; elig = np.zeros(n, bool); elig[20:80] = True                      # last tau rows unscored
    onsets = np.array([95]); alert = np.zeros(n, bool); alert[75] = True       # follow window (75,95] still inside
    m = R.event_alert_metrics(alert, onsets, tau=20, eligible=elig)
    assert m["events_censored"] == 1 and m["events_evaluable"] == 0
    alert2 = np.zeros(n, bool); alert2[70] = True
    # alarm at 90 would have a follow window beyond the segment end -> censored, not false
    elig2 = np.ones(n, bool); alert3 = np.zeros(n, bool); alert3[90] = True
    m3 = R.event_alert_metrics(alert3, np.array([]), tau=20, eligible=elig2)
    assert m3["alarms_censored"] == 1 and m3["false_alarms"] == 0


def test_alert_cannot_cross_segment_seam():
    seg = np.r_[np.zeros(100, int), np.ones(100, int)]
    onsets = R.realised_onsets(np.abs(_ret(200, 105)), 1.0, 10, seg)
    alert = np.zeros(200, bool); alert[98] = True                              # across the seam, 7 rows before
    m = R.event_alert_metrics(alert, onsets, tau=20, seg=seg)
    assert m["events_censored"] == 1 and m["detected"] == 0                    # lead window starts before the segment -> censored
    seg2 = np.r_[np.zeros(100, int), np.ones(100, int)]
    onsets2 = np.array([150]); alert2 = np.zeros(200, bool); alert2[98] = True
    m2 = R.event_alert_metrics(alert2, onsets2, tau=20, seg=seg2)
    assert m2["detected"] == 0 and m2["false_alarms"] == 0 and m2["alarms_censored"] == 1   # follow window cut by seam -> censored, never "false"


def test_rate_uses_eligible_exposure_and_is_per_unit_rows():
    n = 1000; elig = np.zeros(n, bool); elig[100:600] = True
    alert = np.zeros(n, bool); alert[150] = True; alert[300] = True
    m = R.event_alert_metrics(alert, np.array([], dtype=int), tau=20, eligible=elig, rows_per_unit_time=100)
    assert m["false_alarms"] == 2 and abs(m["false_alarms_per_unit_time"] - 2 / (500 / 100)) < 1e-12


def test_refractory_merges_runs_of_one_event():
    assert list(R.realised_onsets(np.abs(_ret(100, 40, 41, 42, 43, 44, 80)), 1.0, 10)) == [40, 80]
