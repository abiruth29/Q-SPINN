"""Review 4 item 5: active docs agree with current code; no stale outage tolerance or unqualified benchmark claims."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ACTIVE = [ROOT / "README.md", ROOT / "docs" / "RUNBOOK_TEAMMATE.md", ROOT / "presentation" / "index.html",
          ROOT / "presentation" / "qspinn-presentation.html"]


def _text(p): return p.read_text(encoding="utf-8", errors="ignore")


def test_no_stale_outage_tolerance():
    for p in ACTIVE:
        t = _text(p)
        assert not re.search(r"more than 1 ?% of the grid", t), p
        assert not re.search(r"stops?\*{0,2} if more than", t), p


def test_no_unqualified_benchmark_claims():
    for p in ACTIVE[2:]:
        t = _text(p)
        assert "wasting 99% of compute" not in t, p
        assert "directly reduce latency" not in t, p
        assert "Target Metrics</h3>" not in t, p
        assert "All metrics evaluated via walk-forward" not in t, p


def test_runbook_states_current_policies():
    t = _text(ROOT / "docs" / "RUNBOOK_TEAMMATE.md")
    for must in ("ANY flagged outage", "gate_objective", "omega_mode", "split_plan"):
        assert must in t, must
