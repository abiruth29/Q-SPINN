import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


# Release mode: QSPINN_REQUIRE_FULL=1 turns every skipped test into a failure (no silent dependency skips).
import os
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if os.environ.get("QSPINN_REQUIRE_FULL") == "1" and rep.skipped:
        rep.outcome = "failed"
        rep.longrepr = f"SKIPPED test not allowed in release mode (QSPINN_REQUIRE_FULL=1): {rep.longrepr}"
