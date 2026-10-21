#!/usr/bin/env bash
# Dependency-complete release test run: zero skips allowed. Saves a log with versions + hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -r requirements.txt >/dev/null
OUT=docs/TEST_LOG_latest.txt
{
  echo "# Release test log (QSPINN_REQUIRE_FULL=1: skips are failures)"
  date -u +"%Y-%m-%d %H:%M UTC"; python -V
  python - <<'PY'
import importlib, platform
for m in ("torch","numpy","pandas","pyarrow","scipy","pytest"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "MISSING", e)
import torch; print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("platform:", platform.platform(), "| cpu threads:", __import__("os").cpu_count())
PY
  echo
  QSPINN_REQUIRE_FULL=1 python -m pytest tests -q -rs -p no:cacheprovider
} 2>&1 | tee "$OUT"
