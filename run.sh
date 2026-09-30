#!/usr/bin/env bash
# One-command start: creates .venv on first use, installs, opens http://127.0.0.1:8765
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -e ".[test]"
fi
exec .venv/bin/python -m convlab serve "$@"
