#!/usr/bin/env bash
# Octave cross-check of the Python app (see matlab/README.md).
#   1. export the checked presets:  python -m convlab run LAB EXP --preset P --out DIR
#   2. run matlab/run_all.m in GNU Octave; it recomputes every value from the
#      textbook equations and writes matlab/results/octave_crosscheck.json
#   3. exit non-zero when any item is FAIL (TODO items do not fail)
#
# Environment: PYTHON (default .venv/bin/python, then python3), OCTAVE (default
# octave), EXPORT_DIR, OUT_FILE. Exit 2 = NOT_RUN_ENVIRONMENT (no Octave / Python).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT="${EXPORT_DIR:-$ROOT/matlab/results/python_export}"
OUT="${OUT_FILE:-$ROOT/matlab/results/octave_crosscheck.json}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi
OCT="${OCTAVE:-octave}"
if ! command -v "$OCT" >/dev/null 2>&1; then
  echo "NOT_RUN_ENVIRONMENT: '$OCT' not found (GNU Octave is needed for this cross-check)" >&2
  exit 2
fi
if ! PYTHONPATH="$ROOT/src" "$PY" -c "import convlab" >/dev/null 2>&1; then
  echo "NOT_RUN_ENVIRONMENT: '$PY' cannot import convlab (set PYTHON to the app's venv python)" >&2
  exit 2
fi

# lab experiment preset -- one line per exported preset
EXPORTS=(
  "FL08 sps_nominal nominal"
  "FL08 zero_power_mismatch mismatch"
  "EX06 general_modulation textbook"
  "EX02 qe_integrals textbook"
  "EX02 hb_constant_current textbook"
  "EX07 cpl_exact c100u"
  "EX07 cpl_exact c1m"
  "FL09 fha_gain textbook"
  "FL05 grid_boundary nominal"
  "FL05 dclink_power nominal"
  "FL01 buck_ccm nominal"
  # TODO(FL10/EX05 time domain): add the FL10 / EX05 presets here when they are merged
)

mkdir -p "$EXPORT"
rm -f "$EXPORT"/*.json "$EXPORT"/*.csv "$EXPORT"/*.html
echo "== python export -> $EXPORT"
for e in "${EXPORTS[@]}"; do
  read -r lab exp preset <<<"$e"
  echo "   python -m convlab run $lab $exp --preset $preset"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m convlab run "$lab" "$exp" --preset "$preset" --out "$EXPORT" >/dev/null
done

echo "== $("$OCT" --version 2>/dev/null | head -n 1)"
set +e
"$OCT" --no-gui --quiet --eval "addpath('$ROOT/matlab'); ok = run_all('$EXPORT', '$OUT'); exit(double(~ok));"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "FAIL: Octave cross-check reported FAIL items (or did not finish), see $OUT" >&2
  exit 1
fi
echo "PASS: all Octave cross-check items within tolerance ($OUT)"
