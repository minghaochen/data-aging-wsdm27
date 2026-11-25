#!/bin/sh
# Memory-intensive runs executed sequentially so GPU VRAM is not shared
# (RecFlow's 7.2M-item embedding table with SparseAdam moments uses ~2.8 GB):
#   1. RecFlow weekday control (requires the oldest age point)
#   2. RecFlow and Yelp multi-test-period sweeps
#   3. RecFlow rolling-backtest baselines (recency weighting, warm start)
set -u

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src

DEVICE="${DEVICE:-cuda}"
FAILED=0

echo "### recflow weekday control"
"$PY" src/control_weekday.py recflow --device "$DEVICE" \
  || { echo "FAILED recflow weekday control"; FAILED=1; }

for dom in recflow yelp; do
  case "$dom" in
    yelp) COH="--cohort --item-only" ;;
    *)    COH="--cohort" ;;
  esac
  echo "### $dom apparent"
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 --device "$DEVICE" \
    || { echo "FAILED $dom apparent"; FAILED=1; }
  echo "### $dom cohort"
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $COH --device "$DEVICE" \
    || { echo "FAILED $dom cohort"; FAILED=1; }
done

echo "### backtest baselines"
"$PY" src/backtest_baselines.py recflow --mode both --device "$DEVICE" \
  || { echo "FAILED backtest baselines"; FAILED=1; }
echo "### HIGH-MEMORY BATCH COMPLETE"
exit "$FAILED"
