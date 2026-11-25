#!/bin/sh
# Extend cohort sweeps to 9 seeds for RecFlow, Yelp (item cohort), and
# MovieLens-25M (item cohort for like-for-like cross-platform comparison
# against Netflix's item cohort). Resume skips already-completed seeds.
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

DEV_FLAG=""
[ -n "${DEVICE:-}" ] && DEV_FLAG="--device $DEVICE"
FAILED=0

echo "### recflow cohort x9"
# shellcheck disable=SC2086
"$PY" src/run_periods.py recflow --periods 0,1,2 --cohort --seeds 9 $DEV_FLAG \
  || { echo "FAILED recflow cohort9"; FAILED=1; }
echo "### yelp cohort x9"
# shellcheck disable=SC2086
"$PY" src/run_periods.py yelp --periods 0,1,2 --cohort --item-only --seeds 9 $DEV_FLAG \
  || { echo "FAILED yelp cohort9"; FAILED=1; }
echo "### ml25m item-cohort x9"
# shellcheck disable=SC2086
"$PY" src/run_periods.py ml25m --periods 0,1,2 --cohort --item-only --seeds 9 $DEV_FLAG \
  || { echo "FAILED ml25m icohort9"; FAILED=1; }
echo "### COHORT 9-SEED EXTENSION COMPLETE"
exit "$FAILED"
