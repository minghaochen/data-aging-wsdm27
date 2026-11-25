#!/bin/sh
# Extend KuaiRand, MovieLens-25M, Netflix, and KuaiRec cohort sweeps to
# 9 seeds, and run an apparent-curve resume pass to fill any Arm C cells.
#
# At 3 seeds, tau_half is sensitive to seed sampling variance (resampling the
# 9-seed KuaiRand cohort down to 3 seeds spans [4.3, 13.4] d across subsets,
# 5th-95th percentile [5.0, 12.9] d), so 9 seeds are used for period spreads.
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

echo "### cohort sweeps to 9 seeds"
for dom in kuairand ml25m netflix kuairec; do
  case "$dom" in
    netflix) COH="--cohort --item-only" ;;
    *)       COH="--cohort" ;;
  esac
  echo "### $dom cohort x9"
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $COH --seeds 9 $DEV_FLAG \
    || { echo "FAILED $dom cohort9"; FAILED=1; }
done

echo "### apparent resume sweep (completing Arm C cells across periods)"
for dom in kuairand ml25m netflix kuairec; do
  echo "### $dom apparent resume"
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $DEV_FLAG \
    || { echo "FAILED $dom apparent resume"; FAILED=1; }
done
echo "### SEEDS9 PASS COMPLETE"
exit "$FAILED"
