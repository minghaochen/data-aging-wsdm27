#!/bin/sh
# Low-memory multi-test-period sweeps (KuaiRand, MovieLens-25M, Netflix,
# KuaiRec). For larger embedding tables (RecFlow 7.2M items, Yelp 2.0M users),
# see src/run_queue_big.sh. Resume skips completed (exp, train_t0, train_t1,
# seed) runs.
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

for dom in kuairand ml25m netflix kuairec; do
  case "$dom" in
    netflix) COH="--cohort --item-only" ;;
    *)       COH="--cohort" ;;
  esac
  echo "### $dom apparent"
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $DEV_FLAG \
    || { echo "FAILED $dom apparent"; FAILED=1; }
  echo "### $dom cohort"
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $COH $DEV_FLAG \
    || { echo "FAILED $dom cohort"; FAILED=1; }
done
echo "### LOW-MEMORY PERIOD PASS COMPLETE"
exit "$FAILED"
