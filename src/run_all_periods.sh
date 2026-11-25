#!/bin/sh
# Multi-test-period sweep with the prior-shift control across six primary
# domains (serial execution). Every invocation resumes: completed
# (exp, train_t0, train_t1, seed) runs are skipped automatically.
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

for dom in kuairand recflow ml25m netflix kuairec yelp; do
  case "$dom" in
    netflix|yelp) COH="--cohort --item-only" ;;
    *)            COH="--cohort" ;;
  esac
  echo "===== $dom apparent ====="
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $DEV_FLAG \
    || { echo "FAILED $dom apparent"; FAILED=1; }
  echo "===== $dom cohort ====="
  # shellcheck disable=SC2086
  "$PY" src/run_periods.py "$dom" --periods 0,1,2 $COH $DEV_FLAG \
    || { echo "FAILED $dom cohort"; FAILED=1; }
done
echo "ALL PERIOD SWEEPS DONE"
exit "$FAILED"
