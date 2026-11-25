#!/bin/sh
# Robustness experiments (batch 2):
#   1. Held-out validation stopping (valstop=0.1) and fixed-step comparisons
#      for warm-start and recency-weighted rolling backtest policies.
#   2. Cohort subsample-seed sensitivity (subseed=1) on KuaiRand and RecFlow.
#   3. Recency half-life sensitivity sweep (1, 4, 8 days) on RecFlow.
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

DEVICE="${DEVICE:-}"
DEV_FLAG=""
[ -n "$DEVICE" ] && DEV_FLAG="--device $DEVICE"
FAILED=0
log() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

log "warm policy under held-out validation stopping"
# shellcheck disable=SC2086
"$PY" src/backtest_baselines.py recflow --mode warm --valstop 0.1 --fixedstep 300 $DEV_FLAG \
  || { echo "FAILED warm valstop"; FAILED=1; }

log "recency policy under held-out validation stopping"
# shellcheck disable=SC2086
"$PY" src/backtest_baselines.py recflow --mode recency --valstop 0.1 --fixedstep 300 $DEV_FLAG \
  || { echo "FAILED recency valstop"; FAILED=1; }

log "kuairand cohort, subsample seed 1"
# shellcheck disable=SC2086
"$PY" src/run_domain.py kuairand mf cohort subseed=1 $DEV_FLAG \
  || { echo "FAILED kuairand cohort subseed=1"; FAILED=1; }

log "recflow cohort, subsample seed 1"
# shellcheck disable=SC2086
"$PY" src/run_domain.py recflow mf cohort subseed=1 $DEV_FLAG \
  || { echo "FAILED recflow cohort subseed=1"; FAILED=1; }

log "recency half-life sweep (1, 4, 8 days)"
for hl in 1 4 8; do
  # shellcheck disable=SC2086
  "$PY" src/backtest_baselines.py recflow --mode recency --halflife "$hl" $DEV_FLAG \
    || { echo "FAILED recency halflife=$hl"; FAILED=1; }
done

log "robustness batch 2 complete"
exit "$FAILED"
