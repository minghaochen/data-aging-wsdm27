#!/bin/sh
# Robustness experiments (batch 1), executed sequentially:
#   1. Paired held-out validation stopping (valstop=0.1) and fixed-step
#      evaluations alongside test-argmin selection on static and backtest runs.
#   2. Volume-matched rolling backtest on RecFlow at D_fix=300,000.
#   3. 9-seed cohort extensions on BeerAdvocate and Yelp.
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

# Validation-stop sensitivity on static age curves
log "recflow static validation-stop sensitivity"
# shellcheck disable=SC2086
"$PY" src/run_domain.py recflow mf valstop=0.1 fixedstep=300 $DEV_FLAG \
  || { echo "FAILED recflow static valstop"; FAILED=1; }

log "kuairand static validation-stop sensitivity"
# shellcheck disable=SC2086
"$PY" src/run_domain.py kuairand mf valstop=0.1 fixedstep=100 $DEV_FLAG \
  || { echo "FAILED kuairand static valstop"; FAILED=1; }

# Validation-stop sensitivity on rolling backtest
log "recflow backtest validation-stop sensitivity"
# shellcheck disable=SC2086
"$PY" src/backtest.py recflow tag=__valstop valstop=0.1 fixedstep=300 $DEV_FLAG \
  || { echo "FAILED recflow backtest valstop"; FAILED=1; }

# Volume-matched rolling backtest at D_fix=300,000
log "recflow backtest at matched training volume D=300k"
# shellcheck disable=SC2086
"$PY" src/backtest.py recflow tag=__d300k dfix=300000 $DEV_FLAG \
  || { echo "FAILED recflow backtest d300k"; FAILED=1; }

# 9-seed extensions for remaining cohort timescales
log "beer item cohort, 9 seeds"
# shellcheck disable=SC2086
"$PY" src/run_domain.py beer mf icohort seeds=9 $DEV_FLAG \
  || { echo "FAILED beer icohort seeds=9"; FAILED=1; }

log "yelp user-cap-item cohort, 9 seeds"
# shellcheck disable=SC2086
"$PY" src/run_domain.py yelp mf cohort seeds=9 $DEV_FLAG \
  || { echo "FAILED yelp cohort seeds=9"; FAILED=1; }

log "robustness batch 1 complete"
exit "$FAILED"
