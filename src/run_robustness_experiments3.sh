#!/bin/sh
# Robustness experiments (batch 3):
#   Cohort subsample-seed sensitivity (subseed=1) for the Game-UGC finish
#   item cohort.
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

log "game-UGC finish item cohort, subsample seed 1"
# shellcheck disable=SC2086
"$PY" src/run_domain.py gameugc_finish mf icohort subseed=1 $DEV_FLAG \
  || { echo "FAILED gameugc_finish icohort subseed=1"; FAILED=1; }

log "robustness batch 3 complete"
exit "$FAILED"
