#!/usr/bin/env bash
# Trains all four context conditions on the same cached pairs.
# Bumps epochs from the smoke-test value of 1 to something that actually
# learns a coupling between audio and visuals.
#
# Override defaults with env vars, e.g.:
#   EPOCHS=80 ./run_context_experiments.sh
set -euo pipefail

EPOCHS="${EPOCHS:-50}"
SAVE_EVERY="${SAVE_EVERY:-${EPOCHS}}"

if [[ ! -f cache/index.pt ]]; then
  python prepare_data.py --data-root data --cache-dir cache
fi

for mode in audio_only style biography combined; do
  echo "==== training mode=${mode} epochs=${EPOCHS} ===="
  python train.py \
    --cache-dir cache \
    --out-dir "checkpoints/${mode}" \
    --epochs "${EPOCHS}" \
    --save-every "${SAVE_EVERY}" \
    --profile-mode "${mode}" \
    --seed 0
done
