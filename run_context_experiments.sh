#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f cache/index.pt ]]; then
  python prepare_data.py --data-root data --cache-dir cache
fi

for mode in style biography combined; do
  python train.py \
    --cache-dir cache \
    --out-dir "checkpoints/${mode}" \
    --epochs 1 \
    --save-every 1 \
    --profile-mode "${mode}" \
    --seed 0
done
