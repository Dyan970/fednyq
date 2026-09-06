#!/usr/bin/env bash
set -euo pipefail
for seed in 2025 2026 2027; do
  bash scripts/run_fedavg.sh --seed "$seed" "$@"
  bash scripts/run_fednyq.sh --seed "$seed" "$@"
done
python train.py summarize \
  --seeds 2025 2026 2027 \
  --fedavg_experiment fedavg \
  --fednyq_experiment fednyq
