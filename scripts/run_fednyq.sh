#!/usr/bin/env bash
set -euo pipefail
python train.py \
  --data data/synthetic.npz \
  --method fednyq \
  --experiment_name fednyq \
  --rates 25 50 100 \
  --rate_probs 0.4 0.4 0.2 \
  --lambda_cons 0.2 \
  "$@"
