#!/usr/bin/env bash
set -euo pipefail
seed="${SEED:-2025}"
for high in 0.8 0.5 0.2; do
  remain=$(python -c "print((1-float('$high'))/2)")
  for method in fedavg fednyq; do
    lambda=0
    if [[ "$method" == "fednyq" ]]; then lambda=0.2; fi
    python train.py \
      --data data/synthetic.npz \
      --method "$method" \
      --experiment_name "sweep_f${high}_${method}" \
      --rates 25 50 100 \
      --rate_probs "$remain" "$remain" "$high" \
      --lambda_cons "$lambda" \
      --seed "$seed" \
      "$@"
  done
done
python train.py rate-sweep-summary --fractions 0.8 0.5 0.2 --seed "$seed"
