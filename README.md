# FedNyq: Nyquist-Support-Normalized Federated Learning

This compact PyTorch repository compares standard FedAvg with FedNyq on multirate
sensor time series. It uses no federated-learning framework. Every client sees the
same physical window duration but has a fixed sampling rate and therefore a different
Nyquist support. The input FFT is applied to the sensor signal—not to model parameters.

## Method

`NyqNet` computes `log1p(abs(rFFT(x)))`, constructs physical frequencies with
`torch.fft.rfftfreq`, and routes the available frequency bins to three independent
lightweight band encoders. Band boundaries are derived from the sorted client rates.
At low rates, unavailable encoders are not executed. Active-band logits are averaged,
which keeps their scale comparable across rates.

For selected clients with sample counts $n_k$, FedAvg uses

$$\theta^{t+1}=\frac{\sum_k n_k\theta_k^{t+1}}{\sum_k n_k}. $$

For a parameter belonging to band $b$, FedNyq aggregates deltas only from clients
whose Nyquist frequency covers the band's upper edge:

$$\Delta_b=\frac{\sum_k n_k m_{k,b}\Delta_{k,b}}
{\sum_k n_km_{k,b}+10^{-12}},\qquad
\theta_b^{t+1}=\theta_b^t+\Delta_b.$$

The observable selected-client mass is
$\rho_b=\sum_kp_km_{k,b}$. The CSV records both theoretical attenuation $\rho_b$
and the empirical ratio between a FedAvg-diluted eligible update and its
support-normalized counterpart.

Full FedNyq also applies cross-rate consistency. A high-rate window is anti-aliased
and downsampled to a randomly chosen lower supported rate. Predictions are compared
only on their common Nyquist support:

$$L=L_{CE}+\lambda\,\frac{KL(p_T\Vert q_T)+KL(q_T\Vert p_T)}{2},$$

where temperature-scaled logits use `log_softmax`/`softmax`. Use `--lambda_cons 0`
for the FedNyq-Agg-only ablation.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Synthetic smoke test

The generator creates three balanced classes containing 3 Hz, 18 Hz, and 30 Hz
components plus Gaussian noise. Lower-rate evaluation should lose information for
the high-frequency classes.

```bash
python make_synthetic.py --output data/synthetic.npz
python train.py --data data/synthetic.npz --method fedavg \
  --experiment_name fedavg --rates 25 50 100 --rate_probs 0.4 0.4 0.2 \
  --lambda_cons 0 --seed 2025
python train.py --data data/synthetic.npz --method fednyq \
  --experiment_name fednyq --rates 25 50 100 --rate_probs 0.4 0.4 0.2 \
  --lambda_cons 0.2 --seed 2025
```

For a quick CPU smoke test, append `--num_clients 4 --client_fraction 0.5
--rounds 2 --local_epochs 1 --batch_size 32 --partition iid
--min_client_samples 2`.

Run the aggregation-only ablation with:

```bash
python train.py --data data/synthetic.npz --method fednyq \
  --experiment_name fednyq_agg --lambda_cons 0
```

Run the standard three seeds and paired summary, or the high-rate fraction sweep:

```bash
bash scripts/run_all_seeds.sh
bash scripts/run_rate_sweep.sh
```

The first command reports mean, sample standard deviation, t-based 95% CI, and a
matched-seed paired t-test. With only three seeds these statistics are descriptive;
the README makes no claim of statistical significance. The rate sweep holds the two
lower-rate probabilities equal while changing the 100 Hz fraction.

## Outputs

Each run writes to `outputs/<experiment_name>/<seed>/`:

- `config.json`: full configuration and four fairness hashes;
- `client_metadata.csv`: client size, fixed rate, and label histogram;
- `round_metrics.csv`: paper, system, and band diagnostics for every round;
- `final_metrics.json`: final row plus reproducibility hashes;
- PNGs for accuracy, macro-F1, per-rate F1, communication, time, and rho/update ratio.

`run_all_seeds.sh` additionally writes `outputs/summary.csv` and cross-method learning
curves. `run_rate_sweep.sh` writes `outputs/rate_sweep.csv`,
`performance_vs_high_rate_fraction.png`, and `band_update_norm_vs_rho.png`.

Communication assumes every selected client downloads and uploads the complete state
for both methods. Bytes are computed from each state tensor's `numel` and dtype size;
FedNyq is not presented as communication-efficient.

## Fairness and reproducibility

For a fixed seed, split, client partitions, exact rate counts/assignment, model
initialization, and the full client-selection schedule are generated before training.
Their hashes are printed and saved. FedAvg and FedNyq therefore use the same NyqNet
and federated realization: **only the optimization/aggregation strategy differs**.
Test indices never enter partitioning or training. All random sources are seeded,
client schedules are precomputed, and local DataLoader seeds depend only on seed,
round, and client ID.

The FIR resampler uses a cached Hann-windowed sinc low-pass kernel and grouped
`conv1d` before decimation. This first version intentionally rejects non-integer
ratios; supported severity configurations include 100/50/25 and 100/50/20 Hz.
