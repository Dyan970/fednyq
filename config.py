from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    data: str = "data/dataset.npz"
    output_dir: str = "outputs"
    experiment_name: str = "default"
    method: str = "fedavg"
    seed: int = 2025
    base_sample_rate: float = 100.0
    window_seconds: float = 4.0
    rates: tuple[float, ...] = (25.0, 50.0, 100.0)
    rate_probs: tuple[float, ...] = (0.4, 0.4, 0.2)
    num_clients: int = 20
    client_fraction: float = 0.5
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 10
    test_size: float = 0.2
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 64
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    lambda_cons: float = 0.2
    cons_temperature: float = 2.0
    num_workers: int = 0
    amp: bool = False
    device: str = "auto"
    fir_taps: int = 63
    plot_compare_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name / str(self.seed)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="FedAvg vs Nyquist-support-normalized FL")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--experiment_name", default="default")
    p.add_argument("--method", choices=["fedavg", "fednyq"], default="fedavg")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--base_sample_rate", type=float, default=100.0)
    p.add_argument("--window_seconds", type=float, default=4.0)
    p.add_argument("--rates", type=float, nargs="+", default=[25, 50, 100])
    p.add_argument("--rate_probs", type=float, nargs="+", default=[0.4, 0.4, 0.2])
    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--client_fraction", type=float, default=0.5)
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--min_client_samples", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--lambda_cons", type=float, default=None)
    p.add_argument("--cons_temperature", type=float, default=2.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--fir_taps", type=int, default=63)
    p.add_argument("--plot_compare_dir", default=None)
    a = p.parse_args()
    if len(a.rates) != len(a.rate_probs) or any(v <= 0 for v in a.rate_probs):
        p.error("--rates and --rate_probs must have equal length and positive probabilities")
    if len(a.rates) != 3 or list(a.rates) != sorted(set(a.rates)):
        p.error("--rates must contain exactly three unique ascending values")
    if any(abs(a.base_sample_rate / rate - round(a.base_sample_rate / rate)) > 1e-9
           for rate in a.rates):
        p.error("all rates must be integer divisors of --base_sample_rate in this implementation")
    values = vars(a)
    values["rates"] = tuple(float(x) for x in a.rates)
    total = sum(a.rate_probs)
    values["rate_probs"] = tuple(float(x / total) for x in a.rate_probs)
    if a.lambda_cons is None:
        values["lambda_cons"] = 0.2 if a.method == "fednyq" else 0.0
    if a.method == "fedavg" and values["lambda_cons"] != 0:
        p.error("FedAvg requires --lambda_cons 0")
    return Config(**values)
