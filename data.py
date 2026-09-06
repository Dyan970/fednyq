from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

_FIR_DEVICE_CACHE: dict[tuple[float, float, int, str, torch.dtype], Tensor] = {}


class ArrayDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.indices = np.arange(len(y)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        i = self.indices[index]
        return self.x[i], self.y[i]


@lru_cache(maxsize=64)
def _fir_cpu(source_rate: float, target_rate: float, taps: int) -> Tensor:
    if taps % 2 == 0:
        raise ValueError("fir_taps must be odd")
    ratio = source_rate / target_rate
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"Only integer decimation is supported, got {source_rate}->{target_rate}")
    if source_rate == target_rate:
        return torch.ones(1)
    n = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
    cutoff = 0.5 / ratio * 0.94
    h = 2 * cutoff * torch.sinc(2 * cutoff * n) * torch.hann_window(taps, periodic=False, dtype=torch.float64)
    return (h / h.sum()).float()


def lowpass_resample(x: Tensor, source_rate: float, target_rate: float, taps: int = 63) -> Tensor:
    """Anti-aliased integer-ratio decimation along the last dimension."""
    if source_rate == target_rate:
        return x
    ratio_f = source_rate / target_rate
    if ratio_f < 1 or abs(ratio_f - round(ratio_f)) > 1e-9:
        raise ValueError(f"Only integer decimation is supported, got {source_rate}->{target_rate}")
    ratio = round(ratio_f)
    channels = x.shape[-2]
    key = (float(source_rate), float(target_rate), taps, str(x.device), x.dtype)
    if key not in _FIR_DEVICE_CACHE:
        _FIR_DEVICE_CACHE[key] = _fir_cpu(*key[:3]).to(device=x.device, dtype=x.dtype)
    h = _FIR_DEVICE_CACHE[key]
    flat = x.reshape(-1, channels, x.shape[-1])
    kernel = h.view(1, 1, -1).expand(channels, 1, -1)
    filtered = F.conv1d(flat, kernel, padding=taps // 2, groups=channels)
    decimated = filtered[..., ::ratio]
    return decimated.reshape(*x.shape[:-1], decimated.shape[-1])


def load_and_split(path: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    x, y = d["x"].astype(np.float32, copy=False), d["y"].astype(np.int64, copy=False)
    if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected x [N,C,L] and y [N]")
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
    return x, y, np.sort(train_idx), np.sort(test_idx)


def partition_clients(labels: np.ndarray, train_idx: np.ndarray, num_clients: int,
                      kind: str, alpha: float, min_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(train_idx) < num_clients * min_size:
        raise ValueError("Not enough training samples for min_client_samples")
    if kind == "iid":
        shuffled = rng.permutation(train_idx)
        return [np.sort(v) for v in np.array_split(shuffled, num_clients)]
    for _ in range(1000):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in np.unique(labels[train_idx]):
            ids = rng.permutation(train_idx[labels[train_idx] == cls])
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(ids)).astype(int)
            for k, chunk in enumerate(np.split(ids, cuts)):
                parts[k].extend(chunk.tolist())
        if min(map(len, parts)) >= min_size:
            return [np.sort(np.asarray(v, dtype=np.int64)) for v in parts]
    raise RuntimeError("Could not make a Dirichlet partition satisfying min_client_samples")


def assign_rates(num_clients: int, rates: tuple[float, ...], probs: tuple[float, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    expected = np.asarray(probs) * num_clients
    counts = np.floor(expected).astype(int)
    remainder = num_clients - int(counts.sum())
    order = np.argsort(-(expected - counts))
    counts[order[:remainder]] += 1
    assigned = np.concatenate([np.full(n, rate) for rate, n in zip(rates, counts)])
    rng.shuffle(assigned)
    return assigned.astype(np.float64)


def make_schedule(rounds: int, num_clients: int, fraction: float, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed + 31)
    n = max(1, round(num_clients * fraction))
    return [sorted(rng.choice(num_clients, n, replace=False).tolist()) for _ in range(rounds)]


def stable_hash(value: object) -> str:
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        payload = b"".join(np.asarray(v, dtype=np.int64).tobytes() + b"|" for v in value)
    elif isinstance(value, np.ndarray):
        payload = value.tobytes()
    else:
        payload = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
