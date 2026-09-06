from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_dataset(n: int, channels: int, sample_rate: int, seconds: float,
                 noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64) % 3
    rng.shuffle(labels)
    t = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    x = np.empty((n, channels, len(t)), dtype=np.float32)
    class_freqs = ((3.0,), (3.0, 18.0), (6.0, 30.0))
    for cls, freqs in enumerate(class_freqs):
        ids = np.flatnonzero(labels == cls)
        phases = rng.uniform(0, 2 * np.pi, size=(len(ids), channels, len(freqs), 1))
        amps = rng.uniform(0.8, 1.2, size=(len(ids), channels, len(freqs), 1))
        waves = np.sin(2 * np.pi * np.asarray(freqs)[None, None, :, None] * t + phases)
        x[ids] = (amps * waves).sum(axis=2) + rng.normal(0, noise, (len(ids), channels, len(t)))
    return x.astype(np.float32), labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic.npz")
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--sample_rate", type=int, default=100)
    p.add_argument("--window_seconds", type=float, default=4.0)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    x, y = make_dataset(a.num_samples, a.channels, a.sample_rate, a.window_seconds, a.noise, a.seed)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    print(f"Wrote {path}: x={x.shape} {x.dtype}, y={y.shape} {y.dtype}")


if __name__ == "__main__":
    main()
