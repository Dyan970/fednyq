from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _colored_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    white = rng.normal(size=shape)
    spectrum = np.fft.rfft(white, axis=-1)
    frequencies = np.fft.rfftfreq(shape[-1])
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    colored = np.fft.irfft(spectrum * scale, n=shape[-1], axis=-1)
    return colored / (colored.std(axis=-1, keepdims=True) + 1e-8)


def _complex_class(rng: np.random.Generator, count: int, channels: int,
                   t: np.ndarray, cls: int, noise: float) -> np.ndarray:
    specs = (((3.0, 1.0), (8.0, 0.25)),
             ((3.0, 0.85), (18.0, 0.80), (9.0, 0.18)),
             ((6.0, 0.90), (30.0, 0.75), (11.0, 0.18)))
    freqs = np.asarray([v[0] for v in specs[cls]])
    base_amps = np.asarray([v[1] for v in specs[cls]])
    components = len(freqs)
    jitter = rng.normal(0, 0.35, (count, components, 1))
    phase = rng.uniform(0, 2 * np.pi, (count, components, 1))
    amplitude = base_amps[None, :, None] * rng.lognormal(0, 0.22, (count, components, 1))
    mod_freq = rng.uniform(0.15, 0.65, (count, components, 1))
    mod_phase = rng.uniform(0, 2 * np.pi, (count, components, 1))
    modulation = 1 + rng.uniform(0.05, 0.30, (count, components, 1)) * np.sin(
        2 * np.pi * mod_freq * t + mod_phase)
    latent = amplitude * modulation * np.sin(2 * np.pi * (freqs[None, :, None] + jitter) * t + phase)
    if cls in (1, 2):
        center = rng.uniform(0.8, t[-1] - 0.8, (count, 1, 1))
        width = rng.uniform(0.35, 0.9, (count, 1, 1))
        envelope = 0.35 + 0.65 * np.exp(-0.5 * ((t - center) / width) ** 2)
        latent[:, 1:2] *= envelope
    mixing = rng.normal(1.0, 0.30, (count, channels, components))
    signal = np.einsum("ncf,nfl->ncl", mixing, latent)
    drift_freq = rng.uniform(0.12, 0.7, (count, channels, 1))
    drift_phase = rng.uniform(0, 2 * np.pi, (count, channels, 1))
    drift = rng.uniform(0.05, 0.25, (count, channels, 1)) * np.sin(2 * np.pi * drift_freq * t + drift_phase)
    signal += drift + noise * _colored_noise(rng, signal.shape)
    artifacts = rng.random(signal.shape) < 0.002
    signal += artifacts * rng.normal(0, 3.0, signal.shape)
    gains = rng.lognormal(0, 0.18, (count, channels, 1))
    dropped = rng.random((count, channels, 1)) < 0.03
    return signal * gains * np.where(dropped, 0.05, 1.0)


def make_dataset(n: int, channels: int, sample_rate: int, seconds: float,
                 noise: float, seed: int, profile: str = "simple") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64) % 3
    rng.shuffle(labels)
    t = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    x = np.empty((n, channels, len(t)), dtype=np.float32)
    class_freqs = ((3.0,), (3.0, 18.0), (6.0, 30.0))
    for cls, freqs in enumerate(class_freqs):
        ids = np.flatnonzero(labels == cls)
        if profile == "complex":
            x[ids] = _complex_class(rng, len(ids), channels, t, cls, noise)
        else:
            phases = rng.uniform(0, 2 * np.pi, size=(len(ids), channels, len(freqs), 1))
            amps = rng.uniform(0.8, 1.2, size=(len(ids), channels, len(freqs), 1))
            waves = np.sin(2 * np.pi * np.asarray(freqs)[None, None, :, None] * t + phases)
            x[ids] = (amps * waves).sum(axis=2) + rng.normal(0, noise, (len(ids), channels, len(t)))
    return x.astype(np.float32), labels


def print_band_report(x: np.ndarray, y: np.ndarray, sample_rate: int, rates: list[float]) -> None:
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1 / sample_rate)
    uppers = [rate / 2 for rate in rates]
    bands = list(zip([0.0] + uppers[:-1], uppers))
    for cls in np.unique(y):
        powers = []
        class_spectrum = spectrum[y == cls]
        for i, (low, high) in enumerate(bands):
            mask = (freqs >= low if i == 0 else freqs > low) & (freqs <= high)
            powers.append(float(class_spectrum[..., mask].mean()))
        relative = np.asarray(powers) / (sum(powers) + 1e-12)
        print(f"class {cls} relative band power: " + ", ".join(f"B{i + 1}={v:.3f}" for i, v in enumerate(relative)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic.npz")
    p.add_argument("--num_samples", type=int, default=3000)
    p.add_argument("--profile", choices=["simple", "complex"], default="simple")
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--sample_rate", type=int, default=100)
    p.add_argument("--rates", type=float, nargs="+", default=[25, 50, 100])
    p.add_argument("--window_seconds", type=float, default=4.0)
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    if len(a.rates) != 3 or a.rates != sorted(set(a.rates)) or max(a.rates) > a.sample_rate:
        p.error("--rates must be three unique ascending values not exceeding --sample_rate")
    if a.sample_rate < 60:
        p.error("--sample_rate must be at least 60 Hz for the built-in 30 Hz component")
    channels = a.channels if a.channels is not None else (3 if a.profile == "complex" else 1)
    noise = a.noise if a.noise is not None else (0.65 if a.profile == "complex" else 0.5)
    x, y = make_dataset(a.num_samples, channels, a.sample_rate, a.window_seconds,
                        noise, a.seed, a.profile)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, profile=np.asarray(a.profile),
                        sample_rate=np.asarray(a.sample_rate), window_seconds=np.asarray(a.window_seconds),
                        rates=np.asarray(a.rates, dtype=np.float32))
    print(f"Wrote {path}: profile={a.profile}, x={x.shape} {x.dtype}, y={y.shape} {y.dtype}")
    print_band_report(x, y, a.sample_rate, a.rates)


if __name__ == "__main__":
    main()
