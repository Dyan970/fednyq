from __future__ import annotations

import torch
from torch import Tensor, nn


class BandBlock(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
        )
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.encoder(x)
        return self.head(feature), feature


class NyqNet(nn.Module):
    def __init__(self, channels: int, num_classes: int, base_rate: float,
                 window_seconds: float, rates: tuple[float, ...], embedding_dim: int = 64) -> None:
        super().__init__()
        self.base_rate = float(base_rate)
        self.window_seconds = float(window_seconds)
        self.rates = tuple(sorted(rates))
        self.band_uppers = tuple(r / 2 for r in self.rates)
        bins = self._band_bin_counts(base_rate, round(base_rate * window_seconds))
        self.band1 = BandBlock(channels * bins[0], embedding_dim, num_classes)
        self.band2 = BandBlock(channels * bins[1], embedding_dim, num_classes)
        self.band3 = BandBlock(channels * bins[2], embedding_dim, num_classes)
        if len(self.rates) != 3:
            raise ValueError("NyqNet currently requires exactly three rates/bands")

    def _band_bin_counts(self, sample_rate: float, length: int) -> list[int]:
        freqs = torch.fft.rfftfreq(length, d=1.0 / sample_rate)
        counts, lower = [], 0.0
        for i, upper in enumerate(self.band_uppers):
            mask = (freqs >= lower if i == 0 else freqs > lower) & (freqs <= upper + 1e-7)
            counts.append(int(mask.sum()))
            lower = upper
        return counts

    def active_bands(self, sample_rate: float, max_active_rate: float | None = None) -> list[int]:
        nyquist = min(sample_rate, max_active_rate or sample_rate) / 2
        return [i + 1 for i, upper in enumerate(self.band_uppers) if upper <= nyquist + 1e-7]

    def forward(self, x: Tensor, sample_rate: float, max_active_rate: float | None = None,
                return_features: bool = False) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        expected = round(sample_rate * self.window_seconds)
        if x.shape[-1] != expected:
            raise AssertionError(f"Expected {expected} samples at {sample_rate} Hz, got {x.shape[-1]}")
        spectrum = torch.log1p(torch.abs(torch.fft.rfft(x, dim=-1)))
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate, device=x.device)
        logits, features, lower = [], {}, 0.0
        for band in self.active_bands(sample_rate, max_active_rate):
            upper = self.band_uppers[band - 1]
            mask = (freqs >= lower if band == 1 else freqs > lower) & (freqs <= upper + 1e-7)
            values = spectrum[..., mask].flatten(1)
            block = getattr(self, f"band{band}")
            expected_bins = block.encoder[0].in_features // x.shape[1]
            assert values.shape[1] == expected_bins * x.shape[1], "Physical-frequency bin mismatch"
            z, feat = block(values)
            logits.append(z)
            features[band] = feat
            lower = upper
        if not logits:
            raise ValueError("No active band for sample_rate")
        output = torch.stack(logits).mean(0)
        return (output, features) if return_features else output


def get_param_band(param_name: str) -> int | None:
    for band in (1, 2, 3):
        if param_name.startswith(f"band{band}."):
            return band
    return None
