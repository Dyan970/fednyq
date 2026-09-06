from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from config import Config
from data import ArrayDataset, lowpass_resample
from model import get_param_band


def symmetric_kl(a: Tensor, b: Tensor, temperature: float) -> Tensor:
    log_p, log_q = F.log_softmax(a / temperature, 1), F.log_softmax(b / temperature, 1)
    p, q = log_p.exp(), log_q.exp()
    return 0.5 * (F.kl_div(log_p, q, reduction="batchmean") +
                  F.kl_div(log_q, p, reduction="batchmean")) * temperature ** 2


def _optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def client_train(global_model: nn.Module, dataset: ArrayDataset, sampling_rate: float,
                 cfg: Config, device: torch.device, round_idx: int, client_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    model = copy.deepcopy(global_model).to(device)
    model.train()
    optimizer = _optimizer(model, cfg)
    amp_enabled = cfg.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator().manual_seed(cfg.seed + round_idx * 100003 + client_id)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_loss, correct, seen = 0.0, 0, 0
    lower_rates = [r for r in cfg.rates if r < sampling_rate]
    rng = np.random.default_rng(cfg.seed + round_idx * 100003 + client_id)
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_rate = lowpass_resample(x, cfg.base_sample_rate, sampling_rate, cfg.fir_taps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x_rate, sampling_rate)
                loss = F.cross_entropy(logits, y)
                if cfg.lambda_cons > 0 and lower_rates:
                    low_rate = float(rng.choice(lower_rates))
                    x_low = lowpass_resample(x_rate, sampling_rate, low_rate, cfg.fir_taps)
                    shared = model(x_rate, sampling_rate, max_active_rate=low_rate)
                    low_logits = model(x_low, low_rate)
                    loss = loss + cfg.lambda_cons * symmetric_kl(shared, low_logits, cfg.cons_temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at round={round_idx} client={client_id}: {loss.item()}")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
    state = OrderedDict((n, v.detach().cpu().clone()) for n, v in model.state_dict().items())
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"state_dict": state, "num_samples": len(dataset), "sampling_rate": sampling_rate,
            "train_loss": total_loss / max(seen, 1), "train_acc": correct / max(seen, 1),
            "train_time": time.perf_counter() - started, "peak_cuda_memory_mb": peak}


def _vector_norm(named: list[tuple[str, Tensor]], band: int) -> float:
    squared = sum(v.float().pow(2).sum().item() for n, v in named if get_param_band(n) == band)
    return math.sqrt(squared)


def aggregate(global_model: nn.Module, results: list[dict[str, Any]], method: str,
              band_uppers: tuple[float, ...]) -> dict[str, float]:
    global_state = global_model.state_dict()
    total = float(sum(r["num_samples"] for r in results))
    deltas = [{n: r["state_dict"][n] - v.cpu() for n, v in global_state.items()} for r in results]
    diagnostics: dict[str, float] = {}
    new_state: OrderedDict[str, Tensor] = OrderedDict()
    for name, old in global_state.items():
        band = get_param_band(name)
        eligible = list(range(len(results))) if band is None else [i for i, r in enumerate(results)
                    if r["sampling_rate"] / 2 + 1e-7 >= band_uppers[band - 1]]
        chosen = list(range(len(results))) if method == "fedavg" else eligible
        denom = sum(results[i]["num_samples"] for i in chosen)
        update = sum((deltas[i][name] * results[i]["num_samples"] for i in chosen),
                     torch.zeros_like(old, device="cpu")) / (denom + 1e-12) if chosen else torch.zeros_like(old.cpu())
        new_state[name] = old.cpu() + update
    global_model.load_state_dict(new_state)
    for band, upper in enumerate(band_uppers, 1):
        eligible = [i for i, r in enumerate(results) if r["sampling_rate"] / 2 + 1e-7 >= upper]
        mass = float(sum(results[i]["num_samples"] for i in eligible))
        rho = mass / (total + 1e-12)
        normalized, diluted = [], []
        for name, old in global_state.items():
            if get_param_band(name) != band:
                continue
            raw = sum((deltas[i][name] * results[i]["num_samples"] for i in eligible), torch.zeros_like(old.cpu()))
            normalized.append((name, raw / (mass + 1e-12) if eligible else torch.zeros_like(raw)))
            diluted.append((name, raw / (total + 1e-12)))
        norm_support, norm_diluted = _vector_norm(normalized, band), _vector_norm(diluted, band)
        client_norms = [_vector_norm(list(deltas[i].items()), band) for i in eligible]
        actual = diluted if method == "fedavg" else normalized
        diagnostics.update({f"rho_band{band}": rho, f"theoretical_attenuation_band{band}": rho,
                            f"eligible_clients_band{band}": len(eligible),
                            f"mean_eligible_delta_norm_band{band}": float(np.mean(client_norms)) if client_norms else 0.0,
                            f"aggregated_delta_norm_band{band}": _vector_norm(actual, band),
                            f"empirical_update_ratio_band{band}": norm_diluted / (norm_support + 1e-12)})
    return diagnostics
