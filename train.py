from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config, parse_args
from data import (ArrayDataset, assign_rates, load_and_split, make_schedule,
                  partition_clients, stable_hash)
from federated import aggregate, client_train
from metrics import evaluate, plot_run, summarize_outputs
from model import NyqNet
from utils import model_hash, model_stats, resolve_device, seed_everything, write_json


def make_model(x: np.ndarray, y: np.ndarray, cfg: Config) -> NyqNet:
    expected = round(cfg.base_sample_rate * cfg.window_seconds)
    if x.shape[-1] != expected:
        raise ValueError(f"Data length {x.shape[-1]} != base_rate*window_seconds ({expected})")
    return NyqNet(x.shape[1], int(y.max()) + 1, cfg.base_sample_rate,
                  cfg.window_seconds, cfg.rates, cfg.embedding_dim)


def save_client_metadata(path: Path, parts: list[np.ndarray], rates: np.ndarray, y: np.ndarray) -> None:
    rows = []
    classes = np.unique(y)
    for client_id, (indices, rate) in enumerate(zip(parts, rates)):
        counts = {str(int(c)): int((y[indices] == c).sum()) for c in classes}
        rows.append({"client_id": client_id, "num_samples": len(indices), "sampling_rate": rate,
                     "label_distribution": json.dumps(counts, sort_keys=True)})
    pd.DataFrame(rows).to_csv(path, index=False)


def run(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    x, y, train_idx, test_idx = load_and_split(cfg.data, cfg.test_size, cfg.seed)
    parts = partition_clients(y, train_idx, cfg.num_clients, cfg.partition,
                              cfg.dirichlet_alpha, cfg.min_client_samples, cfg.seed)
    client_rates = assign_rates(cfg.num_clients, cfg.rates, cfg.rate_probs, cfg.seed)
    schedule = make_schedule(cfg.rounds, cfg.num_clients, cfg.client_fraction, cfg.seed)
    model = make_model(x, y, cfg)
    hashes = {"client_partitions_hash": stable_hash(parts),
              "sampling_rate_assignment_hash": stable_hash(client_rates),
              "initial_model_hash": model_hash(model),
              "client_selection_schedule_hash": stable_hash(schedule)}
    config_log = cfg.as_dict() | {"fairness_hashes": hashes, "device_resolved": str(device)}
    write_json(run_dir / "config.json", config_log)
    save_client_metadata(run_dir / "client_metadata.csv", parts, client_rates, y)
    print("Fairness hashes:", hashes)
    test_loader = DataLoader(ArrayDataset(x, y, test_idx), batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    num_params, model_bytes = model_stats(model)
    cumulative_bytes, rows = 0, []
    for round_idx, selected in enumerate(schedule, 1):
        round_start = time.perf_counter()
        local_results = []
        for client_id in selected:
            result = client_train(model, ArrayDataset(x, y, parts[client_id]),
                                  float(client_rates[client_id]), cfg, device, round_idx, client_id)
            local_results.append(result)
        aggregation_start = time.perf_counter()
        diagnostics = aggregate(model, local_results, cfg.method, model.band_uppers)
        aggregation_time = time.perf_counter() - aggregation_start
        model.to(device)
        scores = evaluate(model, test_loader, cfg.rates, cfg.base_sample_rate, device, cfg.fir_taps)
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        transmitted = 2 * len(selected) * model_bytes
        cumulative_bytes += transmitted
        row = {"round": round_idx, "method": cfg.method, **scores, **diagnostics,
               "round_time_sec": time.perf_counter() - round_start,
               "mean_client_train_time_sec": float(np.mean([r["train_time"] for r in local_results])),
               "server_aggregation_time_sec": aggregation_time,
               "mean_client_peak_cuda_memory_mb": float(np.mean([r["peak_cuda_memory_mb"] for r in local_results])),
               "max_client_peak_cuda_memory_mb": float(np.max([r["peak_cuda_memory_mb"] for r in local_results])),
               "mean_client_train_loss": float(np.mean([r["train_loss"] for r in local_results])),
               "mean_client_train_acc": float(np.mean([r["train_acc"] for r in local_results])),
               "model_num_params": num_params, "model_size_MB": model_bytes / 2 ** 20,
               "downlink_bytes_per_round": len(selected) * model_bytes,
               "uplink_bytes_per_round": len(selected) * model_bytes,
               "total_bytes_per_round": transmitted,
               "cumulative_communication_MB": cumulative_bytes / 2 ** 20}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "round_metrics.csv", index=False)
        print(f"Round {round_idx:03d}/{cfg.rounds}: acc={scores['accuracy']:.4f} "
              f"macro_f1={scores['macro_f1']:.4f} time={row['round_time_sec']:.2f}s")
    final = rows[-1] | {"seed": cfg.seed, "method": cfg.method, **hashes,
                        "mean_round_time_sec_all": float(np.mean([r["round_time_sec"] for r in rows])),
                        "peak_cuda_memory_mb_all": float(np.max([r["max_client_peak_cuda_memory_mb"] for r in rows]))}
    write_json(run_dir / "final_metrics.json", final)
    plot_run(run_dir / "round_metrics.csv", cfg.rates)
    print_final(final, cfg)


def print_final(final: dict[str, object], cfg: Config) -> None:
    f = lambda key: float(final[key])
    print("=" * 40, f"Method: {cfg.method}", f"Seed: {cfg.seed}",
          f"Final Accuracy: {f('accuracy'):.4f}", f"Final Macro-F1: {f('macro_f1'):.4f}",
          f"Worst-rate Accuracy: {f('worst_rate_accuracy'):.4f}",
          f"Worst-rate Macro-F1: {f('worst_rate_macro_f1'):.4f}", sep="\n")
    for rate in cfg.rates:
        print(f"{rate:g} Hz F1: {f(f'macro_f1_{rate:g}'):.4f}")
    print(f"Communication MB: {f('cumulative_communication_MB'):.2f}",
          f"Mean Round Time: {f('mean_round_time_sec_all'):.2f}",
          f"Peak GPU Memory: {f('peak_cuda_memory_mb_all'):.2f}", "=" * 40, sep="\n")


def summary_cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    p.add_argument("--fedavg_experiment", default="fedavg")
    p.add_argument("--fednyq_experiment", default="fednyq")
    a = p.parse_args(sys.argv[2:])
    experiments = {"fedavg": a.fedavg_experiment, "fednyq": a.fednyq_experiment}
    table = summarize_outputs(a.output_dir, a.seeds, experiments)
    print(table.to_string(index=False))
    for metric, filename in (("accuracy", "accuracy_vs_round.png"), ("macro_f1", "macro_f1_vs_round.png")):
        plt.figure(figsize=(6, 4))
        for method, experiment in experiments.items():
            frames = [pd.read_csv(Path(a.output_dir) / experiment / str(s) / "round_metrics.csv") for s in a.seeds]
            values = np.stack([df[metric].to_numpy() for df in frames])
            plt.plot(frames[0]["round"], values.mean(0), label=method)
        plt.xlabel("Round"); plt.ylabel(metric); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(Path(a.output_dir) / filename, dpi=160); plt.close()


def rate_sweep_summary_cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.8, 0.5, 0.2])
    p.add_argument("--seed", type=int, default=2025)
    a = p.parse_args(sys.argv[2:])
    rows = []
    for fraction in a.fractions:
        for method in ("fedavg", "fednyq"):
            path = Path(a.output_dir) / f"sweep_f{fraction:g}_{method}" / str(a.seed)
            final = json.loads((path / "final_metrics.json").read_text(encoding="utf-8"))
            rows.append({"high_rate_fraction": fraction, "method": method, "accuracy": final["accuracy"],
                         "macro_f1": final["macro_f1"], "worst_rate_f1": final["worst_rate_macro_f1"],
                         "rho_band3": final["rho_band3"],
                         "band3_update_norm": final["aggregated_delta_norm_band3"]})
    df = pd.DataFrame(rows); df.to_csv(Path(a.output_dir) / "rate_sweep.csv", index=False)
    for y, filename in (("accuracy", "performance_vs_high_rate_fraction.png"),
                        ("band3_update_norm", "band_update_norm_vs_rho.png")):
        plt.figure(figsize=(6, 4))
        for method, group in df.groupby("method"):
            x = group["high_rate_fraction"] if y == "accuracy" else group["rho_band3"]
            plt.plot(x, group[y], marker="o", label=method)
        plt.xlabel("High-rate fraction" if y == "accuracy" else "rho_band3"); plt.ylabel(y)
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(Path(a.output_dir) / filename, dpi=160); plt.close()
    print(df.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summary_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == "rate-sweep-summary":
        rate_sweep_summary_cli()
    else:
        run(parse_args())
