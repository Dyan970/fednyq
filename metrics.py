from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from data import lowpass_resample


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, rates: tuple[float, ...], base_rate: float,
             device: torch.device, fir_taps: int) -> dict[str, float]:
    model.eval()
    rate_results: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for rate in rates:
        ys, predictions, loss = [], [], 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_rate = lowpass_resample(x, base_rate, rate, fir_taps)
            logits = model(x_rate, rate)
            loss += criterion(logits, y).item()
            ys.append(y.cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
        true = np.concatenate(ys) if ys else np.array([], dtype=int)
        pred = np.concatenate(predictions) if predictions else np.array([], dtype=int)
        rate_results[rate] = (true, pred, loss / max(len(true), 1))
    high = max(rates)
    true, pred, test_loss = rate_results[high]
    out = {
        "test_loss": test_loss,
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(true, pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
    }
    for rate, (yr, pr, _) in rate_results.items():
        key = f"{rate:g}"
        out[f"accuracy_{key}"] = accuracy_score(yr, pr) if len(yr) else np.nan
        out[f"macro_f1_{key}"] = f1_score(yr, pr, average="macro", zero_division=0) if len(yr) else np.nan
    accs = [out[f"accuracy_{r:g}"] for r in rates]
    f1s = [out[f"macro_f1_{r:g}"] for r in rates]
    out["worst_rate_accuracy"] = float(np.nanmin(accs))
    out["worst_rate_macro_f1"] = float(np.nanmin(f1s))
    out["high_low_accuracy_gap"] = accs[-1] - accs[0]
    out["high_low_f1_gap"] = f1s[-1] - f1s[0]
    return out


def plot_run(csv_path: Path, rates: tuple[float, ...]) -> None:
    df = pd.read_csv(csv_path)
    plots = [
        ("accuracy", "Accuracy", "accuracy_vs_round.png"),
        ("macro_f1", "Macro-F1", "macro_f1_vs_round.png"),
        ("cumulative_communication_MB", "Communication (MB)", "communication_vs_round.png"),
        ("round_time_sec", "Seconds", "training_time_vs_round.png"),
    ]
    for column, ylabel, filename in plots:
        if column in df:
            plt.figure(figsize=(6, 4)); plt.plot(df["round"], df[column], marker="o", ms=3)
            plt.xlabel("Round"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(csv_path.parent / filename, dpi=160); plt.close()
    plt.figure(figsize=(6, 4))
    for rate in rates:
        col = f"macro_f1_{rate:g}"
        if col in df:
            plt.plot(df["round"], df[col], label=f"{rate:g} Hz")
    plt.xlabel("Round"); plt.ylabel("Macro-F1"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(csv_path.parent / "per_rate_f1.png", dpi=160); plt.close()
    plt.figure(figsize=(5, 5))
    for band in (1, 2, 3):
        x, y = f"rho_band{band}", f"empirical_update_ratio_band{band}"
        if x in df and y in df:
            plt.scatter(df[x], df[y], s=18, label=f"band{band}")
    plt.plot([0, 1], [0, 1], "k--", label="y=x")
    plt.xlabel("Support mass rho"); plt.ylabel("Empirical update ratio"); plt.legend(); plt.tight_layout()
    plt.savefig(csv_path.parent / "rho_vs_update_ratio.png", dpi=160); plt.close()


def summarize_outputs(output_dir: str, seeds: list[int], experiments: dict[str, str]) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "worst_rate_accuracy", "worst_rate_macro_f1"]
    rows: list[dict[str, object]] = []
    raw: dict[str, dict[str, list[float]]] = {}
    for method, experiment in experiments.items():
        raw[method] = {m: [] for m in metrics}
        for seed in seeds:
            with (Path(output_dir) / experiment / str(seed) / "final_metrics.json").open(encoding="utf-8") as f:
                values = json.load(f)
            for metric in metrics:
                raw[method][metric].append(float(values[metric]))
        for metric, vals in raw[method].items():
            a = np.asarray(vals); sem = stats.sem(a) if len(a) > 1 else np.nan
            ci = stats.t.ppf(.975, len(a) - 1) * sem if len(a) > 1 else np.nan
            rows.append({"method": method, "metric": metric, "mean": a.mean(), "std": a.std(ddof=1),
                         "ci95_low": a.mean() - ci, "ci95_high": a.mean() + ci})
    if "fedavg" in raw and "fednyq" in raw:
        for metric in metrics:
            delta = np.asarray(raw["fednyq"][metric]) - np.asarray(raw["fedavg"][metric])
            sem = stats.sem(delta) if len(delta) > 1 else np.nan
            ci = stats.t.ppf(.975, len(delta) - 1) * sem if len(delta) > 1 else np.nan
            pvalue = stats.ttest_rel(raw["fednyq"][metric], raw["fedavg"][metric]).pvalue if len(delta) > 1 else np.nan
            rows.append({"method": "fednyq-vs-fedavg", "metric": metric, "paired_delta_mean": delta.mean(),
                         "paired_delta_std": delta.std(ddof=1), "paired_delta_ci95_low": delta.mean() - ci,
                         "paired_delta_ci95_high": delta.mean() + ci, "paired_ttest_pvalue": pvalue})
    result = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(output_dir) / "summary.csv", index=False)
    return result
