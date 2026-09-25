"""Seed-level aggregation and Phase 11 figures."""

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(".venv") / "matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.evaluation_config import ExperimentConfig


def calculate_mean_std(values):
    values = list(values)
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def read_raw_results(filepath):
    path = Path(filepath)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows):
    """Aggregate episodes within seed, then seed estimates across seeds."""
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["training_seed"])].append(row)
    seed_rows = []
    for seed in sorted(by_seed):
        episodes = by_seed[seed]
        captures = [int(row["captured"]) for row in episodes]
        capture_times = [
            float(row["capture_time"]) for row in episodes
            if row.get("capture_time") not in (None, "", "None", "nan")
        ]
        seed_rows.append({
            "training_seed": seed,
            "capture_rate": sum(captures) / len(captures),
            "episode_length": sum(float(row["episode_length"]) for row in episodes) / len(episodes),
            "capture_time": (
                sum(capture_times) / len(capture_times) if capture_times else float("nan")
            ),
            "episode_reward": sum(float(row["episode_reward"]) for row in episodes) / len(episodes),
        })

    summary = {"method": rows[0]["method"] if rows else ""}
    for field in ("capture_rate", "episode_length", "capture_time", "episode_reward"):
        finite = [row[field] for row in seed_rows if math.isfinite(row[field])]
        mean, std = calculate_mean_std(finite) if finite else (float("nan"), float("nan"))
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = std
        summary[f"{field}_valid_seeds"] = len(finite)
    summary["seed_count"] = len(seed_rows)
    return summary, seed_rows


def _write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(config, summaries, field, ylabel, title, filename, ylim=None):
    labels = [row["method"] for row in summaries]
    means = [row[f"{field}_mean"] for row in summaries]
    errors = [row[f"{field}_std"] for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(labels)), means, yerr=errors, alpha=0.7, ecolor="black", capsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(config.plots_dir / filename, dpi=150)
    plt.close(fig)


def _training_curve_source(config):
    rows = []
    for method_key, method_name in (
        ("independent_q", "Independent Q-Learning"),
        ("cooperative_q", "Cooperative Q-Learning"),
    ):
        by_episode = defaultdict(list)
        for path in sorted(config.training_dir.glob(f"{method_key}_seed_*.csv")):
            for row in read_raw_results(path):
                by_episode[int(row["episode"])].append(row)
        for episode in sorted(by_episode):
            episode_rows = by_episode[episode]
            result = {"method": method_name, "episode": episode, "seed_count": len(episode_rows)}
            for field in ("episode_reward", "rolling_capture_rate"):
                mean, std = calculate_mean_std(float(row[field]) for row in episode_rows)
                result[f"{field}_mean"] = mean
                result[f"{field}_std"] = std
            rows.append(result)
    return rows


def _plot_training_curve(config, source, field, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted({row["method"] for row in source})
    for method in methods:
        data = [row for row in source if row["method"] == method]
        x = [row["episode"] for row in data]
        y = [row[f"{field}_mean"] for row in data]
        std = [row[f"{field}_std"] for row in data]
        ax.plot(x, y, label=method)
        ax.fill_between(x, [a - b for a, b in zip(y, std)], [a + b for a, b in zip(y, std)], alpha=0.2)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.plots_dir / filename, dpi=150)
    plt.close(fig)


def aggregate_and_plot(config: ExperimentConfig = None):
    config = config or ExperimentConfig()
    config.summaries_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    seed_summaries = []
    for path in sorted(config.raw_dir.glob("*_results.csv")):
        rows = read_raw_results(path)
        if not rows:
            continue
        summary, per_seed = aggregate_rows(rows)
        summaries.append(summary)
        for row in per_seed:
            seed_summaries.append({"method": summary["method"], **row})

    method_order = {
        "Random": 0, "Heuristic": 1,
        "Independent Q-Learning": 2, "Cooperative Q-Learning": 3,
    }
    summaries.sort(key=lambda row: method_order.get(row["method"], 99))
    seed_summaries.sort(key=lambda row: (method_order.get(row["method"], 99), row["training_seed"]))
    _write_rows(config.summaries_dir / "comparison_summary.csv", summaries)
    _write_rows(config.summaries_dir / "per_seed_summary.csv", seed_summaries)
    if summaries:
        _plot_bar(config, summaries, "capture_rate", "Capture rate", "Capture Rate Comparison", "capture_rate_comparison.png", (0, 1))
        _plot_bar(config, summaries, "episode_length", "Episode length", "Episode Length Comparison", "episode_length_comparison.png")
        _plot_bar(config, summaries, "episode_reward", "Episode reward", "Episode Reward Comparison", "reward_comparison.png")

    training_source = _training_curve_source(config)
    _write_rows(config.summaries_dir / "training_curve_source.csv", training_source)
    if training_source:
        _plot_training_curve(config, training_source, "episode_reward", "Mean episode reward", "training_reward_curve.png")
        _plot_training_curve(config, training_source, "rolling_capture_rate", "100-episode capture rate", "training_capture_curve.png")
    return summaries


if __name__ == "__main__":
    aggregate_and_plot()
