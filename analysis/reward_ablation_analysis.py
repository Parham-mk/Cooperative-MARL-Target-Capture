"""Rebuild Phase 13 summaries and figures from saved data, without training."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from configs.ablation_config import AblationConfig, REWARD_VARIANTS, VARIANT_ORDER
from dataclasses import asdict
from experiments.generate_behavioral_examples import aggregate_behavioral_rows
from experiments.statistical_analysis import aggregate_rows, read_raw_results, plt
from experiments.run_comparison import save_rows


LABELS = ["Full reward", "No distance", "No step penalty", "No capture reward"]
COLORS = ["#355f8d", "#d79028", "#278e73", "#b04b66"]


def mean_sample_std(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return (
        statistics.mean(values) if values else float("nan"),
        statistics.stdev(values) if len(values) > 1 else float("nan"),
    )


def aggregate_variant(rows):
    # Reuse Phase 11 episode -> seed aggregation, with undefined one-seed SD.
    summary, seeds = aggregate_rows([{**row, "method": row["variant"]} for row in rows])
    summary["variant"] = summary.pop("method")
    summary["evaluation_episode_count"] = len(rows)
    for seed in seeds:
        episodes = [r for r in rows if int(r["training_seed"]) == seed["training_seed"]]
        seed["episode_count"] = len(episodes)
        seed["success_count"] = sum(int(r["captured"]) for r in episodes)
    for field in ("capture_rate", "capture_time", "episode_length", "episode_reward"):
        summary[f"{field}_mean"], summary[f"{field}_std"] = mean_sample_std(row[field] for row in seeds)
    return summary, [{"variant": summary["variant"], **row} for row in seeds]


def threshold_episode(history, window=100, threshold=0.50):
    """Return episodes completed, requiring a complete window of real captures."""
    captures = [int(row["captured"]) for row in history]
    count = sum(captures[:window])
    for end in range(window - 1, len(captures)):
        if end >= window:
            count += captures[end] - captures[end - window]
        if count / window >= threshold:
            return int(history[end]["episode"]) + 1
    return None


def _bar_plot(config, summaries, field, ylabel, filename, limit=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, row in enumerate(summaries):
        value = row[f"{field}_mean"]
        sd = row[f"{field}_std"]
        if math.isfinite(value):
            ax.bar(index, value, color=COLORS[index], alpha=0.85,
                   yerr=sd if math.isfinite(sd) else None, capsize=6)
            if field == "capture_time":
                ax.text(index, 0.02, f"valid seeds: {row['capture_time_valid_seeds']}",
                        transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        else:
            ax.text(index, 0.06, "N/A\n0 successful seeds", ha="center",
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(range(4), LABELS)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(ylabel)
    ax.set_title("Frozen evaluation: mean ± sample SD across training seeds")
    if limit:
        ax.set_ylim(*limit)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(config.plots_dir / filename, dpi=150)
    plt.close(fig)


def _learning_plot(config, source):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index, variant in enumerate(VARIANT_ORDER):
        data = [row for row in source if row["variant"] == variant]
        x = [row["episodes_completed"] for row in data]
        for ax, field in zip(axes, ("rolling_capture_rate", "rolling_episode_length")):
            y = [row[f"{field}_mean"] for row in data]
            sd = [row[f"{field}_std"] for row in data]
            ax.plot(x, y, label=LABELS[index], color=COLORS[index])
            ax.fill_between(x, [a - b for a, b in zip(y, sd)],
                            [a + b for a, b in zip(y, sd)], color=COLORS[index], alpha=0.15)
            ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(config.capture_threshold, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel(f"{config.rolling_window}-episode capture rate")
    axes[0].set_title("Training with exploration: mean ± sample SD across seeds")
    axes[0].legend(loc="lower right")
    axes[1].set_ylabel(f"{config.rolling_window}-episode mean length")
    axes[1].set_xlabel("Training episodes completed (first complete window at 100 by default)")
    axes[1].set_ylim(bottom=0)
    if config.train_episodes < config.rolling_window:
        axes[0].text(0.5, 0.7, "No complete rolling windows in this budget", ha="center", transform=axes[0].transAxes)
    fig.tight_layout()
    fig.savefig(config.plots_dir / "ablation_learning_curves.png", dpi=150)
    plt.close(fig)


def analyze_ablation(config):
    summaries, seed_summaries, learning, learning_summary, curves, behavioral = [], [], [], [], [], []
    config.summaries_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANT_ORDER:
        rows = read_raw_results(config.raw_dir / f"{variant}.csv")
        expected = {(seed, episode) for seed in config.seeds for episode in range(config.eval_episodes)}
        observed = {(int(r["training_seed"]), int(r["episode"])) for r in rows}
        if observed != expected or len(rows) != len(expected) or any(r["variant"] != variant for r in rows):
            raise ValueError(f"incomplete or inconsistent evaluation data for {variant}")
        if any(int(r["evaluation_seed"]) != config.evaluation_seed(int(r["training_seed"]), int(r["episode"])) for r in rows):
            raise ValueError(f"evaluation seed schedule differs for {variant}")
        summary, seeds = aggregate_variant(rows)
        summaries.append(summary)
        seed_summaries.extend(seeds)
        by_episode = defaultdict(list)
        variant_learning = []
        for seed in config.seeds:
            history = read_raw_results(config.history_path(variant, seed))
            if [int(r["episode"]) for r in history] != list(range(config.train_episodes)):
                raise ValueError(f"incomplete training history for {variant}, seed {seed}")
            if any(r["variant"] != variant or int(r["training_seed"]) != seed for r in history):
                raise ValueError(f"training history identity differs for {variant}, seed {seed}")
            crossed = threshold_episode(history, config.rolling_window, config.capture_threshold)
            item = {"variant": variant, "training_seed": seed, "reached": int(crossed is not None),
                    "episodes_completed": crossed, "status": "Reached" if crossed is not None else "Not reached"}
            variant_learning.append(item)
            learning.append(item)
            for row in history:
                by_episode[int(row["episode"])].append(row)
        reached = [r["episodes_completed"] for r in variant_learning if r["reached"]]
        mean, sd = mean_sample_std(reached)
        learning_summary.append({"variant": variant, "seed_count": len(config.seeds),
                                 "reached_seed_count": len(reached), "conditional_threshold_episode_mean": mean,
                                 "conditional_threshold_episode_std": sd})
        for episode, values in sorted(by_episode.items()):
            row = {"variant": variant, "episode": episode, "episodes_completed": episode + 1,
                   "seed_count": len(values)}
            for field in ("rolling_capture_rate", "rolling_episode_length", "episode_reward"):
                row[f"{field}_mean"], row[f"{field}_std"] = mean_sample_std(v[field] for v in values)
            curves.append(row)
        behavior = read_raw_results(config.behavioral_dir / f"{variant}_metrics.csv")
        if ({(int(r["training_seed"]), int(r["episode"])) for r in behavior} != expected
                or len(behavior) != len(expected) or any(r["variant"] != variant for r in behavior)):
            raise ValueError(f"incomplete behavioral data for {variant}")
        quantitative = {(r["training_seed"], r["episode"]): r for r in rows}
        for row in behavior:
            match = quantitative[(row["training_seed"], row["episode"])]
            if any(row[field] != match[field] for field in ("evaluation_seed", "captured", "episode_length")):
                raise ValueError("behavioral and quantitative rollouts differ")
        behavioral.extend({**row, "method": variant} for row in behavior)
    bseeds, bsummary = aggregate_behavioral_rows(behavioral)
    for row in bseeds + bsummary:
        row["variant"] = row.pop("method")
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    bseeds.sort(key=lambda r: (order[r["variant"]], r["training_seed"]))
    bsummary.sort(key=lambda r: order[r["variant"]])
    for row in bsummary:
        if row["seed_count"] < 2:
            for field in list(row):
                if field.endswith("_std"):
                    row[field] = float("nan")
    for name, data in (("reward_ablation_summary.csv", summaries), ("per_seed_summary.csv", seed_summaries),
                       ("learning_speed_per_seed.csv", learning), ("learning_speed_summary.csv", learning_summary),
                       ("bar_plot_source.csv", summaries), ("learning_curve_source.csv", curves)):
        save_rows(data, config.summaries_dir / name)
    save_rows(bseeds, config.behavioral_dir / "behavioral_seed_summary.csv")
    save_rows(bsummary, config.behavioral_dir / "behavioral_summary.csv")
    _bar_plot(config, summaries, "capture_rate", "Capture rate", "capture_rate_ablation.png", (0, 1))
    _bar_plot(config, summaries, "capture_time", "Capture time (steps; successes only)", "capture_time_ablation.png")
    _bar_plot(config, summaries, "episode_length", "Episode length (steps; all episodes)", "episode_length_ablation.png")
    _learning_plot(config, curves)
    return summaries


def load_saved_config(output_dir):
    data = json.loads((Path(output_dir) / "summaries/ablation_config.json").read_text(encoding="utf-8"))
    if data["variants"] != {name: asdict(reward) for name, reward in REWARD_VARIANTS.items()}:
        raise ValueError("saved reward definitions differ from the implemented study")
    return AblationConfig(**{name: data[name] for name in AblationConfig.__dataclass_fields__ if name in data} | {"output_dir": Path(output_dir)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablations"))
    args = parser.parse_args()
    analyze_ablation(load_saved_config(args.output_dir))
    from experiments.reproducibility import create_manifest
    create_manifest(args.output_dir)
    print(f"Rebuilt Phase 13 analysis in {args.output_dir}")


if __name__ == "__main__":
    main()
