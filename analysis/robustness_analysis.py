"""Seed-level robustness summaries and plots regenerated from saved raw rollouts."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from configs.robustness_config import METHODS, PRIMARY_CONDITIONS
from experiments.statistical_analysis import _write_rows, read_raw_results
import matplotlib.pyplot as plt


METRICS = ("capture_rate", "capture_time", "episode_length", "episode_reward", "q_coverage", "unseen_state_rate")


def mean_std(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return (statistics.mean(values) if values else float("nan"),
            statistics.stdev(values) if len(values) > 1 else float("nan"), len(values))


def aggregate_robustness(rows):
    """Aggregate visits within seed; uncertainty is sample SD of seed estimates."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["condition"], int(row["training_seed"]))].append(row)
    per_seed = []
    for (method, condition, seed), episodes in sorted(groups.items()):
        count = len(episodes)
        queries = sum(int(r["state_queries"]) for r in episodes)
        seen = sum(int(r["seen_state_queries"]) for r in episodes)
        successes = [r for r in episodes if int(r["captured"])]
        per_seed.append({
            "method": method, "condition": condition, "training_seed": seed,
            "episode_count": count, "success_count": len(successes), "state_queries": queries,
            "seen_state_queries": seen, "capture_rate": len(successes) / count,
            "capture_time": statistics.mean(float(r["capture_time"]) for r in successes) if successes else float("nan"),
            "episode_length": statistics.mean(float(r["episode_length"]) for r in episodes),
            "episode_reward": statistics.mean(float(r["episode_reward"]) for r in episodes),
            "q_coverage": seen / queries, "unseen_state_rate": 1 - seen / queries,
        })
    baselines = {(r["method"], r["training_seed"]): r for r in per_seed if r["condition"] == "standard"}
    for row in per_seed:
        baseline = baselines[(row["method"], row["training_seed"])]
        for field in METRICS:
            row[f"delta_{field}"] = row[field] - baseline[field]
    summaries = []
    for method, condition in sorted({(r["method"], r["condition"]) for r in per_seed}):
        seeds = [r for r in per_seed if r["method"] == method and r["condition"] == condition]
        summary = {"method": method, "condition": condition, "seed_count": len(seeds),
                   "evaluation_episode_count": sum(r["episode_count"] for r in seeds)}
        for field in (*METRICS, *(f"delta_{field}" for field in METRICS)):
            mean, std, valid = mean_std(r[field] for r in seeds)
            summary.update({f"{field}_mean": mean, f"{field}_std": std, f"{field}_valid_seeds": valid})
        summaries.append(summary)
    return per_seed, summaries


def validate_rows(rows, saved):
    """Reject missing, unmatched, or internally inconsistent saved episodes."""
    source, protocol = saved["source_comparison"], saved["protocol"]
    counts = {"standard": protocol["eval_episodes"], "held_out_seeds": protocol["eval_episodes"],
              "held_out_initial_states": protocol["initial_state_cases"]}
    counts.update({f"stress:{case['name']}": protocol["stress_repeats"] for case in saved["stress_cases"]})
    expected = {(method, condition, seed, ep) for method in METHODS for condition, count in counts.items()
                for seed in source["seeds"] for ep in range(count)}
    observed, paired = set(), {}
    for row in rows:
        key = (row["method"], row["condition"], int(row["training_seed"]), int(row["episode"]))
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate robustness episode")
        observed.add(key)
        length, captured = int(row["episode_length"]), int(row["captured"])
        seen, queries = int(row["seen_state_queries"]), int(row["state_queries"])
        if not 1 <= length <= source["max_steps"] or captured not in (0, 1):
            raise ValueError("invalid episode outcome")
        if captured:
            if float(row["capture_time"]) != length:
                raise ValueError("capture time must equal successful episode length")
        elif row["capture_time"] not in (None, "", "None") or length != source["max_steps"]:
            raise ValueError("failed episode must truncate with missing capture time")
        if queries != 2 * length or not 0 <= seen <= queries:
            raise ValueError("invalid action-query denominator")
        if not math.isclose(float(row["q_state_coverage"]), seen / queries) or not math.isclose(float(row["unseen_state_rate"]), 1 - seen / queries):
            raise ValueError("coverage disagrees with action-query counts")
        if not math.isfinite(float(row["episode_reward"])):
            raise ValueError("invalid episode reward")
        paired_key = key[1:]
        start = tuple(int(row[f"initial_{name}_{axis}"]) for name in ("agent_0", "agent_1", "target") for axis in ("x", "y"))
        signature = (int(row["evaluation_seed"]), int(row["action_seed_0"]), int(row["action_seed_1"]), start)
        if paired_key in paired and paired[paired_key] != signature:
            raise ValueError("methods have unmatched seeds or initial states")
        paired[paired_key] = signature
    if observed != expected:
        raise ValueError("robustness episode inventory is incomplete")


def geometry_summary(rows):
    """Descriptive start geometry, conditional on outcome, aggregated by seed."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["condition"], int(row["captured"]), int(row["training_seed"]))].append(row)
    seed_rows = []
    fields = ("initial_target_distance_sum", "initial_hunter_separation", "initial_target_boundary_distance")
    for (method, condition, outcome, seed), episodes in sorted(groups.items()):
        seed_rows.append({"method": method, "condition": condition, "captured": outcome,
                          "training_seed": seed, "episode_count": len(episodes),
                          **{field: statistics.mean(float(r[field]) for r in episodes) for field in fields}})
    summary = []
    for method, condition, outcome in sorted({(r["method"], r["condition"], r["captured"]) for r in seed_rows}):
        seeds = [r for r in seed_rows if (r["method"], r["condition"], r["captured"]) == (method, condition, outcome)]
        row = {"method": method, "condition": condition, "captured": outcome,
               "seed_count": len(seeds), "episode_count": sum(r["episode_count"] for r in seeds)}
        for field in fields:
            mean, std, _ = mean_std(r[field] for r in seeds)
            row.update({f"{field}_mean": mean, f"{field}_std": std})
        summary.append(row)
    return summary


def _plot(root, summaries, conditions, labels, field, ylabel, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = .36
    for index, method in enumerate(METHODS):
        selected = [next(r for r in summaries if r["method"] == method and r["condition"] == c) for c in conditions]
        ax.bar([x + (index - .5) * width for x in range(len(conditions))],
               [r[f"{field}_mean"] for r in selected], width,
               yerr=[r[f"{field}_std"] if math.isfinite(r[f"{field}_std"]) else 0 for r in selected],
               capsize=4, label=method)
    ax.set_xticks(range(len(conditions)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title("Frozen original policies · mean ± sample SD across training seeds")
    ax.legend(fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    if field.startswith("delta_"):
        ax.axhline(0, color="black", linewidth=.8)
    fig.tight_layout()
    fig.savefig(root / "plots" / filename, dpi=150)
    plt.close(fig)


def analyze_robustness(output_dir):
    """Rebuild all quantitative summaries and plots from one saved study."""
    root = Path(output_dir)
    saved = json.loads((root / "summaries" / "robustness_config.json").read_text(encoding="utf-8"))
    rows = read_raw_results(root / "raw" / "robustness_raw.csv")
    validate_rows(rows, saved)
    seed_rows, summaries = aggregate_robustness(rows)
    _write_rows(root / "summaries" / "per_seed_summary.csv", seed_rows)
    _write_rows(root / "summaries" / "robustness_summary.csv", summaries)
    _write_rows(root / "summaries" / "plot_source.csv", summaries)
    _write_rows(root / "summaries" / "initial_geometry_summary.csv", geometry_summary(rows))
    (root / "plots").mkdir(parents=True, exist_ok=True)
    labels = ("Standard", "Fresh seeds", "Held-out initial configurations")
    _plot(root, summaries, PRIMARY_CONDITIONS, labels, "capture_rate", "Capture rate", "capture_rate_robustness.png", (0, 1.08))
    _plot(root, summaries, PRIMARY_CONDITIONS[1:], labels[1:], "delta_capture_rate", "Capture-rate change from standard", "performance_drop.png")
    _plot(root, summaries, PRIMARY_CONDITIONS, labels, "q_coverage", "Seen action queries / all action queries", "q_table_coverage.png", (0, 1.08))
    cases = [case["name"] for case in saved["stress_cases"]]
    _plot(root, summaries, [f"stress:{case}" for case in cases], [case.replace("_", "\n") for case in cases],
          "capture_rate", "Capture rate (fixed spatial probes)", "stress_case_capture_rate.png", (0, 1.08))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="existing robustness study with raw rollouts and saved configuration")
    args = parser.parse_args()
    analyze_robustness(args.output_dir)


if __name__ == "__main__":
    main()
