"""Generate reproducible Phase 12 trajectories, metrics, plots, and GIFs."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from agents.heuristic_agent import HeuristicAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.behavioral_metrics import calculate_all_behavioral_metrics
from analysis.trajectory_analysis import TrajectoryRecorder
from configs.evaluation_config import ExperimentConfig
from env.rewards import RewardCalculator
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import make_observations, q_table_snapshot
from experiments.statistical_analysis import calculate_mean_std
from visualization.gif_generator import GIFGenerator
from visualization.renderer import GridWorldRenderer


SUCCESS_RULE = (
    "successful episode with capture time closest to the method median; "
    "ties broken by training seed then evaluation seed"
)
FAILURE_RULE = "first failure ordered by training seed then evaluation seed"


def _write_rows(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_with_recorder(
    agent0,
    agent1,
    method_name: str,
    training_seed: int,
    eval_episodes: int,
    grid_size: int,
    max_steps: int,
    checkpoint: str = "N/A",
    evaluation_seed_base: int = 10_000_000,
    seed_index: int = 0,
):
    """Evaluate a frozen policy and return complete real trajectories."""
    env = TargetCaptureEnv(grid_size, max_steps)
    reward_calc = RewardCalculator()
    old_eps = [getattr(agent0, "epsilon", None), getattr(agent1, "epsilon", None)]
    before = [q_table_snapshot(agent0), q_table_snapshot(agent1)]
    for agent in (agent0, agent1):
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0
    recorders = []
    try:
        for episode in range(eval_episodes):
            eval_seed = evaluation_seed_base + seed_index * eval_episodes + episode
            recorder = TrajectoryRecorder()
            recorder.set_metadata(
                method_name, training_seed, eval_seed, grid_size, max_steps, checkpoint
            )
            state = env.reset(seed=eval_seed)
            recorder.record_step(
                0, state, {"agent_0": "START", "agent_1": "START"},
                {"target_action": "START"}, 0.0,
            )
            while True:
                obs0, obs1 = make_observations(state)
                action0 = agent0.select_action(obs0)
                action1 = agent1.select_action(obs1)
                next_state, info = env.step({"agent_0": action0, "agent_1": action1})
                reward = reward_calc.calculate(
                    [env.agent_0, env.agent_1], env.target, state, info["captured"]
                )["total_reward"]
                recorder.record_step(
                    env.current_step, next_state,
                    {"agent_0": action0, "agent_1": action1}, info, reward,
                )
                state = next_state
                if info["terminated"] or info["truncated"]:
                    break
            recorder.metadata["episode"] = episode
            recorders.append(recorder)
    finally:
        for agent, epsilon in zip((agent0, agent1), old_eps):
            if epsilon is not None:
                agent.epsilon = epsilon
    if before != [q_table_snapshot(agent0), q_table_snapshot(agent1)]:
        raise RuntimeError("behavioral evaluation mutated a learned Q-table")
    return recorders


def load_independent_q(config: ExperimentConfig, seed: int, seed_index: int):
    base = config.checkpoints_dir / "independent_q"
    checkpoint0 = base / f"seed_{seed}_agent0.pkl"
    checkpoint1 = base / f"seed_{seed}_agent1.pkl"
    action_seed = config.action_seed_base + seed_index * 2
    agent0 = QLearningAgent(epsilon=0.0, seed=action_seed)
    agent1 = QLearningAgent(epsilon=0.0, seed=action_seed + 1)
    agent0.load(checkpoint0)
    agent1.load(checkpoint1)
    return agent0, agent1, f"{checkpoint0.resolve()}|{checkpoint1.resolve()}"


def load_cooperative_q(config: ExperimentConfig, seed: int, seed_index: int):
    checkpoint = config.checkpoints_dir / "cooperative_q" / f"seed_{seed}.pkl"
    table = SharedQTable()
    table.load(checkpoint)
    action_seed = config.action_seed_base + seed_index * 2
    agents = (
        SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=action_seed),
        SharedQAgent(table, "agent_1", "agent_0", epsilon=0.0, seed=action_seed + 1),
    )
    return *agents, str(checkpoint.resolve())


def find_representative_episodes(recorders):
    successes = [r for r in recorders if r.trajectory[-1]["captured"]]
    failures = [r for r in recorders if not r.trajectory[-1]["captured"]]
    success = None
    median_time = None
    if successes:
        median_time = statistics.median(r.trajectory[-1]["step"] for r in successes)
        success = min(
            successes,
            key=lambda r: (
                abs(r.trajectory[-1]["step"] - median_time),
                r.metadata["training_seed"],
                r.metadata["evaluation_seed"],
            ),
        )
    failure = min(
        failures,
        key=lambda r: (r.metadata["training_seed"], r.metadata["evaluation_seed"]),
        default=None,
    )
    return success, failure, median_time


def aggregate_behavioral_rows(rows):
    fields = [
        "captured", "episode_length", "mean_target_distance",
        "mean_hunter_separation", "simultaneous_adjacency_fraction",
        "distinct_side_fraction",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["training_seed"]))].append(row)
    seed_rows = []
    for (method, seed), episodes in sorted(grouped.items()):
        output = {"method": method, "training_seed": seed, "episode_count": len(episodes)}
        for field in fields:
            output[field] = sum(float(row[field]) for row in episodes) / len(episodes)
        seed_rows.append(output)

    summaries = []
    for method in sorted({row["method"] for row in seed_rows}):
        seeds = [row for row in seed_rows if row["method"] == method]
        summary = {"method": method, "seed_count": len(seeds)}
        for field in fields:
            mean, std = calculate_mean_std(row[field] for row in seeds)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
        summaries.append(summary)
    return seed_rows, summaries


def generate_behavioral_evidence(config: ExperimentConfig):
    behavioral_dir = config.output_dir / "behavioral"
    trajectories_dir = config.output_dir / "trajectories"
    gifs_dir = config.output_dir / "gifs"
    for path in (behavioral_dir, trajectories_dir, gifs_dir):
        path.mkdir(parents=True, exist_ok=True)

    all_recorders = defaultdict(list)
    metrics = []
    for seed_index, seed in enumerate(config.seeds):
        action_seed = config.action_seed_base + seed_index * 2
        methods = [
            ("Random", RandomAgent(action_seed), RandomAgent(action_seed + 1), "N/A"),
            ("Heuristic", HeuristicAgent(), HeuristicAgent(), "N/A"),
            ("Independent Q-Learning", *load_independent_q(config, seed, seed_index)),
            ("Cooperative Q-Learning", *load_cooperative_q(config, seed, seed_index)),
        ]
        for method, agent0, agent1, checkpoint in methods:
            recorders = evaluate_with_recorder(
                agent0, agent1, method, seed, config.eval_episodes,
                config.grid_size, config.max_steps, checkpoint,
                config.evaluation_seed_base, seed_index,
            )
            all_recorders[method].extend(recorders)
            for recorder in recorders:
                row = calculate_all_behavioral_metrics(recorder.trajectory)
                final = recorder.trajectory[-1]
                row.update({
                    "method": method,
                    "training_seed": seed,
                    "evaluation_seed": recorder.metadata["evaluation_seed"],
                    "episode": recorder.metadata["episode"],
                    "captured": int(final["captured"]),
                    "episode_length": int(final["step"]),
                })
                metrics.append(row)

    method_order = {
        "Random": 0, "Heuristic": 1,
        "Independent Q-Learning": 2, "Cooperative Q-Learning": 3,
    }
    metrics.sort(key=lambda row: (
        method_order[row["method"]], row["training_seed"], row["evaluation_seed"]
    ))
    _write_rows(behavioral_dir / "behavioral_metrics.csv", metrics)
    seed_rows, summaries = aggregate_behavioral_rows(metrics)
    seed_rows.sort(key=lambda row: (method_order[row["method"]], row["training_seed"]))
    summaries.sort(key=lambda row: method_order[row["method"]])
    _write_rows(behavioral_dir / "behavioral_seed_summary.csv", seed_rows)
    _write_rows(behavioral_dir / "behavioral_summary.csv", summaries)

    selection_rows = []
    renderer = GridWorldRenderer(config.grid_size)
    gif_generator = GIFGenerator(config.grid_size)
    for method in method_order:
        success, failure, median_time = find_representative_episodes(all_recorders[method])
        safe = method.lower().replace(" ", "_").replace("-", "_")
        if success:
            success.metadata["selection_rule"] = SUCCESS_RULE
            success.metadata["method_median_success_time"] = median_time
            success_path = trajectories_dir / f"{safe}_success.json"
            success.save(success_path)
            renderer.plot_static_trajectory(
                success.trajectory, f"{method} representative success",
                behavioral_dir / f"{safe}_trajectory.png",
            )
            gif_generator.generate(
                success.trajectory, method, gifs_dir / f"{safe}_success.gif"
            )
            selection_rows.append({
                "method": method, "outcome": "success",
                "training_seed": success.metadata["training_seed"],
                "evaluation_seed": success.metadata["evaluation_seed"],
                "episode_length": success.trajectory[-1]["step"],
                "method_median_success_time": median_time,
                "selection_rule": SUCCESS_RULE,
                "trajectory": str(success_path),
            })
        else:
            selection_rows.append({
                "method": method, "outcome": "success", "training_seed": "",
                "evaluation_seed": "", "episode_length": "",
                "method_median_success_time": "",
                "selection_rule": "no successful episode observed",
                "trajectory": "",
            })
        if method in ("Independent Q-Learning", "Cooperative Q-Learning"):
            if failure:
                failure.metadata["selection_rule"] = FAILURE_RULE
                failure_path = trajectories_dir / f"{safe}_failure.json"
                failure.save(failure_path)
                gif_generator.generate(
                    failure.trajectory, f"{method} failure",
                    gifs_dir / f"{safe}_failure.gif",
                )
                selection_rows.append({
                    "method": method, "outcome": "failure",
                    "training_seed": failure.metadata["training_seed"],
                    "evaluation_seed": failure.metadata["evaluation_seed"],
                    "episode_length": failure.trajectory[-1]["step"],
                    "method_median_success_time": median_time,
                    "selection_rule": FAILURE_RULE,
                    "trajectory": str(failure_path),
                })
            else:
                selection_rows.append({
                    "method": method, "outcome": "failure", "training_seed": "",
                    "evaluation_seed": "", "episode_length": "",
                    "method_median_success_time": median_time if median_time is not None else "",
                    "selection_rule": "no failure observed in the complete evaluation dataset",
                    "trajectory": "",
                })
    _write_rows(behavioral_dir / "representative_selection.csv", selection_rows)
    return {"metrics": metrics, "seed_summary": seed_rows, "summary": summaries, "selection": selection_rows}


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Phase 12 behavioral evidence")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = _parse_args()
    config = ExperimentConfig(
        grid_size=args.grid_size, max_steps=args.max_steps,
        eval_episodes=args.eval_episodes, seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(f"Generating Phase 12 evidence in {config.output_dir}")
    generate_behavioral_evidence(config)
    print(f"Behavioral summary: {config.output_dir / 'behavioral' / 'behavioral_summary.csv'}")


if __name__ == "__main__":
    main()
