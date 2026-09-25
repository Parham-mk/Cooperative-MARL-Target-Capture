"""Reproducible Phase 11 comparison for all four policies."""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

from agents.heuristic_agent import HeuristicAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from configs.evaluation_config import ExperimentConfig
from env.rewards import RewardCalculator
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import evaluate_policy
from experiments.statistical_analysis import aggregate_and_plot


def save_rows(rows: Iterable[dict], filename) -> None:
    rows = list(rows)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_raw_results(results, filename):
    """Compatibility wrapper retained for existing callers."""
    save_rows(results, filename)


def _rolling_capture(captures: List[int], window: int = 100) -> float:
    values = captures[-window:]
    return sum(values) / len(values)


def _training_seed(config: ExperimentConfig, seed_index: int, episode: int) -> int:
    return seed_index * config.training_seed_stride + episode


def train_independent_q(seed: int, config: ExperimentConfig = None, return_history: bool = False):
    config = config or ExperimentConfig()
    seed_index = config.seeds.index(seed) if seed in config.seeds else seed
    env = TargetCaptureEnv(config.grid_size, config.max_steps)
    reward_calc = RewardCalculator()
    agent0 = QLearningAgent(seed=seed * 2)
    agent1 = QLearningAgent(seed=seed * 2 + 1)
    history = []
    captures = []

    for episode in range(config.train_episodes):
        state = env.reset(seed=_training_seed(config, seed_index, episode))
        episode_reward = 0.0
        while True:
            obs0 = {"agent_position": state["agent_0"], "target_position": state["target"]}
            obs1 = {"agent_position": state["agent_1"], "target_position": state["target"]}
            action0 = agent0.select_action(obs0)
            action1 = agent1.select_action(obs1)
            next_state, info = env.step({"agent_0": action0, "agent_1": action1})
            reward = reward_calc.calculate(
                [env.agent_0, env.agent_1], env.target, state, info["captured"]
            )["total_reward"]
            episode_reward += reward
            next_obs0 = {"agent_position": next_state["agent_0"], "target_position": next_state["target"]}
            next_obs1 = {"agent_position": next_state["agent_1"], "target_position": next_state["target"]}
            agent0.update(obs0, action0, reward, next_obs0, terminated=info["terminated"])
            agent1.update(obs1, action1, reward, next_obs1, terminated=info["terminated"])
            state = next_state
            if info["terminated"] or info["truncated"]:
                break

        agent0.decay_epsilon()
        agent1.decay_epsilon()
        captures.append(int(info["captured"]))
        history.append({
            "method": "Independent Q-Learning",
            "training_seed": seed,
            "episode": episode,
            "episode_reward": episode_reward,
            "captured": captures[-1],
            "episode_length": info["step"],
            "epsilon": agent0.epsilon,
            "rolling_capture_rate": _rolling_capture(captures),
        })

    base = config.checkpoints_dir / "independent_q"
    agent0.save(base / f"seed_{seed}_agent0.pkl")
    agent1.save(base / f"seed_{seed}_agent1.pkl")
    save_rows(history, config.training_dir / f"independent_q_seed_{seed}.csv")
    if return_history:
        return agent0, agent1, history
    return agent0, agent1


def train_cooperative_q(seed: int, config: ExperimentConfig = None, return_history: bool = False):
    config = config or ExperimentConfig()
    seed_index = config.seeds.index(seed) if seed in config.seeds else seed
    env = TargetCaptureEnv(config.grid_size, config.max_steps)
    reward_calc = RewardCalculator()
    table = SharedQTable()
    agent0 = SharedQAgent(table, "agent_0", "agent_1", seed=seed * 2)
    agent1 = SharedQAgent(table, "agent_1", "agent_0", seed=seed * 2 + 1)
    history = []
    captures = []

    for episode in range(config.train_episodes):
        state = env.reset(seed=_training_seed(config, seed_index, episode))
        episode_reward = 0.0
        while True:
            # Both actions are selected from the same pre-step state.
            action0 = agent0.select_action(state)
            action1 = agent1.select_action(state)
            next_state, info = env.step({"agent_0": action0, "agent_1": action1})
            reward = reward_calc.calculate(
                [env.agent_0, env.agent_1], env.target, state, info["captured"]
            )["total_reward"]
            episode_reward += reward
            agent0.update(state, action0, reward, next_state, info["terminated"])
            agent1.update(state, action1, reward, next_state, info["terminated"])
            state = next_state
            if info["terminated"] or info["truncated"]:
                break

        agent0.decay_epsilon()
        agent1.decay_epsilon()
        captures.append(int(info["captured"]))
        history.append({
            "method": "Cooperative Q-Learning",
            "training_seed": seed,
            "episode": episode,
            "episode_reward": episode_reward,
            "captured": captures[-1],
            "episode_length": info["step"],
            "epsilon": agent0.epsilon,
            "rolling_capture_rate": _rolling_capture(captures),
        })

    path = config.checkpoints_dir / "cooperative_q" / f"seed_{seed}.pkl"
    table.save(path)
    save_rows(history, config.training_dir / f"cooperative_q_seed_{seed}.csv")
    if return_history:
        return agent0, agent1, history
    return agent0, agent1


def load_independent(config: ExperimentConfig, seed: int, seed_index: int):
    action_seed = config.action_seed_base + seed_index * 2
    agent0 = QLearningAgent(epsilon=0.0, seed=action_seed)
    agent1 = QLearningAgent(epsilon=0.0, seed=action_seed + 1)
    base = config.checkpoints_dir / "independent_q"
    agent0.load(base / f"seed_{seed}_agent0.pkl")
    agent1.load(base / f"seed_{seed}_agent1.pkl")
    return agent0, agent1


def load_cooperative(config: ExperimentConfig, seed: int, seed_index: int):
    table = SharedQTable()
    table.load(config.checkpoints_dir / "cooperative_q" / f"seed_{seed}.pkl")
    action_seed = config.action_seed_base + seed_index * 2
    return (
        SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=action_seed),
        SharedQAgent(table, "agent_1", "agent_0", epsilon=0.0, seed=action_seed + 1),
    )


def _evaluate(config, agent0, agent1, method, seed, seed_index):
    return evaluate_policy(
        agent0, agent1, method, seed, config.eval_episodes,
        config.grid_size, config.max_steps,
        evaluation_seed_base=config.evaluation_seed_base,
        seed_index=seed_index,
    )


def run_comparison(config: ExperimentConfig) -> dict:
    """Run training, frozen evaluation, aggregation, and plotting."""
    for directory in (
        config.raw_dir, config.summaries_dir, config.plots_dir,
        config.checkpoints_dir, config.training_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    with (config.summaries_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)

    all_results = {
        "Random": [],
        "Heuristic": [],
        "Independent Q-Learning": [],
        "Cooperative Q-Learning": [],
    }
    for seed_index, seed in enumerate(config.seeds):
        action_seed = config.action_seed_base + seed_index * 2
        all_results["Random"].extend(_evaluate(
            config, RandomAgent(action_seed), RandomAgent(action_seed + 1),
            "Random", seed, seed_index,
        ))
        all_results["Heuristic"].extend(_evaluate(
            config, HeuristicAgent(), HeuristicAgent(), "Heuristic", seed, seed_index,
        ))

        train_independent_q(seed, config)
        independent = load_independent(config, seed, seed_index)
        all_results["Independent Q-Learning"].extend(_evaluate(
            config, *independent, "Independent Q-Learning", seed, seed_index,
        ))

        train_cooperative_q(seed, config)
        cooperative = load_cooperative(config, seed, seed_index)
        all_results["Cooperative Q-Learning"].extend(_evaluate(
            config, *cooperative, "Cooperative Q-Learning", seed, seed_index,
        ))

    filenames = {
        "Random": "random_results.csv",
        "Heuristic": "heuristic_results.csv",
        "Independent Q-Learning": "independent_q_results.csv",
        "Cooperative Q-Learning": "cooperative_q_results.csv",
    }
    for method, rows in all_results.items():
        save_rows(rows, config.raw_dir / filenames[method])
    summary = aggregate_and_plot(config)
    return {"results": all_results, "summary": summary}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the reproducible Phase 11 comparison")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = _parse_args()
    config = ExperimentConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    print(f"Running Phase 11 comparison in {config.output_dir}")
    run_comparison(config)
    print(f"Completed Phase 11 comparison: {config.summaries_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
