"""CLI for frozen, seeded cooperative Q-learning evaluation."""

import argparse
from pathlib import Path

from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from experiments.evaluation_utils import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shared-Policy Cooperative Q-Learning")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path("results/checkpoints/cooperative_q_learning.pkl"))
    args = parser.parse_args()
    table = SharedQTable()
    table.load(args.model_path)
    agent0 = SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=args.seed * 2)
    agent1 = SharedQAgent(table, "agent_1", "agent_0", epsilon=0.0, seed=args.seed * 2 + 1)
    rows = evaluate_policy(
        agent0, agent1, "Cooperative Q-Learning", args.seed,
        args.episodes, args.grid_size, args.max_steps,
    )
    successes = sum(row["captured"] for row in rows)
    capture_times = [row["capture_time"] for row in rows if row["capture_time"] is not None]
    print(f"Capture Rate: {successes / len(rows):.4f}")
    print(f"Mean Episode Length: {sum(row['episode_length'] for row in rows) / len(rows):.4f}")
    print(f"Mean Capture Time: {sum(capture_times) / len(capture_times):.4f}" if capture_times else "Mean Capture Time: N/A")
    print(f"Mean Episode Reward: {sum(row['episode_reward'] for row in rows) / len(rows):.4f}")


if __name__ == "__main__":
    main()
