"""CLI for frozen, seeded independent Q-learning evaluation."""

import argparse
from pathlib import Path

from agents.q_learning_agent import QLearningAgent
from experiments.evaluation_utils import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Independent Q-Learning Agents")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent0-path", type=Path, default=Path("results/checkpoints/agent_0_q_learning.pkl"))
    parser.add_argument("--agent1-path", type=Path, default=Path("results/checkpoints/agent_1_q_learning.pkl"))
    args = parser.parse_args()
    agent0 = QLearningAgent(epsilon=0.0, seed=args.seed * 2)
    agent1 = QLearningAgent(epsilon=0.0, seed=args.seed * 2 + 1)
    agent0.load(args.agent0_path)
    agent1.load(args.agent1_path)
    rows = evaluate_policy(
        agent0, agent1, "Independent Q-Learning", args.seed,
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
