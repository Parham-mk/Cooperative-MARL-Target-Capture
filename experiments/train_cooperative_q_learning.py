"""CLI for seeded shared-policy cooperative Q-learning training."""

import argparse
from pathlib import Path

from configs.evaluation_config import ExperimentConfig
from experiments.run_comparison import train_cooperative_q


def main():
    parser = argparse.ArgumentParser(description="Train Shared-Policy Cooperative Q-Learning")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = ExperimentConfig(
        grid_size=args.grid_size, max_steps=args.max_steps,
        train_episodes=args.episodes, eval_episodes=1,
        seeds=[args.seed], output_dir=args.output_dir,
    )
    agent0, _ = train_cooperative_q(args.seed, config)
    agent0.q_table.save(config.checkpoints_dir / "cooperative_q_learning.pkl")
    print(f"Saved seeded cooperative checkpoint under {config.checkpoints_dir}")


if __name__ == "__main__":
    main()
