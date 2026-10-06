"""Small end-to-end demonstration composed from the existing study runners."""

import argparse
import json
from pathlib import Path

from configs.ablation_config import AblationConfig
from configs.evaluation_config import ExperimentConfig
from configs.robustness_config import RobustnessConfig
from experiments.generate_behavioral_examples import generate_behavioral_evidence
from experiments.run_comparison import run_comparison
from experiments.run_reward_ablation import run_reward_ablation
from experiments.run_robustness_evaluation import run_robustness


def reproduce_demo(output_dir):
    """Generate checkpoints, raw data, behavior, ablation, robustness and figures."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("demo output already exists; choose a fresh directory")
    config = ExperimentConfig(grid_size=4, max_steps=30, train_episodes=50,
                              eval_episodes=25, seeds=(0, 1), output_dir=root / "main")
    run_comparison(config)
    generate_behavioral_evidence(config)
    run_reward_ablation(AblationConfig(grid_size=4, max_steps=30, train_episodes=50,
                                      eval_episodes=25, seeds=(0, 1), rolling_window=10,
                                      output_dir=root / "ablations"))
    run_robustness(RobustnessConfig(comparison_dir=config.output_dir,
                                   output_dir=root / "robustness", eval_episodes=25,
                                   initial_state_cases=25, stress_repeats=5))
    record = {"purpose": "functional demonstration; these budgets are not the scientific experiment",
              "main": config.to_dict(), "ablation_rolling_window": 10,
              "robustness": {"episodes_per_primary_condition_and_seed": 25, "stress_repeats": 5}}
    (root / "demo_config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"), help="fresh destination")
    args = parser.parse_args()
    reproduce_demo(args.output_dir)
    print(f"Demo complete: {args.output_dir}; see docs/reproducibility.md for scientific budgets")


if __name__ == "__main__":
    main()
