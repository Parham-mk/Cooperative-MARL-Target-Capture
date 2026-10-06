"""Train the four Phase 13 reward conditions independently and record frozen rollouts."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.behavioral_metrics import calculate_all_behavioral_metrics
from analysis.reward_ablation_analysis import analyze_ablation
from configs.ablation_config import AblationConfig, REWARD_VARIANTS, VARIANT_ORDER
from experiments.generate_behavioral_examples import (
    evaluate_with_recorder, find_representative_episodes, SUCCESS_RULE, FAILURE_RULE,
)
from experiments.reproducibility import create_manifest, sha256
from experiments.run_comparison import save_rows, train_cooperative_q
from visualization.renderer import GridWorldRenderer


def load_ablation_agents(config, variant, seed, seed_index):
    """Load into a new table and new wrappers with fresh evaluation RNGs."""
    table = SharedQTable(config.learning_rate, config.gamma)
    table.load(config.checkpoint_path(variant, seed))
    action_seed = config.action_seed_base + seed_index * 2
    return (
        SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=action_seed),
        SharedQAgent(table, "agent_1", "agent_0", epsilon=0.0, seed=action_seed + 1),
    )


def recorder_result(recorder, variant):
    final = recorder.trajectory[-1]
    captured = int(final["captured"])
    # Match the training/evaluate_policy left-to-right accumulation exactly.
    episode_reward = 0.0
    for frame in recorder.trajectory[1:]:
        episode_reward += frame["reward"]
    return {
        "variant": variant, "training_seed": recorder.metadata["training_seed"],
        "evaluation_seed": recorder.metadata["evaluation_seed"], "episode": recorder.metadata["episode"],
        "captured": captured, "episode_length": final["step"],
        "capture_time": final["step"] if captured else None,
        # Initial frame has zero reward: one team reward per real transition.
        "episode_reward": episode_reward,
    }


def save_representatives(config, variant, recorders):
    success, failure, median = find_representative_episodes(recorders)
    selection = []
    renderer = GridWorldRenderer(config.grid_size)
    for outcome, recorder, rule in (("success", success, SUCCESS_RULE), ("failure", failure, FAILURE_RULE)):
        relative = f"trajectories/{variant}_{outcome}.json"
        if recorder is not None:
            recorder.metadata.update(selection_rule=rule, variant_median_success_time=median)
            recorder.save(config.output_dir / relative)
            renderer.plot_static_trajectory(
                recorder.trajectory, f"{variant}: selected {outcome}",
                config.behavioral_dir / f"{variant}_{outcome}_trajectory.png",
            )
        selection.append({
            "variant": variant, "outcome": outcome,
            "training_seed": recorder.metadata["training_seed"] if recorder else None,
            "evaluation_seed": recorder.metadata["evaluation_seed"] if recorder else None,
            "episode_length": recorder.trajectory[-1]["step"] if recorder else None,
            "variant_median_success_time": median,
            "selection_rule": rule if recorder else f"no {outcome} observed in complete evaluation dataset",
            "trajectory": relative if recorder else "",
        })
    return selection


def run_reward_ablation(config: AblationConfig, executed_command=None):
    """Never warm-start. Completed seed artifacts are saved before proceeding."""
    metadata_path = config.summaries_dir / "ablation_config.json"
    if metadata_path.exists():
        raise FileExistsError(f"study already exists in {config.output_dir}; choose a fresh output directory or rerun analysis")
    for folder in (config.raw_dir, config.training_dir, config.summaries_dir, config.plots_dir,
                   config.checkpoints_dir, config.behavioral_dir, config.trajectories_dir):
        folder.mkdir(parents=True, exist_ok=True)
    metadata = config.to_dict()
    metadata["executed_command"] = list(executed_command or [])
    metadata["runtime"] = {"python": platform.python_version(), "platform": platform.platform()}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    identities, selections = [], []
    status_path = config.summaries_dir / "study_status.json"
    def save_status(complete=False):
        status_path.write_text(json.dumps({"complete": complete, "completed_conditions": identities},
                                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_status()
    for variant in VARIANT_ORDER:
        reward_config = REWARD_VARIANTS[variant]
        results, metrics, variant_recorders = [], [], []
        for index, seed in enumerate(config.seeds):
            print(f"Training {variant}, seed {seed}: {config.train_episodes} episodes", flush=True)
            checkpoint = config.checkpoint_path(variant, seed)
            agent0, agent1, history = train_cooperative_q(
                seed, config, return_history=True, reward_config=reward_config,
                checkpoint_path=checkpoint, history_path=config.history_path(variant, seed),
                learning_rate=config.learning_rate, gamma=config.gamma, epsilon=config.epsilon,
                epsilon_decay=config.epsilon_decay, min_epsilon=config.min_epsilon,
                rolling_window=config.rolling_window,
            )
            # Route the existing loop's history, without duplicating its training logic.
            for row in history:
                row["variant"] = variant
            save_rows(history, config.history_path(variant, seed))
            identity = {"variant": variant, "training_seed": seed,
                        "checkpoint": checkpoint.relative_to(config.output_dir).as_posix(),
                        "sha256": sha256(checkpoint), "state_count": len(agent0.q_table.q_table),
                        "training_history": config.history_path(variant, seed).relative_to(config.output_dir).as_posix()}
            del agent0, agent1, history
            agents = load_ablation_agents(config, variant, seed, index)
            print(f"Evaluating {variant}, seed {seed}: {config.eval_episodes} frozen episodes", flush=True)
            recorders = evaluate_with_recorder(
                *agents, variant, seed, config.eval_episodes, config.grid_size, config.max_steps,
                checkpoint=identity["checkpoint"], evaluation_seed_base=config.evaluation_seed_base,
                seed_index=index, reward_config=reward_config,
            )
            for recorder in recorders:
                recorder.metadata.update(variant=variant, reward_config=asdict(reward_config),
                                         checkpoint_sha256=identity["sha256"],
                                         action_rng_initial_seeds=[config.action_seed_base + index * 2,
                                                                   config.action_seed_base + index * 2 + 1])
                row = recorder_result(recorder, variant)
                results.append(row)
                metrics.append({**row, **calculate_all_behavioral_metrics(recorder.trajectory)})
            variant_recorders.extend(recorders)
            save_rows(results, config.raw_dir / f"{variant}.csv")
            save_rows(metrics, config.behavioral_dir / f"{variant}_metrics.csv")
            identities.append(identity)
            save_status()
            save_rows(identities, config.summaries_dir / "checkpoint_identities.csv")
            del agents, recorders
        selections.extend(save_representatives(config, variant, variant_recorders))
        save_rows(selections, config.behavioral_dir / "representative_selection.csv")
        del variant_recorders
    summary = analyze_ablation(config)
    save_status(complete=True)
    create_manifest(config.output_dir)
    print(f"Completed study: {config.summaries_dir / 'reward_ablation_summary.csv'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablations"))
    args = parser.parse_args()
    run_reward_ablation(AblationConfig(**vars(args)), [sys.executable, "-m", "experiments.run_reward_ablation", *sys.argv[1:]])


if __name__ == "__main__":
    main()
