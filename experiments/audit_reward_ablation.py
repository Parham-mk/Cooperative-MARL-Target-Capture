"""Independent capture/movement audit and fresh-seed evaluation of Phase 13 tables.

No training and no changes to the scientific study. Geometry, movement, target
sampling, reward arithmetic, and termination are checked against separate
reference calculations rather than trusting evaluator flags or capture helpers.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.reward_ablation_analysis import aggregate_variant, load_saved_config
from configs.ablation_config import REWARD_VARIANTS, VARIANT_ORDER
from env.actions import Action
from env.target_capture_env import TargetCaptureEnv
from experiments.reproducibility import create_manifest, sha256
from experiments.run_comparison import save_rows
from experiments.statistical_analysis import read_raw_results


DELTAS = {Action.UP: (0, 1), Action.DOWN: (0, -1), Action.LEFT: (-1, 0),
          Action.RIGHT: (1, 0), Action.STAY: (0, 0)}


def coordinates(state):
    return {name: (position.x, position.y) for name, position in state.items()}


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def geometric_capture(positions):
    return (len(set(positions.values())) == 3
            and distance(positions["agent_0"], positions["target"]) == 1
            and distance(positions["agent_1"], positions["target"]) == 1)


def reference_transition(positions, actions, target_rng, grid_size):
    """Independent sequential collision handling and random-target sampling."""
    expected = dict(positions)
    def proposed(name, action):
        dx, dy = DELTAS[action]
        x, y = expected[name]
        return x + dx, y + dy
    def inside(position):
        return all(0 <= coordinate < grid_size for coordinate in position)
    def move(name, action):
        candidate = proposed(name, action)
        occupied = {position for other, position in expected.items() if other != name}
        if inside(candidate) and candidate not in occupied:
            expected[name] = candidate
    for name in ("agent_0", "agent_1"):
        move(name, actions[name])
    valid = [action for action in Action if inside(proposed("target", action))]
    target_action = target_rng.choice(valid)
    move("target", target_action)
    return expected, target_action


def independent_rollouts(agents, variant, training_seed, episodes, grid_size,
                         max_steps, environment_seed_base, seed_index):
    """Score by geometry; verify reference mechanics and immutable learned table."""
    table = agents[0].q_table
    assert agents[1].q_table is table
    before = deepcopy(table.q_table)
    original_update = table.update
    def reject_update(*args, **kwargs):
        raise AssertionError("learning attempted during audit")
    table.update = reject_update
    env = TargetCaptureEnv(grid_size, max_steps)
    reward = REWARD_VARIANTS.get(variant, REWARD_VARIANTS["full_reward"])
    rows = []
    checks = {"transitions": 0, "hunter_decisions": 0, "seen_state_decisions": 0,
              "initially_adjacent_episodes": 0}
    try:
        for episode in range(episodes):
            seed = environment_seed_base + seed_index * episodes + episode
            state = env.reset(seed=seed)
            previous = coordinates(state)
            target_rng = np.random.default_rng(seed)
            checks["initially_adjacent_episodes"] += int(geometric_capture(previous))
            total = 0.0
            captured = False
            for step in range(1, max_steps + 1):
                for agent in agents:
                    key = agent._get_state_key(state)
                    checks["hunter_decisions"] += 1
                    checks["seen_state_decisions"] += int(key in table.q_table)
                actions = {"agent_0": agents[0].select_action(state), "agent_1": agents[1].select_action(state)}
                expected, target_action = reference_transition(previous, actions, target_rng, grid_size)
                state, info = env.step(actions)
                positions = coordinates(state)
                assert positions == expected, (seed, step, positions, expected)
                assert info["target_action"] == target_action
                assert len(set(positions.values())) == 3
                assert all(0 <= c < grid_size for p in positions.values() for c in p)
                captured = geometric_capture(positions)
                assert info["captured"] == captured and info["terminated"] == captured
                assert info["truncated"] == (step == max_steps) and info["step"] == step
                prev_distance = sum(distance(previous[h], previous["target"]) for h in ("agent_0", "agent_1"))
                next_distance = sum(distance(positions[h], positions["target"]) for h in ("agent_0", "agent_1"))
                total += reward.distance_weight * (prev_distance - next_distance) + reward.capture_weight * captured + reward.step_penalty
                previous = positions
                checks["transitions"] += 1
                if captured:
                    break
            rows.append({"variant": variant, "training_seed": training_seed,
                         "evaluation_seed": seed, "episode": episode, "captured": int(captured),
                         "episode_length": step, "capture_time": step if captured else None,
                         "episode_reward": total})
    finally:
        table.update = original_update
    assert table.q_table == before, "Q-values or membership changed during audit"
    return rows, checks


def make_agents(checkpoint, action_seed):
    table = SharedQTable()
    if checkpoint is not None:
        table.load(checkpoint)
    return (SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=action_seed),
            SharedQAgent(table, "agent_1", "agent_0", epsilon=0.0, seed=action_seed + 1))


def run_audit(study_dir, output_dir, fresh_episodes=2000,
              environment_seed_base=30_000_000, action_seed_base=40_000_000):
    config = load_saved_config(study_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("choose a fresh audit directory")
    output_dir.mkdir(parents=True)
    schedules = config.to_dict()["seed_schedules"]
    training_seeds = {seed for row in schedules for seed in range(row["training_environment_first"], row["training_environment_last"] + 1)}
    original_seeds = {config.evaluation_seed(seed, episode) for seed in config.seeds for episode in range(config.eval_episodes)}
    fresh_seeds = set(range(environment_seed_base, environment_seed_base + len(config.seeds) * fresh_episodes))
    assert not training_seeds & original_seeds
    assert not fresh_seeds & (training_seeds | original_seeds)
    identities = read_raw_results(config.summaries_dir / "checkpoint_identities.csv")
    assert len(identities) == len(VARIANT_ORDER) * len(config.seeds)
    assert len({row["checkpoint"] for row in identities}) == len(identities)
    assert len({row["sha256"] for row in identities}) == len(identities)
    original_hashes = {row["checkpoint"]: sha256(config.output_dir / row["checkpoint"]) for row in identities}
    assert all(original_hashes[row["checkpoint"]] == row["sha256"] for row in identities)
    checks, summaries, per_seed = [], [], []
    for variant in VARIANT_ORDER:
        stored = read_raw_results(config.raw_dir / f"{variant}.csv")
        rerun, fresh = [], []
        for index, seed in enumerate(config.seeds):
            checkpoint = config.checkpoint_path(variant, seed)
            print(f"Auditing {variant}, training seed {seed}", flush=True)
            rows, old_checks = independent_rollouts(
                make_agents(checkpoint, config.action_seed_base + index * 2), variant, seed,
                config.eval_episodes, config.grid_size, config.max_steps, config.evaluation_seed_base, index)
            rerun.extend(rows)
            rows, new_checks = independent_rollouts(
                make_agents(checkpoint, action_seed_base + index * 2), variant, seed,
                fresh_episodes, config.grid_size, config.max_steps, environment_seed_base, index)
            fresh.extend(rows)
            checks.append({"variant": variant, "training_seed": seed, "original": old_checks, "fresh": new_checks})
        for actual, expected in zip(rerun, stored):
            for field in ("training_seed", "evaluation_seed", "episode", "captured", "episode_length"):
                assert actual[field] == int(expected[field]), (variant, field)
            assert actual["episode_reward"] == float(expected["episode_reward"])
            assert actual["capture_time"] == (int(expected["capture_time"]) if expected["capture_time"] else None)
        assert len(rerun) == len(stored)
        save_rows(fresh, output_dir / "raw" / f"{variant}.csv")
        summary, seeds = aggregate_variant(fresh)
        summaries.append(summary)
        per_seed.extend(seeds)
    control = []
    for index, seed in enumerate(config.seeds):
        rows, _ = independent_rollouts(make_agents(None, action_seed_base + index * 2),
            "untrained_shared", seed, config.eval_episodes, config.grid_size, config.max_steps,
            environment_seed_base, index)
        control.extend(rows)
    save_rows(control, output_dir / "raw/untrained_shared.csv")
    control_summary, control_seeds = aggregate_variant(control)
    summaries.append(control_summary)
    per_seed.extend(control_seeds)
    assert all(sha256(config.output_dir / path) == checksum for path, checksum in original_hashes.items())
    save_rows(summaries, output_dir / "fresh_seed_summary.csv")
    save_rows(per_seed, output_dir / "fresh_seed_per_seed.csv")
    report = {"study_dir": str(config.output_dir), "fresh_episodes_per_training_seed": fresh_episodes,
              "environment_seed_base": environment_seed_base, "action_seed_base": action_seed_base,
              "training_original_fresh_environment_seed_overlap": 0,
              "unique_checkpoint_paths_and_hashes": len(identities),
              "original_evaluation_rows_reproduced_by_independent_geometry": len(VARIANT_ORDER) * len(config.seeds) * config.eval_episodes,
              "checkpoints_unchanged": True, "geometry_target_sampling_movement_and_flags_verified": True,
              "learning_calls_forbidden_and_q_tables_unchanged": True, "checks": checks}
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_manifest(output_dir, output_dir / "artifact_manifest.json")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/ablations"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablations_audit"))
    parser.add_argument("--eval-episodes", type=int, default=2000)
    args = parser.parse_args()
    if args.eval_episodes < 1:
        parser.error("eval-episodes must be positive")
    summaries = run_audit(args.study_dir, args.output_dir, args.eval_episodes)
    for row in summaries:
        print(f"{row['variant']}: capture {row['capture_rate_mean']:.6f} ± {row['capture_rate_std']:.6f}, length {row['episode_length_mean']:.4f}")


if __name__ == "__main__":
    main()
