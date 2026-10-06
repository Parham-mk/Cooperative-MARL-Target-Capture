"""Evaluate original comparison checkpoints on reproducible matched conditions."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from analysis.trajectory_analysis import TrajectoryRecorder
from configs.evaluation_config import ExperimentConfig
from configs.robustness_config import METHODS, RobustnessConfig, stress_cases
from env.position import Position
from env.rewards import RewardCalculator
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import make_observations, q_table_snapshot
from experiments.generate_behavioral_examples import load_cooperative_q, load_independent_q
from experiments.reproducibility import create_manifest, sha256, validate_inventory, validate_summary
from experiments.statistical_analysis import _write_rows, read_raw_results
from visualization.renderer import GridWorldRenderer


def configuration_key(state):
    """Ordered absolute positions; hunter identities are retained."""
    return tuple((state[name].x, state[name].y) for name in ("agent_0", "agent_1", "target"))


def validate_initial_state(initial_state, grid_size):
    positions = tuple(initial_state[name] for name in ("agent_0_pos", "agent_1_pos", "target_pos"))
    if len(set(positions)) != 3 or any(not (0 <= p.x < grid_size and 0 <= p.y < grid_size) for p in positions):
        raise ValueError("initial positions must be distinct and inside the grid")


def generate_unseen_initial_states(num_cases, seed, grid_size, excluded_configurations):
    """Sample unique starts excluded from an explicit reset-configuration catalog."""
    excluded = set(excluded_configurations)
    capacity = grid_size ** 2
    if num_cases < 1 or num_cases > capacity * (capacity - 1) * (capacity - 2) - len(excluded):
        raise ValueError("requested initial-state catalog cannot be sampled")
    rng = random.Random(seed)
    cases = []
    while len(cases) < num_cases:
        key = tuple(divmod(index, grid_size) for index in rng.sample(range(capacity), 3))
        if key not in excluded:
            excluded.add(key)
            cases.append(dict(zip(("agent_0_pos", "agent_1_pos", "target_pos"), (Position(*p) for p in key))))
    return cases


def get_q_coverage(agent, observation, frozen_keys=None):
    """Membership of an encoded action-query state, without table insertion."""
    table = getattr(agent.q_table, "q_table", agent.q_table)
    return int(agent._get_state_key(observation) in (table if frozen_keys is None else frozen_keys))


def validate_seed_ranges(source, protocol):
    """Check actual episode ranges, including disjoint robustness conditions."""
    count = len(source.seeds)
    if len(set(source.seeds)) != count or any(seed < 0 for seed in source.seeds):
        raise ValueError("source training seeds must be unique and nonnegative")
    if source.train_episodes > source.training_seed_stride:
        raise ValueError("source training environment ranges overlap")
    prior = [(index * source.training_seed_stride, index * source.training_seed_stride + source.train_episodes - 1)
             for index in range(count)]
    prior += [(source.evaluation_seed_base, source.evaluation_seed_base + count * source.eval_episodes - 1)]
    prior += list(protocol.prior_evaluation_ranges)
    fresh = [
        (protocol.held_out_seed_base, protocol.held_out_seed_base + count * protocol.eval_episodes - 1),
        (protocol.initial_state_seed_base, protocol.initial_state_seed_base + count * protocol.initial_state_cases - 1),
        (protocol.stress_seed_base, protocol.stress_seed_base + 5 * count * protocol.stress_repeats - 1),
    ]
    for index, (first, last) in enumerate(fresh):
        if any(first <= other_last and other_first <= last for other_first, other_last in prior + fresh[:index]):
            raise ValueError("held-out episode environment seed ranges overlap training or prior evaluation")
    return {"training_and_prior_evaluation": prior, "fresh_environment_ranges": fresh}


def reset_configuration_catalog(source):
    """Reconstruct starts from original training and evaluation reset schedules."""
    env = TargetCaptureEnv(source.grid_size, source.max_steps)
    training, evaluation = set(), set()
    for index, seed in enumerate(source.seeds):
        for episode in range(source.train_episodes):
            training.add(configuration_key(env.reset(index * source.training_seed_stride + episode)))
        for episode in range(source.eval_episodes):
            evaluation.add(configuration_key(env.reset(source.evaluation_seed(seed, episode))))
    return training, evaluation


def _forbid_update(*args, **kwargs):
    raise RuntimeError("learning is forbidden in robustness evaluation")


def evaluate_robustness_episode(env, agent0, agent1, initial_state_kwargs=None, seed=None,
                                frozen_keys=None, recorder=None):
    """Seeded rollout; reward once per transition, coverage per hunter action query."""
    if seed is None:
        raise ValueError("every robustness rollout requires an explicit environment seed")
    agent0.epsilon = agent1.epsilon = 0.0
    state = env.reset(seed=seed)
    if initial_state_kwargs is not None:
        validate_initial_state(initial_state_kwargs, env.grid_size)
        env.agent_0.position = initial_state_kwargs["agent_0_pos"]
        env.agent_1.position = initial_state_kwargs["agent_1_pos"]
        env.target.position = initial_state_kwargs["target_pos"]
        state = env.get_state()
    start = configuration_key(state)
    if recorder is not None:
        recorder.record_step(0, state, {"agent_0": "START", "agent_1": "START"}, {"target_action": "START"}, 0.0)
    reward_calc = RewardCalculator()
    seen, queries, total_reward = 0, 0, 0.0
    while True:
        observations = make_observations(state)
        for index, (agent, obs) in enumerate(zip((agent0, agent1), observations)):
            seen += get_q_coverage(agent, obs, None if frozen_keys is None else frozen_keys[index])
            queries += 1
        actions = {"agent_0": agent0.select_action(observations[0]), "agent_1": agent1.select_action(observations[1])}
        next_state, info = env.step(actions)
        reward = reward_calc.calculate([env.agent_0, env.agent_1], env.target, state, info["captured"])["total_reward"]
        total_reward += reward
        if recorder is not None:
            recorder.record_step(env.current_step, next_state, actions, info, reward)
        state = next_state
        if info["terminated"] or info["truncated"]:
            break
    distances = [abs(p[0] - start[2][0]) + abs(p[1] - start[2][1]) for p in start[:2]]
    return {
        "evaluation_seed": seed, "captured": int(info["captured"]), "episode_length": env.current_step,
        "capture_time": env.current_step if info["captured"] else None, "episode_reward": total_reward,
        "seen_state_queries": seen, "state_queries": queries, "q_state_coverage": seen / queries,
        "unseen_state_rate": 1 - seen / queries,
        **{f"initial_{name}_{axis}": p[index] for name, p in zip(("agent_0", "agent_1", "target"), start)
           for index, axis in enumerate(("x", "y"))},
        "initial_target_distance_sum": sum(distances),
        "initial_hunter_separation": abs(start[0][0] - start[1][0]) + abs(start[0][1] - start[1][1]),
        "initial_target_boundary_distance": min(*start[2], env.grid_size - 1 - start[2][0], env.grid_size - 1 - start[2][1]),
    }


def _source_config(directory):
    saved = json.loads((directory / "summaries" / "experiment_config.json").read_text(encoding="utf-8"))
    saved["output_dir"] = directory
    return ExperimentConfig(**saved)


def _checkpoint_paths(source, seed):
    return (source.checkpoints_dir / "independent_q" / f"seed_{seed}_agent0.pkl",
            source.checkpoints_dir / "independent_q" / f"seed_{seed}_agent1.pkl",
            source.checkpoints_dir / "cooperative_q" / f"seed_{seed}.pkl")


def run_robustness(protocol):
    """Validate provenance, evaluate all conditions, and save auditable evidence."""
    from analysis.robustness_analysis import analyze_robustness

    source = _source_config(protocol.comparison_dir)
    if protocol.eval_episodes > source.eval_episodes:
        raise ValueError("standard evaluation budget exceeds the source comparison")
    geometries = stress_cases(source.grid_size)
    ranges = validate_seed_ranges(source, protocol)
    validate_inventory(source.output_dir, source.seeds, source.train_episodes, source.eval_episodes)
    validate_summary(source.output_dir)
    if protocol.verify_against is not None:
        other = _source_config(protocol.verify_against)
        if {k: v for k, v in source.to_dict().items() if k != "output_dir"} != {
            k: v for k, v in other.to_dict().items() if k != "output_dir"
        }:
            raise ValueError("checkpoint comparison protocols differ")
    marker = protocol.output_dir / "summaries" / "robustness_config.json"
    if marker.exists() or (protocol.output_dir / "raw" / "robustness_raw.csv").exists():
        raise FileExistsError("robustness study already exists; use a fresh output directory")
    identities = []
    for seed in source.seeds:
        for path in _checkpoint_paths(source, seed):
            relative = path.relative_to(source.output_dir)
            digest = sha256(path)
            if protocol.verify_against is not None and digest != sha256(protocol.verify_against / relative):
                raise ValueError(f"checkpoint provenance mismatch: {relative}")
            identities.append({"training_seed": seed, "checkpoint": relative.as_posix(), "sha256": digest})
    training, previous_eval = reset_configuration_catalog(source)
    excluded = training | previous_eval
    catalog_hash = hashlib.sha256(json.dumps(sorted(training), separators=(",", ":")).encode()).hexdigest()
    saved_protocol = asdict(protocol)
    for key in ("comparison_dir", "output_dir", "verify_against"):
        if saved_protocol[key] is not None:
            saved_protocol[key] = str(saved_protocol[key])
    record = {
        "protocol": saved_protocol, "source_comparison": source.to_dict(), "seed_ranges": ranges,
        "stress_cases": [asdict(case) for case in geometries],
        "reset_catalog": {"unique_training_starts": len(training), "training_starts_sha256": catalog_hash,
                          "unique_prior_evaluation_starts": len(previous_eval),
                          "held_out_initial_exclusion": "all source training and prior main evaluation starts"},
        "coverage": "sum seen hunter action queries / sum all hunter action queries within training seed; fixed checkpoint keys; revisits counted",
        "evaluation": "epsilon=0; existing seeded ties and zero-valued unseen states; no updates or insertion",
        "action_rng_schedule": "standard: source action base + 2*seed_index; other conditions: 60000000 + 1000000*condition_index + 2*seed_index (base configurable); hunter 1 adds 1; stream continuous within condition",
        "grid_size_transfer": {"included": False, "reason": "same-grid reset and spatial robustness only; different grid sizes change encoded state support and boundary aliasing"},
        "diagnostic_selection": "lowest nonstandard capture-rate condition per method; first failure by training/evaluation seed, otherwise longest episode with the same tie order",
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(marker.parent / "checkpoint_identities.csv", identities)
    rows, initial_rows, candidates = [], [], []
    frozen_checks, standard_matches = [], 0
    for seed_index, seed in enumerate(source.seeds):
        cases = generate_unseen_initial_states(protocol.initial_state_cases, protocol.initial_generator_seed + seed_index,
                                              source.grid_size, excluded)
        for case in cases:
            excluded.add(tuple((case[name].x, case[name].y) for name in ("agent_0_pos", "agent_1_pos", "target_pos")))
        batches = [
            ("standard", [{"seed": source.evaluation_seed(seed, ep)} for ep in range(protocol.eval_episodes)]),
            ("held_out_seeds", [{"seed": protocol.held_out_seed_base + seed_index * protocol.eval_episodes + ep}
                                for ep in range(protocol.eval_episodes)]),
            ("held_out_initial_states", [{"seed": protocol.initial_state_seed_base + seed_index * protocol.initial_state_cases + ep,
                                           "initial_state_kwargs": case} for ep, case in enumerate(cases)]),
        ]
        for case_index, case in enumerate(geometries):
            initial = {name: Position(*p) for name, p in zip(
                ("agent_0_pos", "agent_1_pos", "target_pos"), (case.agent_0, case.agent_1, case.target))}
            validate_initial_state(initial, source.grid_size)
            batches.append((f"stress:{case.name}", [
                {"seed": protocol.stress_seed_base + (case_index * len(source.seeds) + seed_index) * protocol.stress_repeats + ep,
                 "initial_state_kwargs": initial} for ep in range(protocol.stress_repeats)]))
        for condition_index, (condition, setups) in enumerate(batches):
            paired_starts = None
            for method, loader in zip(METHODS, (load_independent_q, load_cooperative_q)):
                a0, a1, _ = loader(source, seed, seed_index)
                action_base = source.action_seed_base if condition == "standard" else protocol.action_seed_base + condition_index * 1_000_000
                a0.rng.seed(action_base + seed_index * 2)
                a1.rng.seed(action_base + seed_index * 2 + 1)
                before = [q_table_snapshot(a) for a in (a0, a1)]
                keys = [frozenset(table) for table in before]
                for agent in (a0, a1):
                    agent.update = _forbid_update
                    if not isinstance(agent.q_table, dict):
                        agent.q_table.update = _forbid_update
                env = TargetCaptureEnv(source.grid_size, source.max_steps)
                best, starts = None, []
                filename = "independent_q_results.csv" if method == METHODS[0] else "cooperative_q_results.csv"
                original = {int(row["episode"]): row for row in read_raw_results(source.raw_dir / filename)
                            if int(row["training_seed"]) == seed}
                paths = _checkpoint_paths(source, seed)[:2] if method == METHODS[0] else _checkpoint_paths(source, seed)[2:]
                for episode, setup in enumerate(setups):
                    recorder = TrajectoryRecorder()
                    recorder.set_metadata(method, seed, setup["seed"], source.grid_size, source.max_steps,
                                          "|".join(p.relative_to(source.output_dir).as_posix() for p in paths))
                    recorder.metadata.update(condition=condition, episode=episode)
                    row = evaluate_robustness_episode(env, a0, a1, frozen_keys=keys, recorder=recorder, **setup)
                    row.update(method=method, condition=condition, training_seed=seed, episode=episode,
                               action_seed_0=action_base + seed_index * 2, action_seed_1=action_base + seed_index * 2 + 1)
                    rows.append(row)
                    starts.append(tuple(row[f"initial_{name}_{axis}"] for name in ("agent_0", "agent_1", "target") for axis in ("x", "y")))
                    if method == METHODS[0] and condition == "held_out_initial_states":
                        initial_rows.append({"training_seed": seed, "episode": episode, "evaluation_seed": setup["seed"],
                                             **{k: v for k, v in row.items() if k.startswith("initial_")}})
                    if condition == "standard":
                        expected = original[episode]
                        if any(float(row[field]) != float(expected[field]) for field in ("captured", "episode_length", "episode_reward")):
                            raise AssertionError("standard robustness rollout differs from original comparison")
                        standard_matches += 1
                    if best is None or (best.trajectory[-1]["captured"] and
                                        (not row["captured"] or row["episode_length"] > best.trajectory[-1]["step"])):
                        best = recorder
                if paired_starts is None:
                    paired_starts = starts
                elif paired_starts != starts:
                    raise AssertionError("methods did not receive matched initial configurations")
                if before != [q_table_snapshot(a) for a in (a0, a1)]:
                    raise RuntimeError("robustness evaluation mutated a Q-table")
                frozen_checks.append({"method": method, "training_seed": seed, "condition": condition,
                                      "state_counts": [len(table) for table in before], "unchanged": True})
                candidates.append(best)
        print(f"Robustness seed {seed} complete", flush=True)
    _write_rows(protocol.output_dir / "raw" / "robustness_raw.csv", rows)
    _write_rows(marker.parent / "held_out_initial_configurations.csv", initial_rows)
    summaries = analyze_robustness(protocol.output_dir)
    selection = []
    for method in METHODS:
        worst = min((r for r in summaries if r["method"] == method and r["condition"] != "standard"),
                    key=lambda r: (r["capture_rate_mean"], r["condition"]))["condition"]
        eligible = [r for r in candidates if r.metadata["method"] == method and r.metadata["condition"] == worst]
        failures = [r for r in eligible if not r.trajectory[-1]["captured"]]
        chosen = min(failures or eligible, key=lambda r: (
            0 if failures else -r.trajectory[-1]["step"], r.metadata["training_seed"], r.metadata["evaluation_seed"]))
        chosen.metadata["selection_rule"] = record["diagnostic_selection"]
        safe = "independent_q" if method == METHODS[0] else "cooperative_q"
        relative = Path("trajectories") / f"{safe}_diagnostic.json"
        chosen.save(protocol.output_dir / relative)
        GridWorldRenderer(source.grid_size).plot_static_trajectory(
            chosen.trajectory, f"{method}: {worst}", protocol.output_dir / "plots" / f"{safe}_diagnostic.png")
        selection.append({"method": method, "condition": worst, "training_seed": chosen.metadata["training_seed"],
                          "evaluation_seed": chosen.metadata["evaluation_seed"], "captured": int(chosen.trajectory[-1]["captured"]),
                          "episode_length": chosen.trajectory[-1]["step"], "trajectory": relative.as_posix(),
                          "selection_rule": record["diagnostic_selection"]})
    _write_rows(marker.parent / "diagnostic_selection.csv", selection)
    for identity in identities:
        if sha256(source.output_dir / identity["checkpoint"]) != identity["sha256"]:
            raise RuntimeError("checkpoint file changed")
    verification = {"episode_rows": len(rows), "standard_rows_match_original": standard_matches,
                    "matched_initial_conditions": True, "no_learning_calls": True, "frozen_table_checks": frozen_checks,
                    "checkpoint_files_unchanged": True,
                    "held_out_initial_configurations_absent_from_source_training_and_evaluation": True,
                    "checkpoint_replica_hashes_match": protocol.verify_against is not None}
    (marker.parent / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    create_manifest(protocol.output_dir)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True, help="completed run_comparison output with checkpoints and raw data")
    parser.add_argument("--output-dir", type=Path, default=Path("results/robustness"))
    parser.add_argument("--verify-against", type=Path, help="optional second original comparison for checkpoint hash verification")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--initial-state-cases", type=int, default=500)
    parser.add_argument("--stress-repeats", type=int, default=100)
    parser.add_argument("--prior-evaluation-range", nargs=2, type=int, action="append", default=[], metavar=("FIRST", "LAST"),
                        help="additional inclusive prior environment seed ranges; historical Phase 13 audit range is always included")
    args = vars(parser.parse_args())
    extra = args.pop("prior_evaluation_range")
    args["prior_evaluation_ranges"] = RobustnessConfig.__dataclass_fields__["prior_evaluation_ranges"].default + tuple(map(tuple, extra))
    run_robustness(RobustnessConfig(**args))


if __name__ == "__main__":
    main()
