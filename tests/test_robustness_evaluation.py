"""Regression checks for matched, frozen and genuinely held-out evaluation."""

import json
import math
from dataclasses import replace

import pytest

from agents.q_learning_agent import QLearningAgent
from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.robustness_analysis import aggregate_robustness, validate_rows
from configs.evaluation_config import ExperimentConfig
from configs.robustness_config import RobustnessConfig, stress_cases
from env.actions import Action
from env.position import Position
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import q_table_snapshot
from experiments.reproducibility import publish_evidence, sha256
from experiments.run_comparison import run_comparison
from experiments.run_robustness_evaluation import (
    configuration_key, evaluate_robustness_episode, generate_unseen_initial_states,
    get_q_coverage, reset_configuration_catalog, run_robustness, validate_initial_state,
    validate_seed_ranges,
)
from experiments.statistical_analysis import read_raw_results


@pytest.mark.parametrize("base", [1_000_002, 10_000_001, 30_000_005, 51_000_001])
def test_ranges_reject_actual_episode_overlap(tmp_path, base):
    source = ExperimentConfig(seeds=(7, 9), train_episodes=5, eval_episodes=3)
    protocol = RobustnessConfig(tmp_path, eval_episodes=3, initial_state_cases=3, held_out_seed_base=base)
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_ranges(source, protocol)


def test_held_out_configurations_exclude_training_starts(tmp_path):
    source = ExperimentConfig(grid_size=4, seeds=(0, 1), train_episodes=20, eval_episodes=4)
    training, evaluation = reset_configuration_catalog(source)
    env = TargetCaptureEnv(4)
    assert configuration_key(env.reset(1_000_019)) in training
    assert configuration_key(env.reset(10_000_007)) in evaluation
    first = generate_unseen_initial_states(1, 42, 4, set())[0]
    first_key = tuple((first[name].x, first[name].y) for name in ("agent_0_pos", "agent_1_pos", "target_pos"))
    excluded = training | evaluation | {first_key}
    cases = generate_unseen_initial_states(20, 42, 4, excluded)
    assert cases == generate_unseen_initial_states(20, 42, 4, excluded)
    keys = set()
    for case in cases:
        validate_initial_state(case, 4)
        key = tuple((case[name].x, case[name].y) for name in ("agent_0_pos", "agent_1_pos", "target_pos"))
        assert key not in excluded and key not in keys
        keys.add(key)


@pytest.mark.parametrize("grid_size", [4, 5, 10])
def test_stress_geometry(grid_size):
    cases = {case.name: case for case in stress_cases(grid_size)}
    for case in cases.values():
        validate_initial_state(dict(zip(("agent_0_pos", "agent_1_pos", "target_pos"),
                                        map(lambda p: Position(*p), (case.agent_0, case.agent_1, case.target)))), grid_size)
    same = cases["same_side_hunters"]
    assert same.agent_0[0] == same.agent_1[0] < same.target[0]
    opposite = cases["opposite_side_hunters"]
    assert opposite.agent_0[0] < opposite.target[0] < opposite.agent_1[0]
    assert cases["boundary_target"].target[0] == 0
    assert cases["widely_separated_hunters"].agent_1 == (grid_size - 1, grid_size - 1)


@pytest.mark.parametrize("shared", [False, True])
def test_seeded_rollout_frozen_and_repeatable(shared):
    def agents():
        if shared:
            table = SharedQTable()
            return (SharedQAgent(table, "agent_0", "agent_1", seed=1),
                    SharedQAgent(table, "agent_1", "agent_0", seed=2))
        return QLearningAgent(seed=1), QLearningAgent(seed=2)
    pair = agents()
    before = [q_table_snapshot(a) for a in pair]
    initial = {"agent_0_pos": Position(0, 0), "agent_1_pos": Position(3, 3), "target_pos": Position(1, 1)}
    row = evaluate_robustness_episode(TargetCaptureEnv(4, 8), *pair, initial, seed=55)
    assert row == evaluate_robustness_episode(TargetCaptureEnv(4, 8), *agents(), initial, seed=55)
    assert [q_table_snapshot(a) for a in pair] == before
    assert row["seen_state_queries"] == 0
    assert row["state_queries"] == 2 * row["episode_length"]
    assert row["unseen_state_rate"] == 1
    assert all(a.epsilon == 0 for a in pair)
    with pytest.raises(ValueError, match="explicit environment seed"):
        evaluate_robustness_episode(TargetCaptureEnv(4), *pair, initial)


def test_membership_uses_frozen_checkpoint_keys():
    agent = QLearningAgent()
    obs = {"agent_position": Position(0, 0), "target_position": Position(1, 1)}
    keys = frozenset(agent.q_table)
    agent.q_table[(0, 0, 1, 1)] = {action: 0 for action in Action}
    assert get_q_coverage(agent, obs) == 1
    assert get_q_coverage(agent, obs, keys) == 0


def test_aggregation_uses_query_denominator_and_missing_success_time():
    rows = [dict(method="example", condition="standard", training_seed=0, captured=0,
                 episode_length=length, episode_reward=0, capture_time=None,
                 state_queries=2 * length, seen_state_queries=seen) for length, seen in ((1, 2), (9, 0))]
    seeds, summaries = aggregate_robustness(rows)
    assert seeds[0]["q_coverage"] == .1  # episode-mean coverage would incorrectly be .5
    assert seeds[0]["unseen_state_rate"] == .9
    assert math.isnan(seeds[0]["capture_time"])
    assert summaries[0]["capture_time_valid_seeds"] == 0
    assert math.isnan(summaries[0]["capture_rate_std"])


def test_deltas_are_paired_seed_estimates():
    rows = []
    for seed in (0, 1):
        for condition in ("standard", "held_out_seeds"):
            captured = int((seed == 0) == (condition == "standard"))
            rows.append(dict(method="example", condition=condition, training_seed=seed, captured=captured,
                             episode_length=1, episode_reward=0, capture_time=1 if captured else None,
                             state_queries=2, seen_state_queries=2))
    _, summaries = aggregate_robustness(rows)
    held = next(r for r in summaries if r["condition"] == "held_out_seeds")
    assert held["delta_capture_rate_mean"] == 0
    assert held["delta_capture_rate_std"] == pytest.approx(math.sqrt(2))


@pytest.fixture(scope="module")
def completed_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("robustness")
    source = ExperimentConfig(grid_size=4, max_steps=6, train_episodes=8, eval_episodes=3,
                              seeds=(0, 1), output_dir=root / "comparison")
    run_comparison(source)
    protocol = RobustnessConfig(source.output_dir, root / "first", eval_episodes=3,
                                initial_state_cases=3, stress_repeats=2, verify_against=source.output_dir)
    run_robustness(protocol)
    return root, source, protocol


def test_complete_pipeline_is_reproducible_and_protects_evidence(completed_study):
    root, source, protocol = completed_study
    run_robustness(replace(protocol, output_dir=root / "second"))
    for relative in ("raw/robustness_raw.csv", "summaries/robustness_summary.csv",
                     "summaries/held_out_initial_configurations.csv", "summaries/checkpoint_identities.csv"):
        assert sha256(protocol.output_dir / relative) == sha256(root / "second" / relative)
    verification = json.loads((protocol.output_dir / "summaries/verification.json").read_text())
    assert verification["episode_rows"] == 2 * 2 * (3 * 3 + 5 * 2)
    assert verification["standard_rows_match_original"] == 12
    assert verification["checkpoint_replica_hashes_match"]
    assert all(check["unchanged"] for check in verification["frozen_table_checks"])
    with pytest.raises(FileExistsError):
        run_robustness(protocol)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "queries", "matched", "capture_time"])
def test_analysis_rejects_corrupted_data(completed_study, corruption):
    _, _, protocol = completed_study
    saved = json.loads((protocol.output_dir / "summaries/robustness_config.json").read_text())
    rows = read_raw_results(protocol.output_dir / "raw/robustness_raw.csv")
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0].copy())
    elif corruption == "queries":
        rows[0]["state_queries"] = "1"
    elif corruption == "matched":
        rows[0]["evaluation_seed"] = "0"
    else:
        rows[0]["capture_time"] = "-1"
    with pytest.raises(ValueError):
        validate_rows(rows, saved)


def test_evidence_publishing_does_not_delete_existing_directory(tmp_path):
    destination = tmp_path / "evidence"
    destination.mkdir()
    original = destination / "keep.txt"
    original.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        publish_evidence(tmp_path / "source", destination)
    assert original.read_text() == "existing evidence"
