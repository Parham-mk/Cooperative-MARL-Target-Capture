import copy
import csv
import hashlib
import math
from pathlib import Path

import pytest
from PIL import Image

from agents.q_learning_agent import QLearningAgent
from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.trajectory_analysis import TrajectoryRecorder
from configs.evaluation_config import ExperimentConfig
from env.actions import Action
from env.position import Position
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import evaluate_policy
from experiments.generate_behavioral_examples import (
    FAILURE_RULE,
    SUCCESS_RULE,
    aggregate_behavioral_rows,
    evaluate_with_recorder,
    find_representative_episodes,
    generate_behavioral_evidence,
)
from experiments.reproducibility import (
    compare_reproductions,
    create_manifest,
    validate_behavioral,
    validate_inventory,
    validate_summary,
)
from experiments.run_comparison import run_comparison
from experiments.statistical_analysis import aggregate_rows


def observation(agent=(1, 1), target=(2, 2)):
    return {
        "agent_position": Position(*agent),
        "target_position": Position(*target),
    }


def test_seeded_independent_action_sequences_and_read_only_unseen_state():
    first = QLearningAgent(epsilon=0.0, seed=42)
    second = QLearningAgent(epsilon=0.0, seed=42)
    obs = observation()
    assert [first.select_action(obs) for _ in range(30)] == [
        second.select_action(obs) for _ in range(30)
    ]
    assert first.q_table == second.q_table == {}


def test_independent_terminal_and_truncation_bootstrap():
    obs = observation()
    nxt = observation((1, 2), (2, 2))
    next_key = (1, 2, 2, 2)
    terminal = QLearningAgent(learning_rate=1.0, gamma=0.9)
    terminal.q_table[next_key] = {action: 10.0 for action in Action}
    terminal.update(obs, Action.UP, 2.0, nxt, terminated=True)
    assert terminal.q_table[(1, 1, 2, 2)][Action.UP] == 2.0

    truncated = QLearningAgent(learning_rate=1.0, gamma=0.9)
    truncated.q_table[next_key] = {action: 10.0 for action in Action}
    truncated.update(obs, Action.UP, 2.0, nxt, terminated=False)
    assert truncated.q_table[(1, 1, 2, 2)][Action.UP] == 11.0


def test_shared_greedy_read_does_not_insert_unseen_state():
    table = SharedQTable()
    agent = SharedQAgent(table, "agent_0", "agent_1", epsilon=0.0, seed=7)
    state = {
        "agent_0": Position(0, 0),
        "agent_1": Position(1, 0),
        "target": Position(2, 2),
    }
    assert agent.select_action(state) in Action
    assert table.q_table == {}


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"grid_size": 1}, ValueError),
        ({"max_steps": 0}, ValueError),
        ({"grid_size": 2.5}, TypeError),
        ({"max_steps": 1.5}, TypeError),
    ],
)
def test_environment_rejects_unusable_configuration(kwargs, exception):
    with pytest.raises(exception):
        TargetCaptureEnv(**kwargs)


def test_environment_reports_real_target_action():
    env = TargetCaptureEnv(grid_size=4, max_steps=2)
    env.reset(seed=31)
    _, info = env.step({"agent_0": Action.STAY, "agent_1": Action.STAY})
    assert isinstance(info["target_action"], Action)


def test_learned_policy_evaluation_is_frozen_for_unseen_states():
    agent0 = QLearningAgent(epsilon=0.7, seed=1)
    agent1 = QLearningAgent(epsilon=0.7, seed=2)
    results = evaluate_policy(agent0, agent1, "Independent", 0, 2, 4, 3)
    assert len(results) == 2
    assert agent0.q_table == agent1.q_table == {}
    assert agent0.epsilon == agent1.epsilon == 0.7

    table = SharedQTable()
    shared0 = SharedQAgent(table, "agent_0", "agent_1", epsilon=0.8, seed=1)
    shared1 = SharedQAgent(table, "agent_1", "agent_0", epsilon=0.8, seed=2)
    evaluate_policy(shared0, shared1, "Cooperative", 0, 2, 4, 3)
    assert table.q_table == {}
    assert shared0.epsilon == shared1.epsilon == 0.8


def test_zero_success_capture_time_is_nan_and_seed_count_is_explicit():
    rows = [
        {
            "method": "Never", "training_seed": seed, "evaluation_seed": seed,
            "episode": 0, "captured": 0, "episode_length": 5,
            "episode_reward": -1.0, "capture_time": None,
        }
        for seed in (0, 1)
    ]
    summary, per_seed = aggregate_rows(rows)
    assert all(math.isnan(row["capture_time"]) for row in per_seed)
    assert math.isnan(summary["capture_time_mean"])
    assert summary["capture_time_valid_seeds"] == 0
    assert summary["seed_count"] == 2


def test_behavioral_aggregation_uses_seed_estimates():
    rows = []
    for seed, values in ((0, [0.0, 2.0]), (1, [10.0, 14.0])):
        for episode, value in enumerate(values):
            rows.append({
                "method": "M", "training_seed": seed, "captured": 0,
                "episode_length": 5, "mean_target_distance": value,
                "mean_hunter_separation": 1,
                "simultaneous_adjacency_fraction": 0,
                "distinct_side_fraction": 0,
            })
    seed_rows, summary = aggregate_behavioral_rows(rows)
    assert [row["mean_target_distance"] for row in seed_rows] == [1.0, 12.0]
    assert summary[0]["mean_target_distance_mean"] == 6.5
    assert summary[0]["mean_target_distance_std"] == pytest.approx(11 / math.sqrt(2))


def _recorder(seed, evaluation_seed, length, captured):
    recorder = TrajectoryRecorder()
    recorder.set_metadata("M", seed, evaluation_seed, 4, 10, "checkpoint.pkl")
    recorder.trajectory = [
        {"step": length, "captured": captured}
    ]
    return recorder


def test_representative_selection_is_median_based_and_deterministic():
    recorders = [
        _recorder(1, 12, 8, True),
        _recorder(0, 11, 4, True),
        _recorder(0, 10, 6, True),
        _recorder(1, 9, 10, False),
        _recorder(0, 20, 10, False),
    ]
    success, failure, median = find_representative_episodes(recorders)
    assert median == 6
    assert success.metadata["evaluation_seed"] == 10
    assert failure.metadata["evaluation_seed"] == 20


def test_behavioral_rollout_records_real_actions_and_metadata():
    table = SharedQTable()
    agent0 = SharedQAgent(table, "agent_0", "agent_1", epsilon=0, seed=1)
    agent1 = SharedQAgent(table, "agent_1", "agent_0", epsilon=0, seed=2)
    recorders = evaluate_with_recorder(
        agent0, agent1, "Cooperative", 3, 1, 4, 3,
        "checkpoint.pkl", evaluation_seed_base=100, seed_index=0,
    )
    recorder = recorders[0]
    assert [step["step"] for step in recorder.trajectory] == list(range(len(recorder.trajectory)))
    assert recorder.trajectory[0]["target_action"] == "START"
    assert all(
        step["target_action"] in Action.__members__
        for step in recorder.trajectory[1:]
    )
    assert recorder.metadata["checkpoint"] == "checkpoint.pkl"
    assert table.q_table == {}


@pytest.fixture(scope="module")
def small_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    config_a = ExperimentConfig(
        grid_size=4, max_steps=4, train_episodes=4, eval_episodes=3,
        seeds=[0, 1], output_dir=root / "run_a",
    )
    config_b = ExperimentConfig(
        grid_size=4, max_steps=4, train_episodes=4, eval_episodes=3,
        seeds=[0, 1], output_dir=root / "run_b",
    )
    run_comparison(config_a)
    run_comparison(config_b)
    generate_behavioral_evidence(config_a)
    generate_behavioral_evidence(config_b)
    return config_a, config_b


def test_small_pipeline_inventory_summary_and_reproduction(small_pipeline):
    config_a, config_b = small_pipeline
    validate_inventory(
        config_a.output_dir, config_a.seeds,
        config_a.train_episodes, config_a.eval_episodes,
    )
    validate_summary(config_a.output_dir)
    validate_behavioral(config_a.output_dir, config_a.seeds, config_a.eval_episodes)
    checked = compare_reproductions(config_a.output_dir, config_b.output_dir)
    assert checked
    assert any(row["path"].endswith(".pkl") for row in checked)


def test_small_pipeline_behavioral_outputs_are_readable(small_pipeline):
    config, _ = small_pipeline
    metrics = list(csv.DictReader(
        (config.output_dir / "behavioral" / "behavioral_metrics.csv").open()
    ))
    assert len(metrics) == 4 * len(config.seeds) * config.eval_episodes
    selections = list(csv.DictReader(
        (config.output_dir / "behavioral" / "representative_selection.csv").open()
    ))
    assert any(row["selection_rule"] == SUCCESS_RULE for row in selections)
    for path in (config.output_dir / "gifs").glob("*.gif"):
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            assert image.n_frames >= 2
            assert image.size[0] > 0 and image.size[1] > 0
    for path in (config.output_dir / "behavioral").glob("*_trajectory.png"):
        with Image.open(path) as image:
            assert image.size[0] > 0 and image.size[1] > 0


def test_manifest_has_sizes_and_hashes(small_pipeline):
    config, _ = small_pipeline
    entries = create_manifest(config.output_dir)
    assert entries
    assert all(row["size_bytes"] > 0 and len(row["sha256"]) == 64 for row in entries)
