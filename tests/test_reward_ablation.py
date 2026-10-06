"""Fast correctness, experimental-control, and reproducibility contracts for Phase 13."""

import copy
import json
import math
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from agents.shared_q_agent import SharedQAgent
from algorithms.cooperative_q_learning import SharedQTable
from analysis.reward_ablation_analysis import (
    aggregate_variant, analyze_ablation, load_saved_config, threshold_episode,
)
from configs.ablation_config import AblationConfig, REWARD_VARIANTS, VARIANT_ORDER
from configs.evaluation_config import ExperimentConfig
from env.actions import Action
from env.entities import Agent, Target
from env.position import Position
from env.rewards import RewardCalculator, RewardConfig
from env.target_capture_env import TargetCaptureEnv
from experiments.evaluation_utils import evaluate_policy
from experiments.generate_behavioral_examples import evaluate_with_recorder
from experiments.run_comparison import train_cooperative_q
from experiments.run_comparison import save_rows
from experiments.run_reward_ablation import run_reward_ablation, load_ablation_agents, recorder_result
from experiments.statistical_analysis import read_raw_results


def pair(seed=0, epsilon=0.0):
    table = SharedQTable()
    return (SharedQAgent(table, "agent_0", "agent_1", epsilon=epsilon, seed=seed),
            SharedQAgent(table, "agent_1", "agent_0", epsilon=epsilon, seed=seed + 1))


def test_exact_variants_and_immutable_isolation():
    assert [tuple(asdict(REWARD_VARIANTS[v]).values()) for v in VARIANT_ORDER] == [
        (1.0, 20.0, -0.05), (0.0, 20.0, -0.05), (1.0, 20.0, 0.0), (1.0, 0.0, -0.05),
    ]
    with pytest.raises(FrozenInstanceError):
        REWARD_VARIANTS["full_reward"].capture_weight = 0
    with pytest.raises(TypeError):
        REWARD_VARIANTS["new"] = RewardConfig()
    a = REWARD_VARIANTS["full_reward"].calculator()
    a.distance_weight = 8
    assert REWARD_VARIANTS["full_reward"].calculator().distance_weight == 1
    assert replace(REWARD_VARIANTS["full_reward"], distance_weight=8) != REWARD_VARIANTS["full_reward"]


@pytest.mark.parametrize("captured", [False, True])
def test_known_moving_target_transition_and_zero_components(captured):
    previous = {"agent_0": Position(0, 0), "agent_1": Position(4, 0), "target": Position(2, 2)}
    agents = [Agent("agent_0", Position(1, 1)), Agent("agent_1", Position(3, 1))]
    target = Target(Position(2, 1))
    # Previous team distance = 8, current = 2, including target movement.
    values = {name: config.calculator().calculate(agents, target, previous, captured)
              for name, config in REWARD_VARIANTS.items()}
    assert values["full_reward"]["total_reward"] == pytest.approx(6 + 20 * captured - .05)
    assert values["no_distance"]["distance_reward"] == 0
    assert values["no_step_penalty"]["step_penalty"] == 0
    assert values["no_capture_reward"]["capture_reward"] == 0
    assert values["full_reward"]["total_reward"] - values["no_distance"]["total_reward"] == pytest.approx(6)
    assert values["full_reward"]["total_reward"] - values["no_step_penalty"]["total_reward"] == pytest.approx(-.05)
    assert values["full_reward"]["total_reward"] - values["no_capture_reward"]["total_reward"] == pytest.approx(20 * captured)


def test_capture_termination_with_zero_capture_weight():
    env = TargetCaptureEnv(4, 10)
    env.reset(seed=0)
    env.agent_0.position = Position(1, 1)
    env.agent_1.position = Position(3, 1)
    env.target.position = Position(2, 1)
    env.target_policy.choose_action = lambda *args: Action.STAY
    previous = env.get_state()
    _, info = env.step({"agent_0": Action.STAY, "agent_1": Action.STAY})
    assert info["captured"] and info["terminated"] and not info["truncated"]
    reward = REWARD_VARIANTS["no_capture_reward"].calculator().calculate(
        [env.agent_0, env.agent_1], env.target, previous, info["captured"])
    assert reward["capture_reward"] == 0 and reward["total_reward"] == -.05


class KnownEnvironment:
    """Two real position transitions, second capturing, for accounting tests."""
    def __init__(self, *args, **kwargs):
        pass

    def reset(self, seed):
        self.current_step = 0
        self.agent_0 = Agent("agent_0", Position(0, 0))
        self.agent_1 = Agent("agent_1", Position(4, 0))
        self.target = Target(Position(2, 2))
        return self.state()

    def state(self):
        return {"agent_0": self.agent_0.position, "agent_1": self.agent_1.position, "target": self.target.position}

    def step(self, actions):
        self.current_step += 1
        self.agent_0.position = Position(1, self.current_step - 1)
        self.agent_1.position = Position(3, self.current_step - 1)
        self.target.position = Position(2, 1)
        captured = self.current_step == 2
        return self.state(), {"step": self.current_step, "captured": captured, "terminated": captured,
                              "truncated": False, "target_action": Action.STAY}


@pytest.mark.parametrize("variant", VARIANT_ORDER)
def test_reward_reaches_all_helpers_and_once_per_transition(monkeypatch, tmp_path, variant):
    for module in ("experiments.run_comparison", "experiments.evaluation_utils", "experiments.generate_behavioral_examples"):
        monkeypatch.setattr(f"{module}.TargetCaptureEnv", KnownEnvironment)
    updates, calculations = [], []
    original_update, original_calculate = SharedQAgent.update, RewardCalculator.calculate
    def update(agent, obs, action, reward, next_obs, done):
        updates.append((agent.agent_id, reward, done))
        return original_update(agent, obs, action, reward, next_obs, done)
    def calculate(calculator, *args, **kwargs):
        result = original_calculate(calculator, *args, **kwargs)
        calculations.append(result)
        return result
    monkeypatch.setattr(SharedQAgent, "update", update)
    monkeypatch.setattr(RewardCalculator, "calculate", calculate)
    config = AblationConfig(train_episodes=1, eval_episodes=1, seeds=[0], output_dir=tmp_path)
    reward_config = REWARD_VARIANTS[variant]
    a0, a1, history = train_cooperative_q(0, config, True, reward_config=reward_config)
    first = 4 * reward_config.distance_weight + reward_config.step_penalty
    second = 2 * reward_config.distance_weight + reward_config.capture_weight + reward_config.step_penalty
    assert len(calculations) == 2
    assert updates == [("agent_0", first, False), ("agent_1", first, False),
                       ("agent_0", second, True), ("agent_1", second, True)]
    assert history[0]["episode_reward"] == pytest.approx(first + second)
    assert a0.q_table is a1.q_table
    before = copy.deepcopy(a0.q_table.q_table)
    quantitative = evaluate_policy(a0, a1, variant, 0, 1, 4, 10, reward_config=reward_config)
    assert quantitative[0]["episode_reward"] == pytest.approx(first + second)
    recorded = evaluate_with_recorder(a0, a1, variant, 0, 1, 4, 10, reward_config=reward_config)
    assert recorder_result(recorded[0], variant)["episode_reward"] == pytest.approx(first + second)
    assert len(calculations) == 6 and len(updates) == 4
    assert a0.q_table.q_table == before


@pytest.mark.parametrize("variant", VARIANT_ORDER)
def test_frozen_evaluation_identical_recording_and_unseen_states(variant):
    agents = pair(7, epsilon=.8)
    table = agents[0].q_table
    table.q_table[(90, 90, 80, 80)] = {action: float(action.value) for action in Action}
    before = copy.deepcopy(table.q_table)
    rows = evaluate_policy(*agents, variant, 0, 3, 4, 5, seed_index=0, reward_config=REWARD_VARIANTS[variant])
    assert table.q_table == before and agents[0].epsilon == agents[1].epsilon == .8
    fresh = pair(7, epsilon=.8)
    fresh[0].q_table.q_table = copy.deepcopy(before)
    recordings = evaluate_with_recorder(*fresh, variant, 0, 3, 4, 5, reward_config=REWARD_VARIANTS[variant])
    for expected, recorder in zip(rows, recordings):
        assert {k: v for k, v in expected.items() if k != "method"} == {
            k: v for k, v in recorder_result(recorder, variant).items() if k != "variant"}
    assert fresh[0].q_table.q_table == before
    empty = pair()
    evaluate_with_recorder(*empty, variant, 0, 1, 4, 3, reward_config=REWARD_VARIANTS[variant])
    assert empty[0].q_table.q_table == {}


def test_seed_aggregation_sample_sd_and_no_success():
    rows = [{"variant": "V", "training_seed": s, "captured": c, "capture_time": t,
             "episode_length": n, "episode_reward": 2}
            for s, c, t, n in ((0, 1, 2, 2), (0, 0, None, 10),
                               (1, 0, None, 10), (1, 0, None, 10), (1, 0, None, 10))]
    summary, seeds = aggregate_variant(rows)
    assert summary["capture_rate_mean"] == .25  # Not pooled 1/5.
    assert summary["capture_rate_std"] == pytest.approx(.5 / math.sqrt(2))
    assert summary["episode_length_mean"] == 8
    assert summary["capture_time_mean"] == 2 and summary["capture_time_valid_seeds"] == 1
    assert math.isnan(seeds[1]["capture_time"]) and math.isnan(summary["capture_time_std"])
    zero, _ = aggregate_variant(rows[1:2] + rows[2:])
    assert zero["capture_time_valid_seeds"] == 0 and math.isnan(zero["capture_time_mean"])


def test_threshold_complete_window_and_completed_episode_index():
    history = [{"episode": i, "captured": 1} for i in range(99)]
    assert threshold_episode(history) is None
    history.append({"episode": 99, "captured": 1})
    assert threshold_episode(history) == 100
    sequence = [0] * 100 + [1] * 50
    assert threshold_episode([{"episode": i, "captured": v} for i, v in enumerate(sequence)]) == 150
    assert threshold_episode([{"episode": i, "captured": 0} for i in range(300)]) is None


def test_default_helper_compatibility(tmp_path):
    config = ExperimentConfig(grid_size=4, max_steps=3, train_episodes=3, eval_episodes=2,
                              seeds=[0], output_dir=tmp_path / "legacy")
    old = train_cooperative_q(0, config, True)
    explicit = train_cooperative_q(0, replace(config, output_dir=tmp_path / "explicit"), True,
                                  reward_config=RewardConfig())
    assert old[0].q_table.q_table == explicit[0].q_table.q_table and old[2] == explicit[2]
    assert "rolling_episode_length" not in old[2][0]
    assert evaluate_policy(*pair(3), "M", 0, 2, 4, 3) == evaluate_policy(
        *pair(3), "M", 0, 2, 4, 3, reward_config=RewardConfig())
    assert [r.trajectory for r in evaluate_with_recorder(*pair(3), "M", 0, 2, 4, 3)] == [
        r.trajectory for r in evaluate_with_recorder(*pair(3), "M", 0, 2, 4, 3, reward_config=RewardConfig())]


@pytest.fixture(scope="module")
def tiny_studies(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablations")
    configs = [AblationConfig(grid_size=4, max_steps=5, train_episodes=4, eval_episodes=3,
                             seeds=[0, 1], output_dir=root / name) for name in ("a", "b")]
    for config in configs:
        run_reward_ablation(config)
    return configs


def test_tiny_study_inventory_seed_control_and_fresh_wrappers(tiny_studies):
    config = tiny_studies[0]
    assert len(list(config.checkpoints_dir.rglob("*.pkl"))) == 8
    assert len(list(config.training_dir.glob("*.csv"))) == 8
    identities = read_raw_results(config.summaries_dir / "checkpoint_identities.csv")
    assert len({r["checkpoint"] for r in identities}) == 8
    for variant in VARIANT_ORDER:
        rows = read_raw_results(config.raw_dir / f"{variant}.csv")
        assert len(rows) == 6
        assert [int(r["evaluation_seed"]) for r in rows] == list(range(10_000_000, 10_000_006))
        for index, seed in enumerate(config.seeds):
            history = read_raw_results(config.history_path(variant, seed))
            assert len(history) == 4 and all(math.isnan(float(r["rolling_capture_rate"])) for r in history)
            agents = load_ablation_agents(config, variant, seed, index)
            fresh = load_ablation_agents(config, variant, seed, index)
            assert agents[0] is not fresh[0] and agents[0].q_table is not fresh[0].q_table
            assert agents[0].q_table.q_table and agents[0].q_table is agents[1].q_table
    speed = read_raw_results(config.summaries_dir / "learning_speed_per_seed.csv")
    assert all(row["status"] == "Not reached" and row["episodes_completed"] == "" for row in speed)
    saved = json.loads((config.summaries_dir / "ablation_config.json").read_text())
    assert saved["seed_schedules"][1]["training_environment_first"] == 1_000_000
    assert json.loads((config.summaries_dir / "study_status.json").read_text())["complete"]


def test_repeated_tiny_studies_identical_numerical_artifacts(tiny_studies):
    a, b = tiny_studies
    files = [p for p in a.output_dir.rglob("*") if p.is_file() and p.suffix in (".csv", ".pkl")]
    files += list(a.trajectories_dir.glob("*.json"))
    assert files
    for path in files:
        assert path.read_bytes() == (b.output_dir / path.relative_to(a.output_dir)).read_bytes(), str(path)


def test_analysis_only_rebuild_and_overwrite_protection(tiny_studies):
    config = tiny_studies[0]
    before = (config.summaries_dir / "reward_ablation_summary.csv").read_bytes()
    analyze_ablation(load_saved_config(config.output_dir))
    assert before == (config.summaries_dir / "reward_ablation_summary.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_reward_ablation(config)


def test_conditional_threshold_counts_and_missing_capture_plot(tiny_studies, tmp_path):
    source = tiny_studies[0]
    config = replace(source, output_dir=tmp_path, train_episodes=100)
    for variant in VARIANT_ORDER:
        rows = read_raw_results(source.raw_dir / f"{variant}.csv")
        behavior = read_raw_results(source.behavioral_dir / f"{variant}_metrics.csv")
        # A zero-success condition must have no fabricated capture time in plots.
        for row in rows + behavior:
            row["captured"] = 0
            row["capture_time"] = None
        save_rows(rows, config.raw_dir / f"{variant}.csv")
        save_rows(behavior, config.behavioral_dir / f"{variant}_metrics.csv")
        for seed in config.seeds:
            history = [{"variant": variant, "training_seed": seed, "episode": episode,
                        "captured": int(seed == 0 and episode < 50), "episode_reward": 0,
                        "rolling_capture_rate": .5 if seed == 0 and episode == 99 else float("nan"),
                        "rolling_episode_length": 5 if episode == 99 else float("nan")}
                       for episode in range(100)]
            save_rows(history, config.history_path(variant, seed))
    summaries = analyze_ablation(config)
    assert all(row["capture_time_valid_seeds"] == 0 and math.isnan(row["capture_time_mean"]) for row in summaries)
    speed = read_raw_results(config.summaries_dir / "learning_speed_summary.csv")
    assert all(int(row["reached_seed_count"]) == 1 and float(row["conditional_threshold_episode_mean"]) == 100 for row in speed)
    assert all(math.isnan(float(row["conditional_threshold_episode_std"])) for row in speed)
    assert (config.plots_dir / "capture_time_ablation.png").stat().st_size > 0


def test_analysis_rejects_wrong_evaluation_seed_schedule(tiny_studies, tmp_path):
    source = tiny_studies[0]
    config = replace(source, output_dir=tmp_path)
    rows = read_raw_results(source.raw_dir / "full_reward.csv")
    rows[0]["evaluation_seed"] = 3
    save_rows(rows, config.raw_dir / "full_reward.csv")
    with pytest.raises(ValueError, match="seed schedule"):
        analyze_ablation(config)


def test_fresh_training_tables_and_isolated_paths(monkeypatch, tmp_path):
    created = []
    original = SharedQTable.__init__
    def init(table, *args, **kwargs):
        original(table, *args, **kwargs)
        assert table.q_table == {}
        created.append(table)
    monkeypatch.setattr(SharedQTable, "__init__", init)
    config = AblationConfig(grid_size=4, max_steps=2, train_episodes=1, eval_episodes=1,
                            seeds=[0], output_dir=tmp_path)
    for variant in VARIANT_ORDER:
        train_cooperative_q(0, config, reward_config=REWARD_VARIANTS[variant],
                            checkpoint_path=config.checkpoint_path(variant, 0), history_path=config.history_path(variant, 0))
    assert len(created) == len({id(table) for table in created}) == 4
    assert all(config.checkpoint_path(v, 0).exists() and config.history_path(v, 0).exists() for v in VARIANT_ORDER)


@pytest.mark.parametrize("kwargs", [{"seeds": [0, 0]}, {"seeds": [-1]}, {"rolling_window": 0},
                                     {"train_episodes": 1_000_001}, {"evaluation_seed_base": 0}])
def test_invalid_control_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        AblationConfig(**kwargs)
