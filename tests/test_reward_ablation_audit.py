"""Independent audit checks, including deliberately corrupted capture flags."""

from itertools import permutations

import pytest

from env.actions import Action
from env.target_capture_env import TargetCaptureEnv
from experiments.audit_reward_ablation import (
    distance, geometric_capture, independent_rollouts, make_agents, reference_transition,
)


def test_distance_minimum_is_capture_under_collision_rule():
    cells = [(x, y) for x in range(3) for y in range(3)]
    for a0, a1, target in permutations(cells, 3):
        positions = {"agent_0": a0, "agent_1": a1, "target": target}
        team_distance = distance(a0, target) + distance(a1, target)
        assert team_distance >= 2
        assert (team_distance == 2) == geometric_capture(positions)


def test_reference_sequential_hunter_and_target_collisions():
    class DownRNG:
        def choice(self, valid):
            assert Action.DOWN in valid
            return Action.DOWN
    positions = {"agent_0": (0, 0), "agent_1": (2, 0), "target": (1, 1)}
    next_positions, target_action = reference_transition(
        positions, {"agent_0": Action.RIGHT, "agent_1": Action.LEFT}, DownRNG(), 4)
    assert next_positions == {"agent_0": (1, 0), "agent_1": (2, 0), "target": (1, 1)}
    assert target_action == Action.DOWN
    assert positions["agent_0"] == (0, 0)  # Reference calculation leaves old state intact.


def test_independent_audit_keeps_unseen_table_empty():
    agents = make_agents(None, 123)
    rows, checks = independent_rollouts(agents, "full_reward", 0, 3, 4, 3, 100, 0)
    assert len(rows) == 3 and checks["transitions"] > 0
    assert checks["seen_state_decisions"] == 0
    assert agents[0].q_table.q_table == {}


def test_independent_audit_rejects_false_capture_flags(monkeypatch):
    class CorruptCaptureEnvironment(TargetCaptureEnv):
        def step(self, actions):
            state, info = super().step(actions)
            info["captured"] = not info["captured"]
            return state, info
    monkeypatch.setattr("experiments.audit_reward_ablation.TargetCaptureEnv", CorruptCaptureEnvironment)
    with pytest.raises(AssertionError):
        independent_rollouts(make_agents(None, 123), "full_reward", 0, 1, 4, 3, 100, 0)
