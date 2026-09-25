import pytest
from env import Agent, Target, Position
from env.capture import CaptureChecker
from env.target_capture_env import TargetCaptureEnv
from env.actions import Action

def test_successful_capture():
    target = Target(Position(5, 5))
    agent0 = Agent("agent_0", Position(5, 4))
    agent1 = Agent("agent_1", Position(5, 6))
    
    assert CaptureChecker.is_captured([agent0, agent1], target) is True

def test_not_captured():
    target = Target(Position(5, 5))
    agent0 = Agent("agent_0", Position(1, 1))
    agent1 = Agent("agent_1", Position(8, 8))
    
    assert CaptureChecker.is_captured([agent0, agent1], target) is False

def test_diagonal_case():
    target = Target(Position(5, 5))
    agent0 = Agent("agent_0", Position(6, 6))
    agent1 = Agent("agent_1", Position(5, 4))
    
    assert CaptureChecker.is_captured([agent0, agent1], target) is False

def test_environment_termination():
    env = TargetCaptureEnv(grid_size=10, max_steps=100)
    env.reset(seed=1)
    
    # Mock target policy to STAY so it doesn't run away
    env.target_policy.choose_action = lambda t, g: Action.STAY
    
    # Manually position agents and target to test the capture
    env.target.position = Position(5, 5)
    env.agent_0.position = Position(5, 3)
    env.agent_1.position = Position(5, 7)
    
    # Actions to move them adjacent to the target
    actions = {
        "agent_0": Action.UP,
        "agent_1": Action.DOWN
    }
    
    state, info = env.step(actions)
    
    assert info["captured"] is True
    assert info["terminated"] is True
    assert env.captured is True

def test_reset_after_capture():
    env = TargetCaptureEnv(grid_size=10, max_steps=100)
    env.reset(seed=2)
    
    # Mock target policy to STAY
    env.target_policy.choose_action = lambda t, g: Action.STAY
    
    env.target.position = Position(5, 5)
    env.agent_0.position = Position(5, 4)
    env.agent_1.position = Position(5, 6)
    
    actions = {"agent_0": Action.STAY, "agent_1": Action.STAY}
    _, info = env.step(actions)
    
    assert env.captured is True
    
    env.reset(seed=3)
    assert env.captured is False


def test_capture_on_final_step_is_terminated_and_truncated():
    env = TargetCaptureEnv(grid_size=10, max_steps=1)
    env.reset(seed=5)
    env.target_policy.choose_action = lambda target, grid: Action.STAY
    env.target.position = Position(5, 5)
    env.agent_0.position = Position(5, 4)
    env.agent_1.position = Position(5, 6)
    _, info = env.step({"agent_0": Action.STAY, "agent_1": Action.STAY})
    assert info["captured"] is True
    assert info["terminated"] is True
    assert info["truncated"] is True
