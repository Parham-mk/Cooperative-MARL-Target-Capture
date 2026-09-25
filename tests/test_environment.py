import pytest
from env.target_capture_env import TargetCaptureEnv
from env.actions import Action
from env.position import Position

def test_environment_creation():
    env = TargetCaptureEnv(grid_size=5, max_steps=10)
    assert env.grid_size == 5
    assert env.max_steps == 10
    
    env.reset(seed=1)
    assert env.grid is not None
    assert env.agent_0 is not None
    assert env.agent_1 is not None
    assert env.target is not None

def test_reset():
    env = TargetCaptureEnv(grid_size=5)
    state = env.reset(seed=2)
    
    assert "agent_0" in state
    assert "agent_1" in state
    assert "target" in state
    
    pos0 = state["agent_0"]
    pos1 = state["agent_1"]
    post = state["target"]
    
    # Ensure all are inside grid
    assert env.grid.is_valid_position(pos0)
    assert env.grid.is_valid_position(pos1)
    assert env.grid.is_valid_position(post)
    
    # Ensure no initial overlap
    assert pos0 != pos1
    assert pos0 != post
    assert pos1 != post

def test_step_execution():
    env = TargetCaptureEnv(grid_size=5)
    env.reset(seed=3)
    
    actions = {
        "agent_0": Action.UP,
        "agent_1": Action.LEFT
    }
    
    state, info = env.step(actions)
    
    assert env.current_step == 1
    assert info["step"] == 1
    
    assert "agent_0" in state
    assert "agent_1" in state
    assert "target" in state

def test_reproducibility():
    env1 = TargetCaptureEnv(grid_size=5)
    state1 = env1.reset(seed=42)
    
    env2 = TargetCaptureEnv(grid_size=5)
    state2 = env2.reset(seed=42)
    
    assert state1["agent_0"] == state2["agent_0"]
    assert state1["agent_1"] == state2["agent_1"]
    assert state1["target"] == state2["target"]

def test_episode_limit():
    env = TargetCaptureEnv(grid_size=5, max_steps=5)
    env.reset(seed=4)
    
    for _ in range(4):
        env.step({"agent_0": Action.STAY, "agent_1": Action.STAY})
        assert not env.is_done()
        
    env.step({"agent_0": Action.STAY, "agent_1": Action.STAY})
    assert env.is_done()

def test_random_simulation():
    env = TargetCaptureEnv(grid_size=10, max_steps=100)
    env.reset(seed=99)
    
    import random
    rng = random.Random(123)
    actions_list = list(Action)
    
    for _ in range(100):
        actions = {
            "agent_0": rng.choice(actions_list),
            "agent_1": rng.choice(actions_list)
        }
        state, info = env.step(actions)
        
        # Verify valid positions
        assert env.grid.is_valid_position(state["agent_0"])
        assert env.grid.is_valid_position(state["agent_1"])
        assert env.grid.is_valid_position(state["target"])
        if info["terminated"] or info["truncated"]:
            break
    assert env.is_done()


def test_identical_seeded_full_trajectories():
    actions = [
        {"agent_0": Action.RIGHT, "agent_1": Action.UP},
        {"agent_0": Action.STAY, "agent_1": Action.LEFT},
        {"agent_0": Action.DOWN, "agent_1": Action.RIGHT},
    ]
    trajectories = []
    for _ in range(2):
        env = TargetCaptureEnv(grid_size=5, max_steps=3)
        states = [env.reset(seed=123)]
        infos = []
        for joint_action in actions:
            state, info = env.step(joint_action)
            states.append(state)
            infos.append(info)
            if info["terminated"] or info["truncated"]:
                break
        trajectories.append((states, infos))
    assert trajectories[0] == trajectories[1]
