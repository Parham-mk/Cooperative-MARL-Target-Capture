import pytest
import os
import tempfile
from env.actions import Action
from env.position import Position
from agents.q_learning_agent import QLearningAgent
from env.target_capture_env import TargetCaptureEnv

@pytest.fixture
def dummy_obs():
    return {
        "agent_position": Position(2, 2),
        "target_position": Position(5, 5)
    }

def test_agent_initialization():
    agent = QLearningAgent(learning_rate=0.2, epsilon=0.5)
    assert agent.learning_rate == 0.2
    assert agent.epsilon == 0.5
    assert isinstance(agent.q_table, dict)
    assert len(agent.q_table) == 0

def test_action_selection_validity(dummy_obs):
    agent = QLearningAgent()
    action = agent.select_action(dummy_obs)
    assert action in Action
    # Q-table should be initialized for this state
    state_key = agent._get_state_key(dummy_obs)
    assert state_key in agent.q_table
    assert len(agent.q_table[state_key]) == len(Action)

def test_q_update(dummy_obs):
    agent = QLearningAgent(learning_rate=0.5, gamma=0.9)
    next_obs = {
        "agent_position": Position(3, 2),
        "target_position": Position(5, 5)
    }
    
    state_key = agent._get_state_key(dummy_obs)
    action = Action.RIGHT
    reward = 10.0
    
    agent.update(dummy_obs, action, reward, next_obs)
    
    # Q(s,a) = 0 + 0.5 * (10 + 0.9 * 0 - 0) = 5.0
    assert agent.q_table[state_key][action] == 5.0

def test_exploration(dummy_obs):
    # Epsilon = 1.0 means always explore
    agent = QLearningAgent(epsilon=1.0)
    agent.q_table[agent._get_state_key(dummy_obs)] = {
        Action.UP: 100.0,
        Action.DOWN: -100.0,
        Action.LEFT: -100.0,
        Action.RIGHT: -100.0,
        Action.STAY: -100.0
    }
    
    # It should not always pick UP
    actions_chosen = set()
    for _ in range(100):
        actions_chosen.add(agent.select_action(dummy_obs))
        
    assert len(actions_chosen) > 1

def test_exploitation(dummy_obs):
    # Epsilon = 0.0 means always exploit
    agent = QLearningAgent(epsilon=0.0)
    agent.q_table[agent._get_state_key(dummy_obs)] = {
        Action.UP: 100.0,
        Action.DOWN: -100.0,
        Action.LEFT: -100.0,
        Action.RIGHT: -100.0,
        Action.STAY: -100.0
    }
    
    # It should ALWAYS pick UP
    for _ in range(50):
        assert agent.select_action(dummy_obs) == Action.UP

def test_training_smoke_test():
    env = TargetCaptureEnv(grid_size=5, max_steps=10)
    agent0 = QLearningAgent()
    agent1 = QLearningAgent()
    
    for ep in range(2):
        state = env.reset()
        done = False
        while not done:
            obs0 = {"agent_position": state["agent_0"], "target_position": state["target"]}
            obs1 = {"agent_position": state["agent_1"], "target_position": state["target"]}
            
            a0 = agent0.select_action(obs0)
            a1 = agent1.select_action(obs1)
            
            next_state, info = env.step({"agent_0": a0, "agent_1": a1})
            
            next_obs0 = {"agent_position": next_state["agent_0"], "target_position": next_state["target"]}
            next_obs1 = {"agent_position": next_state["agent_1"], "target_position": next_state["target"]}
            
            agent0.update(obs0, a0, 1.0, next_obs0)
            agent1.update(obs1, a1, 1.0, next_obs1)
            
            if info.get("terminated") or info.get("truncated"):
                done = True
                
        agent0.decay_epsilon()
        agent1.decay_epsilon()
        
    assert len(agent0.q_table) > 0
    assert len(agent1.q_table) > 0

def test_save_load(dummy_obs):
    agent = QLearningAgent()
    agent.q_table[agent._get_state_key(dummy_obs)] = {a: 5.0 for a in Action}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "agent.pkl")
        agent.save(filepath)
        
        agent_new = QLearningAgent()
        agent_new.load(filepath)
        
        assert agent_new.q_table == agent.q_table
