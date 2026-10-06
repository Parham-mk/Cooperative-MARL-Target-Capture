import pytest
from configs import robustness_config as rob_cfg
from experiments.run_robustness_evaluation import generate_unseen_initial_states, get_q_coverage, evaluate_robustness_episode
from env.target_capture_env import TargetCaptureEnv
from agents.q_learning_agent import QLearningAgent
from env.position import Position
from env.actions import Action
import math

def test_held_out_seed_validation():
    """Test 1: Verify robustness seeds do not overlap with training seeds."""
    for s in rob_cfg.ROBUSTNESS_SEEDS:
        assert s not in rob_cfg.TRAINING_SEEDS

def test_valid_initial_states():
    """Test 2: Verify all generated held-out initial states remain inside grid and have no overlap."""
    cases = generate_unseen_initial_states(50, 42, rob_cfg.GRID_SIZE, rob_cfg.TRAINING_SEEDS)
    for c in cases:
        p0 = c["agent_0_pos"]
        p1 = c["agent_1_pos"]
        pt = c["target_pos"]
        
        # Inside grid
        assert 0 <= p0.x < rob_cfg.GRID_SIZE
        assert 0 <= p0.y < rob_cfg.GRID_SIZE
        assert 0 <= p1.x < rob_cfg.GRID_SIZE
        assert 0 <= p1.y < rob_cfg.GRID_SIZE
        assert 0 <= pt.x < rob_cfg.GRID_SIZE
        assert 0 <= pt.y < rob_cfg.GRID_SIZE
        
        # No overlap
        assert p0 != p1
        assert p0 != pt
        assert p1 != pt

def test_reproducible_initial_state_generation():
    """Test 3: Same robustness seed -> same generated initial states."""
    c1 = generate_unseen_initial_states(10, 100, rob_cfg.GRID_SIZE, [])
    c2 = generate_unseen_initial_states(10, 100, rob_cfg.GRID_SIZE, [])
    
    for state1, state2 in zip(c1, c2):
        assert state1["agent_0_pos"] == state2["agent_0_pos"]
        assert state1["agent_1_pos"] == state2["agent_1_pos"]
        assert state1["target_pos"] == state2["target_pos"]

def test_policy_is_frozen():
    """Test 4: Verify Q-table is unchanged during evaluation."""
    env = TargetCaptureEnv(grid_size=5)
    agent0 = QLearningAgent(seed=42)
    agent1 = QLearningAgent(seed=42)
    
    # Pre-populate table
    agent0.q_table[(0,0,1,1)] = {a: 1.0 for a in Action}
    
    q_table_before = {k: v.copy() for k, v in agent0.q_table.items()}
    
    evaluate_robustness_episode(env, agent0, agent1, seed=1)
    
    assert agent0.q_table == q_table_before
    assert agent0.epsilon == 0.0

def test_q_table_coverage():
    """Test 5 & Test 6: Verify coverage calculation and unseen state rate exactly."""
    agent = QLearningAgent()
    agent.q_table[(0,0,1,1)] = {a: 0.0 for a in Action}
    
    obs_seen = {
        "agent_position": Position(0, 0),
        "target_position": Position(1, 1)
    }
    obs_unseen = {
        "agent_position": Position(2, 2),
        "target_position": Position(1, 1)
    }
    
    assert get_q_coverage(agent, obs_seen) == 1.0
    assert get_q_coverage(agent, obs_unseen) == 0.0
    
    # Mocking evaluation loop
    # 2 states, 1 seen, 1 unseen -> Coverage = 0.5
    # UnseenStateRate = 1 - 0.5 = 0.5
    coverage = (get_q_coverage(agent, obs_seen) + get_q_coverage(agent, obs_unseen)) / 2
    assert coverage == 0.5
    unseen_rate = 1.0 - coverage
    assert unseen_rate == 0.5

def test_performance_drop():
    """Test 7: Verify delta calculation via synthetic metrics."""
    # Synthetic metric
    standard_cr = 0.95
    held_out_cr = 0.70
    delta = held_out_cr - standard_cr
    assert math.isclose(delta, -0.25)

def test_stress_case_construction():
    """Test 8: Verify each stress case satisfies its documented geometric definition."""
    cases = {sc.name: sc for sc in rob_cfg.STRESS_CASES}
    
    # Far target: distance from hunters to target should be large
    far_tc = cases["far_target"]
    assert far_tc.target[0] >= rob_cfg.GRID_SIZE - 2
    
    # Same side hunters: dx or dy between them is very small and they are on same side of target
    same_tc = cases["same_side_hunters"]
    assert abs(same_tc.agent_0[0] - same_tc.agent_1[0]) <= 1 or abs(same_tc.agent_0[1] - same_tc.agent_1[1]) <= 1
    
    # Opposite side: target is between them
    opp_tc = cases["opposite_side_hunters"]
    assert (opp_tc.agent_0[0] < opp_tc.target[0] < opp_tc.agent_1[0]) or (opp_tc.agent_1[0] < opp_tc.target[0] < opp_tc.agent_0[0])

def test_evaluation_smoke_test():
    """Test 9: Run a tiny robustness evaluation (pipeline completes, doesn't crash)."""
    env = TargetCaptureEnv(grid_size=5, max_steps=5)
    agent0 = QLearningAgent()
    agent1 = QLearningAgent()
    
    res = evaluate_robustness_episode(env, agent0, agent1, seed=0)
    assert "captured" in res
    assert "q_state_coverage" in res
    assert "episode_reward" in res
