import pytest
import random
from env.actions import Action
from env.position import Position
from algorithms.cooperative_q_learning import SharedQTable
from agents.shared_q_agent import encode_cooperative_state

def test_shared_q_table_updates():
    """Test 1 & 9: Verify shared q-table correctly stores updates across states."""
    table = SharedQTable(learning_rate=1.0, gamma=0.9)
    state = (1, 1, 2, 2)
    next_state = (0, 0, 1, 1)
    
    # Init ensures zero
    assert table.get_q_values(state)[Action.UP] == 0.0
    
    # Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
    # Q(s,a) = 0 + 1.0 * [10 + 0.9 * 0 - 0] = 10
    table.update(state, Action.UP, 10.0, next_state, done=False)
    
    assert table.get_q_values(state)[Action.UP] == 10.0

def test_terminal_update():
    """Test 8: Verify terminal updates don't bootstrap."""
    table = SharedQTable(learning_rate=1.0, gamma=0.9)
    state = (1, 1, 2, 2)
    next_state = (0, 0, 1, 1)
    
    # If done=True, max Q from next state shouldn't be added. Target Q should just be reward.
    # We'll artificially set next state's max Q to something high.
    table.get_q_values(next_state)[Action.UP] = 100.0
    
    table.update(state, Action.DOWN, 20.0, next_state, done=True)
    
    # Expect exactly 20.0, not 20.0 + 0.9*100.0
    assert table.get_q_values(state)[Action.DOWN] == 20.0

def test_agent_centric_state_encoding():
    """Test 2 & 3: Agent-Centric State Encoding and Symmetry"""
    agent0 = Position(2, 2)
    agent1 = Position(4, 2)
    target = Position(3, 5)
    
    # For Agent 0:
    s0 = encode_cooperative_state(agent0, agent1, target)
    assert s0 == (1, 3, 2, 0) # target_dx, target_dy, teammate_dx, teammate_dy
    
    # For Agent 1:
    s1 = encode_cooperative_state(agent1, agent0, target)
    assert s1 == (-1, 3, -2, 0)

def test_random_tie_breaking():
    """Test 6: Random Tie Breaking"""
    table = SharedQTable()
    state = (0,0,0,0)
    rng = random.Random(42)
    
    # All Q-values are 0 initially, meaning they are tied.
    actions_chosen = set()
    for _ in range(100):
        actions_chosen.add(table.get_best_action(state, rng))
        
    assert len(actions_chosen) > 1 # Should have picked multiple different actions randomly

def test_valid_action():
    """Test 4: Valid Action output"""
    table = SharedQTable()
    state = (1, 1, -1, -1)
    rng = random.Random(1)
    
    action = table.get_best_action(state, rng)
    assert isinstance(action, Action)
    assert action in Action
