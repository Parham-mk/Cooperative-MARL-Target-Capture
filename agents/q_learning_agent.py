import random
import pickle
from typing import Dict, Any, Tuple
from env.actions import Action
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Independent Q-Learning Agent that maintains a simple tabular Q-function.
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state_tuple -> dict of action -> q_value
        self.q_table: Dict[Tuple[int, int, int, int], Dict[Action, float]] = {}

    def _get_state_key(self, observation: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Converts observation into a simple discrete tuple."""
        agent_pos = observation["agent_position"]
        target_pos = observation["target_position"]
        return (agent_pos.x, agent_pos.y, target_pos.x, target_pos.y)

    def _ensure_state_exists(self, state_key: Tuple[int, int, int, int]):
        """Initializes Q-values for unvisited states."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in Action}

    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Epsilon-greedy action selection."""
        state_key = self._get_state_key(observation)
        self._ensure_state_exists(state_key)
        
        # Explore
        if random.random() < self.epsilon:
            return random.choice(list(Action))
            
        # Exploit
        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        
        # Break ties randomly
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
        
    def update(self, obs: Dict[str, Any], action: Action, reward: float, next_obs: Dict[str, Any]):
        """Q-learning update rule."""
        state_key = self._get_state_key(obs)
        next_state_key = self._get_state_key(next_obs)
        
        self._ensure_state_exists(state_key)
        self._ensure_state_exists(next_state_key)
        
        best_next_q = max(self.q_table[next_state_key].values())
        current_q = self.q_table[state_key][action]
        
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        self.q_table[state_key][action] = current_q + self.learning_rate * (
            reward + self.gamma * best_next_q - current_q
        )

    def decay_epsilon(self):
        """Reduces epsilon after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath: str):
        """Saves Q-table to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath: str):
        """Loads Q-table from disk."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
