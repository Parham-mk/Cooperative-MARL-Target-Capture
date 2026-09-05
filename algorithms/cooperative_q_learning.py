import random
import pickle
from typing import Dict, Tuple, Optional
from env.actions import Action

class SharedQTable:
    """
    A single shared tabular Q-function used by multiple homogeneous agents.
    Maintains Q-values for relative, agent-centric states.
    """
    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.95):
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Mapping: state -> Dict[Action, float]
        self.q_table: Dict[Tuple[int, int, int, int], Dict[Action, float]] = {}

    def _ensure_state_exists(self, state: Tuple[int, int, int, int]):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in Action}

    def get_q_values(self, state: Tuple[int, int, int, int]) -> Dict[Action, float]:
        """Returns the dictionary of Q-values for the state."""
        self._ensure_state_exists(state)
        return self.q_table[state]

    def get_best_action(self, state: Tuple[int, int, int, int], rng: random.Random) -> Action:
        """Returns the best action, breaking ties randomly."""
        q_values = self.get_q_values(state)
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return rng.choice(best_actions)

    def update(
        self, 
        state: Tuple[int, int, int, int], 
        action: Action, 
        reward: float, 
        next_state: Tuple[int, int, int, int], 
        done: bool
    ):
        """
        Standard Q-learning update. If 'done' is true (e.g. target captured), 
        bootstrap value is zero.
        """
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
        
        current_q = self.q_table[state][action]
        
        if done:
            target_q = float(reward)
        else:
            best_next_q = max(self.q_table[next_state].values())
            target_q = reward + self.gamma * best_next_q
            
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
