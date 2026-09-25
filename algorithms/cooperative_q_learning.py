import random
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from env.actions import Action
from algorithms.q_update import q_learning_value

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
        q_values = self.q_table.get(state)
        if q_values is None:
            q_values = {action: 0.0 for action in Action}
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
        
        best_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = q_learning_value(
            current_q=current_q,
            reward=reward,
            best_next_q=best_next_q,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            terminated=done,
        )

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
