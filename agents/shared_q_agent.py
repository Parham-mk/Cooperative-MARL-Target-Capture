import random
from typing import Dict, Any, Tuple, Optional
from env.actions import Action
from env.position import Position
from .base_agent import BaseAgent
from algorithms.cooperative_q_learning import SharedQTable

def encode_cooperative_state(
    agent_pos: Position, 
    teammate_pos: Position, 
    target_pos: Position
) -> Tuple[int, int, int, int]:
    """
    Creates an agent-centric relative state representation.
    Returns: (target_dx, target_dy, teammate_dx, teammate_dy)
    """
    return (
        target_pos.x - agent_pos.x,
        target_pos.y - agent_pos.y,
        teammate_pos.x - agent_pos.x,
        teammate_pos.y - agent_pos.y
    )

class SharedQAgent(BaseAgent):
    """
    Agent that uses a central SharedQTable but acts on an agent-centric state.
    """
    def __init__(
        self,
        shared_q_table: SharedQTable,
        agent_id: str,
        teammate_id: str,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: Optional[int] = None
    ):
        self.q_table = shared_q_table
        self.agent_id = agent_id
        self.teammate_id = teammate_id
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = random.Random(seed)
        
    def _get_state_key(self, observation: Dict[str, Any]) -> Tuple[int, int, int, int]:
        return encode_cooperative_state(
            agent_pos=observation[self.agent_id],
            teammate_pos=observation[self.teammate_id],
            target_pos=observation["target"]
        )

    def select_action(self, observation: Dict[str, Any]) -> Action:
        """Epsilon-greedy action selection."""
        state_key = self._get_state_key(observation)
        
        # Explore
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))
            
        # Exploit
        return self.q_table.get_best_action(state_key, self.rng)
        
    def update(self, obs: Dict[str, Any], action: Action, reward: float, next_obs: Dict[str, Any], done: bool):
        """Passes the transition to the shared Q-table."""
        state_key = self._get_state_key(obs)
        next_state_key = self._get_state_key(next_obs)
        self.q_table.update(state_key, action, reward, next_state_key, done)

    def decay_epsilon(self):
        """Decays exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
