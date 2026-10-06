from typing import List, Dict
from dataclasses import dataclass, asdict
from .entities import Agent, Target
from .position import Position


@dataclass(frozen=True)
class RewardConfig:
    """Immutable parameters; all reward mathematics stays in RewardCalculator."""

    distance_weight: float = 1.0
    capture_weight: float = 20.0
    step_penalty: float = -0.05

    def calculator(self):
        return RewardCalculator(**asdict(self))

class RewardCalculator:
    """Calculates rewards for the multi-agent target capture environment."""

    def __init__(
        self,
        distance_weight: float = 1.0,
        capture_weight: float = 20.0,
        step_penalty: float = -0.05
    ):
        self.distance_weight = distance_weight
        self.capture_weight = capture_weight
        self.step_penalty = step_penalty

    def calculate_distance_reward(self, previous_distance: int, current_distance: int) -> float:
        """
        Rewards agents when they reduce the distance to the target.
        """
        return float(previous_distance - current_distance) * self.distance_weight

    def calculate_capture_reward(self, captured: bool) -> float:
        """
        Returns the team capture reward if the target is successfully captured.
        """
        return float(self.capture_weight) if captured else 0.0

    def calculate_step_penalty(self) -> float:
        """
        Returns a constant penalty applied at each step to encourage faster captures.
        """
        return float(self.step_penalty)

    def calculate_total_reward(self, distance_reward: float, capture_reward: float, step_penalty: float) -> float:
        """
        Combines all reward components into a total reward.
        """
        return distance_reward + capture_reward + step_penalty

    def calculate(
        self,
        agents: List[Agent],
        target: Target,
        previous_positions: Dict[str, Position],
        captured: bool
    ) -> Dict[str, float]:
        """
        Calculates all reward components given the current and previous state.
        
        Args:
            agents: List of current hunter agents.
            target: The current target.
            previous_positions: A dictionary mapping agent.id and "target" to their Position from the previous step.
            captured: Boolean indicating if the target was captured in this step.
            
        Returns:
            A dictionary containing individual reward components and the total reward.
        """
        prev_target_pos = previous_positions.get("target", target.position)
        
        prev_dist = 0
        curr_dist = 0
        
        for agent in agents:
            prev_agent_pos = previous_positions.get(agent.id, agent.position)
            prev_dist += prev_agent_pos.distance_to(prev_target_pos)
            curr_dist += agent.position.distance_to(target.position)
            
        dist_rew = self.calculate_distance_reward(prev_dist, curr_dist)
        cap_rew = self.calculate_capture_reward(captured)
        step_pen = self.calculate_step_penalty()
        total_rew = self.calculate_total_reward(dist_rew, cap_rew, step_pen)
        
        return {
            "distance_reward": dist_rew,
            "capture_reward": cap_rew,
            "step_penalty": step_pen,
            "total_reward": total_rew
        }
