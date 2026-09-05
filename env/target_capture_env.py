import numpy as np
from typing import Dict, Tuple, Any, Optional

from .grid_world import GridWorld
from .entities import Agent, Target
from .position import Position
from .actions import Action
from .movement import MovementController
from .target_policy import RandomTargetPolicy
from .capture import CaptureChecker

class TargetCaptureEnv:
    """Core simulation environment for target capture."""

    def __init__(self, grid_size: int = 10, max_steps: int = 100, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        
        self.grid = GridWorld(grid_size)
        self.target_policy = RandomTargetPolicy(seed)
        
        self.agent_0 = None
        self.agent_1 = None
        self.target = None
        
        self.current_step = 0
        self.captured = False

    def reset(self, seed: Optional[int] = None) -> Dict[str, Position]:
        """
        Resets the environment to an initial state.
        Ensures agents and target do not overlap initially.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.target_policy = RandomTargetPolicy(seed)
            self.grid = GridWorld(self.grid_size) # GridWorld uses its own rng, though it's barely used when we pass explicit seeds.

        self.current_step = 0
        self.captured = False
        
        # We need 3 unique positions for the 2 agents and 1 target
        all_pos = self.grid.get_all_positions()
        
        # Randomly select 3 unique indices
        indices = self._rng.choice(len(all_pos), size=3, replace=False)
        
        pos_a0 = all_pos[indices[0]]
        pos_a1 = all_pos[indices[1]]
        pos_t = all_pos[indices[2]]
        
        self.agent_0 = Agent("agent_0", pos_a0)
        self.agent_1 = Agent("agent_1", pos_a1)
        self.target = Target(pos_t)
        
        return self.get_state()

    def get_state(self) -> Dict[str, Position]:
        """Returns the current state dictionary."""
        return {
            "agent_0": self.agent_0.position,
            "agent_1": self.agent_1.position,
            "target": self.target.position
        }
        
    def _get_occupied_positions(self, exclude: Any = None) -> list[Position]:
        """Helper to get positions occupied by entities, excluding a specific entity."""
        positions = []
        if self.agent_0 and self.agent_0 != exclude:
            positions.append(self.agent_0.position)
        if self.agent_1 and self.agent_1 != exclude:
            positions.append(self.agent_1.position)
        if self.target and self.target != exclude:
            positions.append(self.target.position)
        return positions

    def step(self, actions: Dict[str, Action]) -> Tuple[Dict[str, Position], Dict[str, Any]]:
        """
        Executes one environment step.
        1. Apply hunter actions
        2. Move target
        3. Update timestep
        4. Check capture
        5. Update episode status
        6. Return state and info
        """
        # 1. Apply hunter actions sequentially, preventing overlaps
        if "agent_0" in actions:
            MovementController.move_agent(
                self.agent_0, actions["agent_0"], self.grid, self._get_occupied_positions(self.agent_0)
            )
        if "agent_1" in actions:
            MovementController.move_agent(
                self.agent_1, actions["agent_1"], self.grid, self._get_occupied_positions(self.agent_1)
            )
            
        # 2. Move target, preventing overlaps
        target_action = self.target_policy.choose_action(self.target, self.grid)
        MovementController.move_target(
            self.target, target_action, self.grid, self._get_occupied_positions(self.target)
        )
        
        # 3. Update timestep
        self.current_step += 1
        
        # 4. Check capture
        self.captured = CaptureChecker.is_captured([self.agent_0, self.agent_1], self.target)
        
        # 5. Update episode status
        terminated = self.captured
        truncated = self.current_step >= self.max_steps
        
        # 6. Return state and info
        state = self.get_state()
        info = {
            "step": self.current_step,
            "captured": self.captured,
            "terminated": terminated,
            "truncated": truncated
        }
        
        return state, info

    def is_done(self) -> bool:
        """Returns True if the maximum steps are reached or target is captured."""
        return (self.current_step >= self.max_steps) or self.captured

    def render(self) -> None:
        """Renders the environment to the console."""
        print(f"Step: {self.current_step}")
        
        grid_map = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Target
        tx, ty = self.target.position.x, self.target.position.y
        grid_map[ty][tx] = 'T'
        
        # Agents (Agents might overlap with target or each other, so we overwrite)
        a0_x, a0_y = self.agent_0.position.x, self.agent_0.position.y
        grid_map[a0_y][a0_x] = 'H'
        
        a1_x, a1_y = self.agent_1.position.x, self.agent_1.position.y
        grid_map[a1_y][a1_x] = 'H'
        
        # Print grid (y-axis inverted for top-to-bottom rendering)
        for y in range(self.grid_size - 1, -1, -1):
            row_str = " ".join(grid_map[y])
            print(row_str)
        print()
