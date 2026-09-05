from .actions import Action
from .position import Position
from .entities import Agent, Target
from .grid_world import GridWorld

class MovementController:
    """Handles entity movement in the GridWorld."""

    @staticmethod
    def calculate_new_position(position: Position, action: Action) -> Position:
        """Calculates intended new position without applying boundaries."""
        x, y = position.x, position.y
        if action == Action.UP:
            y += 1
        elif action == Action.DOWN:
            y -= 1
        elif action == Action.RIGHT:
            x += 1
        elif action == Action.LEFT:
            x -= 1
        elif action == Action.STAY:
            pass
        return Position(x, y)

    @staticmethod
    def move_agent(agent: Agent, action: Action, grid: GridWorld, occupied_positions: list = None) -> None:
        """
        Updates the agent's position based on the action and grid boundaries.
        Prevents moving into occupied positions.
        """
        new_pos = MovementController.calculate_new_position(agent.position, action)
        if grid.is_valid_position(new_pos):
            if occupied_positions is None or new_pos not in occupied_positions:
                agent.set_position(new_pos)

    @staticmethod
    def move_target(target: Target, action: Action, grid: GridWorld, occupied_positions: list = None) -> None:
        """
        Updates the target's position based on the action and grid boundaries.
        Prevents moving into occupied positions.
        """
        new_pos = MovementController.calculate_new_position(target.position, action)
        if grid.is_valid_position(new_pos):
            if occupied_positions is None or new_pos not in occupied_positions:
                target.set_position(new_pos)
