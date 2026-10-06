from dataclasses import dataclass
from typing import List, Tuple
import configs.evaluation_config as eval_cfg

GRID_SIZE = eval_cfg.GRID_SIZE
MAX_STEPS = eval_cfg.MAX_STEPS

TRAINING_SEEDS = eval_cfg.SEEDS
# Condition B: Unseen seeds
ROBUSTNESS_SEEDS = [1000 + s for s in TRAINING_SEEDS]

NUM_EVAL_EPISODES = 100  # For condition B and C
NUM_INITIAL_STATE_CASES = 100

# Condition D: Stress cases
@dataclass(frozen=True)
class StressCase:
    name: str
    agent_0: Tuple[int, int]
    agent_1: Tuple[int, int]
    target: Tuple[int, int]

STRESS_CASES = [
    # Hunters far from target
    StressCase(
        name="far_target",
        agent_0=(0, 0),
        agent_1=(0, GRID_SIZE - 1),
        target=(GRID_SIZE - 1, GRID_SIZE // 2)
    ),
    # Hunters on same side
    StressCase(
        name="same_side_hunters",
        agent_0=(1, 1),
        agent_1=(1, 2),
        target=(GRID_SIZE - 2, GRID_SIZE - 2)
    ),
    # Hunters on opposite sides
    StressCase(
        name="opposite_side_hunters",
        agent_0=(1, GRID_SIZE // 2),
        agent_1=(GRID_SIZE - 2, GRID_SIZE // 2),
        target=(GRID_SIZE // 2, GRID_SIZE // 2)
    ),
    # Target near boundary
    StressCase(
        name="boundary_target",
        agent_0=(5, 5),
        agent_1=(6, 5),
        target=(0, 5)
    ),
    # Hunters widely separated
    StressCase(
        name="widely_separated_hunters",
        agent_0=(0, 0),
        agent_1=(GRID_SIZE - 1, GRID_SIZE - 1),
        target=(GRID_SIZE // 2, GRID_SIZE // 2)
    )
]

ENABLE_GRID_SIZE_TRANSFER = False
