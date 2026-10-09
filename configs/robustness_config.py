"""Frozen-policy robustness protocol; environment and training are unchanged."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StressCase:
    name: str
    agent_0: tuple[int, int]
    agent_1: tuple[int, int]
    target: tuple[int, int]


def stress_cases(grid_size: int) -> tuple[StressCase, ...]:
    """Five fixed spatial probes, scaled to the source grid (at least 4)."""
    if grid_size < 4:
        raise ValueError("robustness stress geometry requires grid_size >= 4")
    last, middle = grid_size - 1, grid_size // 2
    return (
        StressCase("far_target", (0, 0), (0, last), (last, middle)),
        StressCase("same_side_hunters", (0, 0), (0, 1), (last, last)),
        StressCase("opposite_side_hunters", (0, middle), (last, middle), (middle, middle)),
        StressCase("boundary_target", (middle, middle), (middle + 1, middle), (0, middle)),
        StressCase("widely_separated_hunters", (0, 0), (last, last), (middle, middle)),
    )


@dataclass(frozen=True)
class RobustnessConfig:
    comparison_dir: Path
    output_dir: Path = Path("results/robustness")
    eval_episodes: int = 500
    initial_state_cases: int = 500
    stress_repeats: int = 100
    held_out_seed_base: int = 50_000_000
    initial_state_seed_base: int = 51_000_000
    stress_seed_base: int = 52_000_000
    action_seed_base: int = 60_000_000
    initial_generator_seed: int = 70_000_000
    # Phase 13 audit: 2,000 episodes for each of five training seeds.
    prior_evaluation_ranges: tuple[tuple[int, int], ...] = ((30_000_000, 30_009_999),)
    verify_against: Path | None = None

    def __post_init__(self):
        for name in ("comparison_dir", "output_dir", "verify_against"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if min(self.eval_episodes, self.initial_state_cases, self.stress_repeats) < 1:
            raise ValueError("robustness episode counts must be positive")
        if min(self.held_out_seed_base, self.initial_state_seed_base, self.stress_seed_base,
               self.action_seed_base, self.initial_generator_seed) < 0:
            raise ValueError("random seeds must be nonnegative")
        if any(first < 0 or last < first for first, last in self.prior_evaluation_ranges):
            raise ValueError("invalid prior evaluation seed range")


PRIMARY_CONDITIONS = ("standard", "held_out_seeds", "held_out_initial_states")
METHODS = ("Independent Q-Learning", "Cooperative Q-Learning")
