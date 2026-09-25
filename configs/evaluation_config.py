"""Defaults and typed configuration for Phase 11 and Phase 12 experiments."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

GRID_SIZE = 10
MAX_STEPS = 100
TRAIN_EPISODES = 5000
EVAL_EPISODES = 500
SEEDS = [0, 1, 2, 3, 4]

RAW_RESULTS_DIR = "results/raw"
SUMMARIES_DIR = "results/summaries"
PLOTS_DIR = "results/plots"
CHECKPOINTS_DIR = "results/checkpoints"


@dataclass(frozen=True)
class ExperimentConfig:
    grid_size: int = GRID_SIZE
    max_steps: int = MAX_STEPS
    train_episodes: int = TRAIN_EPISODES
    eval_episodes: int = EVAL_EPISODES
    seeds: Sequence[int] = tuple(SEEDS)
    output_dir: Path = Path("results")
    evaluation_seed_base: int = 10_000_000
    action_seed_base: int = 20_000_000
    training_seed_stride: int = 1_000_000

    def __post_init__(self):
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.max_steps < 1 or self.train_episodes < 1 or self.eval_episodes < 1:
            raise ValueError("step and episode counts must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def summaries_dir(self) -> Path:
        return self.output_dir / "summaries"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def training_dir(self) -> Path:
        return self.output_dir / "training"

    def evaluation_seed(self, training_seed: int, episode: int) -> int:
        seed_index = self.seeds.index(training_seed) if training_seed in self.seeds else training_seed
        return self.evaluation_seed_base + seed_index * self.eval_episodes + episode

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        data["output_dir"] = str(self.output_dir)
        return data
