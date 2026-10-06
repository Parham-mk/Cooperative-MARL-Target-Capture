"""Controlled one-component-at-a-time Phase 13 protocol."""

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from configs.evaluation_config import ExperimentConfig
from env.rewards import RewardConfig


REWARD_VARIANTS = MappingProxyType({
    "full_reward": RewardConfig(1.0, 20.0, -0.05),
    "no_distance": RewardConfig(0.0, 20.0, -0.05),
    "no_step_penalty": RewardConfig(1.0, 20.0, 0.0),
    "no_capture_reward": RewardConfig(1.0, 0.0, -0.05),
})
VARIANT_ORDER = tuple(REWARD_VARIANTS)


@dataclass(frozen=True)
class AblationConfig(ExperimentConfig):
    output_dir: Path = Path("results/ablations")
    learning_rate: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    rolling_window: int = 100
    capture_threshold: float = 0.50

    def __post_init__(self):
        super().__post_init__()
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("training seeds must be unique nonnegative integers")
        if self.rolling_window < 1 or not 0 <= self.capture_threshold <= 1:
            raise ValueError("invalid rolling window or capture threshold")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("invalid learning parameters")
        if not 0 <= self.min_epsilon <= self.epsilon <= 1 or not 0 < self.epsilon_decay <= 1:
            raise ValueError("invalid exploration schedule")
        # Preserve Phase 11 seed schedules, but reject budgets that overlap them.
        if self.train_episodes > self.training_seed_stride:
            raise ValueError("training episodes exceed the environment seed stride")
        end_training = (len(self.seeds) - 1) * self.training_seed_stride + self.train_episodes - 1
        if self.evaluation_seed_base <= end_training:
            raise ValueError("evaluation environment seeds must follow training seeds")

    @property
    def training_dir(self):
        return self.raw_dir / "training"

    @property
    def behavioral_dir(self):
        return self.output_dir / "behavioral"

    @property
    def trajectories_dir(self):
        return self.output_dir / "trajectories"

    def checkpoint_path(self, variant, seed):
        if variant not in REWARD_VARIANTS:
            raise ValueError(f"unknown reward variant: {variant}")
        return self.checkpoints_dir / variant / f"seed_{seed}.pkl"

    def history_path(self, variant, seed):
        if variant not in REWARD_VARIANTS:
            raise ValueError(f"unknown reward variant: {variant}")
        return self.training_dir / f"{variant}_seed_{seed}.csv"

    def to_dict(self):
        data = super().to_dict()
        data["variants"] = {name: asdict(value) for name, value in REWARD_VARIANTS.items()}
        data["variant_order"] = list(VARIANT_ORDER)
        data["algorithm"] = "Shared-Policy Cooperative Q-Learning"
        data["seed_schedules"] = [
            {
                "training_seed": seed, "seed_index": index,
                "training_environment_first": index * self.training_seed_stride,
                "training_environment_last": index * self.training_seed_stride + self.train_episodes - 1,
                "training_action_rng": [seed * 2, seed * 2 + 1],
                "evaluation_environment_first": self.evaluation_seed(seed, 0),
                "evaluation_environment_last": self.evaluation_seed(seed, self.eval_episodes - 1),
                "evaluation_action_rng": [self.action_seed_base + index * 2, self.action_seed_base + index * 2 + 1],
            }
            for index, seed in enumerate(self.seeds)
        ]
        data["conventions"] = {
            "episode": "zero-based; episodes_completed = episode + 1",
            "epsilon": "after decay; epsilon_used is the value during that training episode",
            "rolling": "complete windows only; earlier values are NaN",
            "threshold": "first complete window with capture rate >= capture_threshold; conditional on attainment",
            "uncertainty": "sample standard deviation across training-seed estimates; N/A when fewer than two valid seeds",
            "evaluation": "epsilon zero, seeded random tie-breaking, no updates or unseen-state insertion",
            "behavioral_frames": "initial frame and every post-transition frame, including capture",
            "team_reward": "sum of distance improvements + capture_weight once + step_penalty once per transition",
            "shared_update_order": "select both actions from pre-step state, transition, update hunter 0 then hunter 1 with the same reward",
            "bootstrap": "zero on capture termination; retained on time-limit truncation",
            "target_policy": "existing RandomTargetPolicy; mechanics unchanged",
        }
        return data
