import json
import os
from typing import List, Dict, Any

class TrajectoryRecorder:
    def __init__(self):
        self.trajectory = []
        self.metadata = {}

    def record_step(self, step: int, state: Dict[str, Any], actions: Dict[str, Any], info: Dict[str, Any], reward: float):
        """Records a single step in the environment."""
        self.trajectory.append({
            "step": step,
            "agent_0_position": [state["agent_0"].x, state["agent_0"].y],
            "agent_1_position": [state["agent_1"].x, state["agent_1"].y],
            "target_position": [state["target"].x, state["target"].y],
            "agent_0_action": actions.get("agent_0", "STAY").name if hasattr(actions.get("agent_0", "STAY"), "name") else str(actions.get("agent_0")),
            "agent_1_action": actions.get("agent_1", "STAY").name if hasattr(actions.get("agent_1", "STAY"), "name") else str(actions.get("agent_1")),
            "target_action": info.get("target_action", "STAY").name if hasattr(info.get("target_action", "STAY"), "name") else str(info.get("target_action")),
            "captured": bool(info.get("captured", False)),
            "reward": float(reward),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False))
        })

    def set_metadata(
        self,
        method: str,
        seed: int,
        eval_seed: int,
        grid_size: int,
        max_steps: int,
        checkpoint: str = "N/A",
        selection_rule: str = "unselected evaluation episode",
    ):
        self.metadata = {
            "method": method,
            "training_seed": seed,
            "evaluation_seed": eval_seed,
            "grid_size": grid_size,
            "max_steps": max_steps,
            "checkpoint": checkpoint,
            "selection_rule": selection_rule,
        }

    def save(self, filepath: str):
        """Serializes the trajectory to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "metadata": self.metadata,
            "trajectory": self.trajectory
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(filepath: str) -> 'TrajectoryRecorder':
        """Loads a trajectory from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        recorder = TrajectoryRecorder()
        recorder.metadata = data.get("metadata", {})
        recorder.trajectory = data.get("trajectory", [])
        return recorder
