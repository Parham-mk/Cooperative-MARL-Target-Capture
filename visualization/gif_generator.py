import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str((Path(".venv") / "matplotlib").resolve()))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Any

class GIFGenerator:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.colors = {
            "agent_0": "blue",
            "agent_1": "green",
            "target": "red",
            "grid": "lightgray"
        }
        
    def _draw_base_grid(self, ax):
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(color=self.colors["grid"], linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

    def generate(self, trajectory: List[Dict[str, Any]], title: str, filepath: str, fps: int = 5):
        """Creates an animated GIF of the trajectory."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        a0_patch = patches.Circle((0, 0), 0.3, fc=self.colors["agent_0"], label="Agent 0")
        a1_patch = patches.Circle((0, 0), 0.3, fc=self.colors["agent_1"], label="Agent 1")
        t_patch = patches.Rectangle((-0.3, -0.3), 0.6, 0.6, fc=self.colors["target"], label="Target")
        
        def init():
            self._draw_base_grid(ax)
            ax.add_patch(a0_patch)
            ax.add_patch(a1_patch)
            ax.add_patch(t_patch)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            return a0_patch, a1_patch, t_patch
            
        def update(frame_idx):
            step = trajectory[frame_idx]
            
            a0_patch.center = (step["agent_0_position"][0], step["agent_0_position"][1])
            a1_patch.center = (step["agent_1_position"][0], step["agent_1_position"][1])
            t_patch.set_xy((step["target_position"][0] - 0.3, step["target_position"][1] - 0.3))
            
            status = "Captured!" if step["captured"] else f"Reward: {step['reward']:.1f}"
            ax.set_title(f"{title}\nStep: {step['step']} | {status}")
            
            return a0_patch, a1_patch, t_patch
            
        ani = FuncAnimation(fig, update, frames=len(trajectory), init_func=init, blit=False)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ani.save(filepath, writer=PillowWriter(fps=fps))
        plt.close(fig)
