import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str((Path(".venv") / "matplotlib").resolve()))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Any

class GridWorldRenderer:
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
        ax.invert_yaxis() # To match array indexing

    def plot_static_trajectory(self, trajectory: List[Dict[str, Any]], title: str, filepath: str):
        """Plots the full paths of A0, A1, and Target."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._draw_base_grid(ax)
        
        a0_x, a0_y = [], []
        a1_x, a1_y = [], []
        t_x, t_y = [], []
        
        for step in trajectory:
            a0_x.append(step["agent_0_position"][0])
            a0_y.append(step["agent_0_position"][1])
            a1_x.append(step["agent_1_position"][0])
            a1_y.append(step["agent_1_position"][1])
            t_x.append(step["target_position"][0])
            t_y.append(step["target_position"][1])
            
        # Draw paths
        ax.plot(a0_x, a0_y, color=self.colors["agent_0"], alpha=0.5, label="Agent 0 Path")
        ax.plot(a1_x, a1_y, color=self.colors["agent_1"], alpha=0.5, label="Agent 1 Path")
        ax.plot(t_x, t_y, color=self.colors["target"], alpha=0.5, label="Target Path", linestyle="dashed")
        
        # Mark Start (circle) and End (star)
        if trajectory:
            ax.plot(a0_x[0], a0_y[0], marker='o', color=self.colors["agent_0"], markersize=8)
            ax.plot(a1_x[0], a1_y[0], marker='o', color=self.colors["agent_1"], markersize=8)
            ax.plot(t_x[0], t_y[0], marker='o', color=self.colors["target"], markersize=8)
            
            ax.plot(a0_x[-1], a0_y[-1], marker='*', color=self.colors["agent_0"], markersize=12)
            ax.plot(a1_x[-1], a1_y[-1], marker='*', color=self.colors["agent_1"], markersize=12)
            ax.plot(t_x[-1], t_y[-1], marker='*', color=self.colors["target"], markersize=12)
        
        ax.set_title(title)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=150)
        plt.close(fig)


