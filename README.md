# Emergent Cooperation in Multi-Agent Reinforcement Learning for Target Capture

This project studies cooperative behavior in multi-agent reinforcement learning.
Multiple agents learn to capture a moving target.
The goal is to investigate the emergence of cooperation.

## Current Implementation Status

Phase 10 completed:

Implemented:
- shared-policy cooperative Q-learning baseline
- agent-centric relative state representation
- shared Q-table parameters
- teammate-aware cooperative exploration
- independent and cooperative evaluation frameworks

Not implemented yet:
- sophisticated deep MARL
- learned multi-agent communication
- continuous action spaces

## Shared-Policy Cooperative Q-Learning

To introduce explicit cooperative learning while keeping the algorithm interpretable, both homogeneous hunter agents share a single tabular Q-function.

Each hunter acts from an agent-centric state containing:
- relative target position
- relative teammate position

Experience from both hunters updates the same Q-table. This experiment evaluates whether parameter sharing and teammate-aware state representations improve coordination compared with Independent Q-Learning.

## Future Roadmap

Phase 0:
Project initialization

Phase 1:
GridWorld environment

Phase 2:
Environment mechanics

Phase 3:
Baseline agents

Phase 4:
Reinforcement learning

Phase 5:
Cooperative learning experiments

Phase 6:
Analysis and documentation

## Installation

```bash
git clone <repository-url>
cd Cooperative-MARL-Target-Capture
pip install -r requirements.txt
```
