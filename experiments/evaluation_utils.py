"""Common, frozen-policy evaluation used by Phases 11 and 12."""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from env.rewards import RewardCalculator
from env.target_capture_env import TargetCaptureEnv


def q_table_snapshot(agent):
    """Return a deep copy of an agent's learned table, if it has one."""
    table = getattr(agent, "q_table", None)
    if table is None:
        return None
    raw = getattr(table, "q_table", table)
    return deepcopy(raw)


def make_observations(state):
    common = {
        "agent_0": state["agent_0"],
        "agent_1": state["agent_1"],
        "target": state["target"],
    }
    return (
        {**common, "agent_position": state["agent_0"], "target_position": state["target"]},
        {**common, "agent_position": state["agent_1"], "target_position": state["target"]},
    )


def evaluate_policy(
    agent0,
    agent1,
    method_name: str,
    training_seed: int,
    eval_episodes: int,
    grid_size: int,
    max_steps: int,
    evaluation_seed_base: int = 10_000_000,
    seed_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a pair without learning or changing its Q-table."""
    env = TargetCaptureEnv(grid_size=grid_size, max_steps=max_steps)
    reward_calc = RewardCalculator()
    old_eps = [getattr(agent0, "epsilon", None), getattr(agent1, "epsilon", None)]
    before = [q_table_snapshot(agent0), q_table_snapshot(agent1)]
    for agent in (agent0, agent1):
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

    results = []
    index = training_seed if seed_index is None else seed_index
    try:
        for episode in range(eval_episodes):
            eval_seed = evaluation_seed_base + index * eval_episodes + episode
            state = env.reset(seed=eval_seed)
            episode_reward = 0.0

            while True:
                obs0, obs1 = make_observations(state)
                action0 = agent0.select_action(obs0)
                action1 = agent1.select_action(obs1)
                next_state, info = env.step({"agent_0": action0, "agent_1": action1})
                reward = reward_calc.calculate(
                    agents=[env.agent_0, env.agent_1],
                    target=env.target,
                    previous_positions=state,
                    captured=info["captured"],
                )["total_reward"]
                episode_reward += reward
                state = next_state
                if info["terminated"] or info["truncated"]:
                    break

            captured = int(info["captured"])
            length = int(info["step"])
            results.append({
                "method": method_name,
                "training_seed": training_seed,
                "evaluation_seed": eval_seed,
                "episode": episode,
                "captured": captured,
                "episode_length": length,
                "episode_reward": episode_reward,
                "capture_time": length if captured else None,
            })
    finally:
        for agent, epsilon in zip((agent0, agent1), old_eps):
            if epsilon is not None:
                agent.epsilon = epsilon

    after = [q_table_snapshot(agent0), q_table_snapshot(agent1)]
    if before != after:
        raise RuntimeError("evaluation mutated a learned Q-table")
    return results
