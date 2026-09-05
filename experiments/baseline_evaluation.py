import argparse
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.target_capture_env import TargetCaptureEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from env.rewards import RewardCalculator

def evaluate_agent(agent_class, num_episodes: int = 100, grid_size: int = 10, max_steps: int = 100):
    """
    Evaluates a specific agent class in the TargetCaptureEnv.
    Returns metrics: capture_rate, avg_steps, avg_reward
    """
    env = TargetCaptureEnv(grid_size=grid_size, max_steps=max_steps)
    reward_calc = RewardCalculator()
    
    successes = 0
    total_steps = 0
    total_reward = 0.0
    
    for ep in range(num_episodes):
        state = env.reset()
        
        if agent_class == RandomAgent:
            agent0 = agent_class(seed=ep)
            agent1 = agent_class(seed=ep+1000)
        else:
            agent0 = agent_class()
            agent1 = agent_class()
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # Create individual observations
            obs0 = {"agent_position": state["agent_0"], "target_position": state["target"]}
            obs1 = {"agent_position": state["agent_1"], "target_position": state["target"]}
            
            # Select actions
            action0 = agent0.select_action(obs0)
            action1 = agent1.select_action(obs1)
            
            # Step environment
            next_state, info = env.step({"agent_0": action0, "agent_1": action1})
            
            # Calculate reward for this step
            rewards = reward_calc.calculate(
                agents=[env.agent_0, env.agent_1],
                target=env.target,
                previous_positions=state,
                captured=info.get("captured", False)
            )
            episode_reward += rewards["total_reward"]
            
            state = next_state
            
            if info.get("terminated", False) or info.get("truncated", False):
                done = True
                
        # Update metrics
        if info.get("captured", False):
            successes += 1
        total_steps += info.get("step", env.current_step)
        total_reward += episode_reward
        
    capture_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_reward = total_reward / num_episodes
    
    return capture_rate, avg_steps, avg_reward

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Evaluation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    args = parser.parse_args()
    
    print(f"Running Baseline Evaluation ({args.episodes} episodes)")
    print("-" * 40)
    
    # 1. Evaluate Random Agent
    rand_cr, rand_steps, rand_reward = evaluate_agent(RandomAgent, args.episodes)
    print("Random Agent:")
    print(f"Capture Rate: {rand_cr:.2f}")
    print(f"Average Steps: {rand_steps:.2f}")
    print(f"Average Reward: {rand_reward:.2f}")
    print("-" * 40)
    
    # 2. Evaluate Heuristic Agent
    heur_cr, heur_steps, heur_reward = evaluate_agent(HeuristicAgent, args.episodes)
    print("Heuristic Agent:")
    print(f"Capture Rate: {heur_cr:.2f}")
    print(f"Average Steps: {heur_steps:.2f}")
    print(f"Average Reward: {heur_reward:.2f}")

if __name__ == "__main__":
    main()
