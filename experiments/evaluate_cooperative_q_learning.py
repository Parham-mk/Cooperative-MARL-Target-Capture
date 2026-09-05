import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.target_capture_env import TargetCaptureEnv
from env.rewards import RewardCalculator
from algorithms.cooperative_q_learning import SharedQTable
from agents.shared_q_agent import SharedQAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate Shared-Policy Cooperative Q-Learning")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    parser.add_argument("--model-path", type=str, default="results/checkpoints/cooperative_q_learning.pkl")
    args = parser.parse_args()
    
    print(f"Evaluating Cooperative Q-Learning Baseline ({args.episodes} episodes)")
    
    env = TargetCaptureEnv(grid_size=args.grid_size, max_steps=100)
    reward_calc = RewardCalculator()
    
    shared_table = SharedQTable()
    if os.path.exists(args.model_path):
        shared_table.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found. Evaluating an untrained agent.")

    # Epsilon=0 for purely deterministic exploitation
    agent0 = SharedQAgent(shared_table, "agent_0", "agent_1", epsilon=0.0)
    agent1 = SharedQAgent(shared_table, "agent_1", "agent_0", epsilon=0.0)
    
    successes = 0
    total_steps = 0
    capture_steps = 0
    total_reward = 0.0
    
    for ep in range(args.episodes):
        state = env.reset(seed=ep + 9999) # Fixed seeds for evaluation fairness
        done = False
        episode_reward = 0.0
        
        while not done:
            a0 = agent0.select_action(state)
            a1 = agent1.select_action(state)
            
            next_state, info = env.step({"agent_0": a0, "agent_1": a1})
            
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
                
        if info.get("captured", False):
            successes += 1
            capture_steps += info.get("step", env.current_step)
            
        total_steps += info.get("step", env.current_step)
        total_reward += episode_reward
        
    capture_rate = successes / args.episodes
    avg_steps = total_steps / args.episodes
    avg_reward = total_reward / args.episodes
    avg_capture_time = (capture_steps / successes) if successes > 0 else 0.0
    
    print("-" * 40)
    print("Cooperative Shared-Policy Q-Learning:")
    print(f"Capture Rate:       {capture_rate:.2f}")
    print(f"Mean Episode Len:   {avg_steps:.2f}")
    print(f"Mean Capture Time:  {avg_capture_time:.2f}")
    print(f"Mean Episode Rew:   {avg_reward:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
