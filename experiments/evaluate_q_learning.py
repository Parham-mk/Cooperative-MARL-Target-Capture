import argparse
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.target_capture_env import TargetCaptureEnv
from env.rewards import RewardCalculator
from agents.q_learning_agent import QLearningAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate Independent Q-Learning Agents")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    args = parser.parse_args()
    
    print(f"Evaluating Q-Learning Baseline ({args.episodes} episodes)")
    
    env = TargetCaptureEnv(grid_size=args.grid_size, max_steps=100)
    reward_calc = RewardCalculator()
    
    # Initialize agents and set epsilon to 0 (Exploitation only)
    agent0 = QLearningAgent(epsilon=0.0)
    agent1 = QLearningAgent(epsilon=0.0)
    
    try:
        agent0.load("results/checkpoints/agent_0_q_learning.pkl")
        agent1.load("results/checkpoints/agent_1_q_learning.pkl")
        print("Loaded models successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_q_learning.py first.")
        sys.exit(1)
        
    successes = 0
    total_steps = 0
    total_reward = 0.0
    
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            obs0 = {"agent_position": state["agent_0"], "target_position": state["target"]}
            obs1 = {"agent_position": state["agent_1"], "target_position": state["target"]}
            
            # Agents select best actions deterministically
            a0 = agent0.select_action(obs0)
            a1 = agent1.select_action(obs1)
            
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
        total_steps += info.get("step", env.current_step)
        total_reward += episode_reward
        
    capture_rate = successes / args.episodes
    avg_steps = total_steps / args.episodes
    avg_reward = total_reward / args.episodes
    
    print("-" * 40)
    print("Independent Q-Learning Agent Evaluation:")
    print(f"Capture Rate: {capture_rate:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
