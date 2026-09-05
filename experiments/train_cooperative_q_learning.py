import argparse
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.target_capture_env import TargetCaptureEnv
from env.rewards import RewardCalculator
from algorithms.cooperative_q_learning import SharedQTable
from agents.shared_q_agent import SharedQAgent

def main():
    parser = argparse.ArgumentParser(description="Train Shared-Policy Cooperative Q-Learning")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    args = parser.parse_args()
    
    print(f"Training Cooperative Q-Learning Baseline ({args.episodes} episodes)")
    
    env = TargetCaptureEnv(grid_size=args.grid_size, max_steps=100)
    reward_calc = RewardCalculator()
    
    # 1. ONE shared Q-table
    shared_table = SharedQTable(learning_rate=0.1, gamma=0.95)
    
    # 2. Both agents use the same table but distinct IDs
    agent0 = SharedQAgent(
        shared_q_table=shared_table, 
        agent_id="agent_0", 
        teammate_id="agent_1",
        epsilon=1.0, 
        epsilon_decay=0.995, 
        min_epsilon=0.05
    )
    agent1 = SharedQAgent(
        shared_q_table=shared_table, 
        agent_id="agent_1", 
        teammate_id="agent_0",
        epsilon=1.0, 
        epsilon_decay=0.995, 
        min_epsilon=0.05
    )
    
    rewards_history = []
    lengths_history = []
    capture_history = []
    
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # We pass the full state dict; agents will extract what they need
            a0 = agent0.select_action(state)
            a1 = agent1.select_action(state)
            
            # Submit joint action
            next_state, info = env.step({"agent_0": a0, "agent_1": a1})
            
            # Calculate shared team reward
            rewards = reward_calc.calculate(
                agents=[env.agent_0, env.agent_1],
                target=env.target,
                previous_positions=state,
                captured=info.get("captured", False)
            )
            step_reward = rewards["total_reward"]
            episode_reward += step_reward
            
            is_done = info.get("terminated", False) or info.get("truncated", False)
            
            # Update shared Q-table from both transitions
            agent0.update(state, a0, step_reward, next_state, info.get("terminated", False))
            agent1.update(state, a1, step_reward, next_state, info.get("terminated", False))
            
            state = next_state
            if is_done:
                done = True
                
        agent0.decay_epsilon()
        agent1.decay_epsilon()
        
        rewards_history.append(episode_reward)
        lengths_history.append(info.get("step", env.current_step))
        capture_history.append(1 if info.get("captured", False) else 0)
        
        if (ep + 1) % 100 == 0:
            avg_rew = sum(rewards_history[-100:]) / 100
            cap_rate = sum(capture_history[-100:]) / 100
            print(f"Episode {ep+1}: Avg Reward: {avg_rew:.2f}, Capture Rate: {cap_rate:.2f}, Epsilon: {agent0.epsilon:.2f}")

    # Save Checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    shared_table.save("results/checkpoints/cooperative_q_learning.pkl")
    print("Model saved to results/checkpoints/cooperative_q_learning.pkl")

    # Plots
    os.makedirs("results/plots", exist_ok=True)
    
    window = 50
    def moving_avg(data):
        return [sum(data[max(0, i-window):i+1]) / len(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    plt.figure()
    plt.plot(moving_avg(rewards_history))
    plt.title("Cooperative Q-Learning: Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Moving Avg)")
    plt.savefig("results/plots/cooperative_q_reward.png")
    
    plt.figure()
    plt.plot(moving_avg(capture_history))
    plt.title("Cooperative Q-Learning: Capture Rate")
    plt.xlabel("Episode")
    plt.ylabel("Capture Rate (Moving Avg)")
    plt.savefig("results/plots/cooperative_q_capture_rate.png")
    
    print("Plots saved to results/plots/")

if __name__ == "__main__":
    main()
