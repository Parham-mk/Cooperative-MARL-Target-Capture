import argparse
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.target_capture_env import TargetCaptureEnv
from env.rewards import RewardCalculator
from agents.q_learning_agent import QLearningAgent

def main():
    parser = argparse.ArgumentParser(description="Train Independent Q-Learning Agents")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    args = parser.parse_args()
    
    print(f"Training Q-Learning Baseline ({args.episodes} episodes)")
    
    env = TargetCaptureEnv(grid_size=args.grid_size, max_steps=100)
    reward_calc = RewardCalculator()
    
    agent0 = QLearningAgent(learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05)
    agent1 = QLearningAgent(learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05)
    
    rewards_history = []
    lengths_history = []
    capture_history = []
    
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            obs0 = {"agent_position": state["agent_0"], "target_position": state["target"]}
            obs1 = {"agent_position": state["agent_1"], "target_position": state["target"]}
            
            a0 = agent0.select_action(obs0)
            a1 = agent1.select_action(obs1)
            
            next_state, info = env.step({"agent_0": a0, "agent_1": a1})
            
            rewards = reward_calc.calculate(
                agents=[env.agent_0, env.agent_1],
                target=env.target,
                previous_positions=state,
                captured=info.get("captured", False)
            )
            step_reward = rewards["total_reward"]
            episode_reward += step_reward
            
            next_obs0 = {"agent_position": next_state["agent_0"], "target_position": next_state["target"]}
            next_obs1 = {"agent_position": next_state["agent_1"], "target_position": next_state["target"]}
            
            agent0.update(obs0, a0, step_reward, next_obs0)
            agent1.update(obs1, a1, step_reward, next_obs1)
            
            state = next_state
            
            if info.get("terminated", False) or info.get("truncated", False):
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

    # Save Checkpoints
    os.makedirs("results/checkpoints", exist_ok=True)
    agent0.save("results/checkpoints/agent_0_q_learning.pkl")
    agent1.save("results/checkpoints/agent_1_q_learning.pkl")
    print("Models saved to results/checkpoints/")

    # Plots
    os.makedirs("results/plots", exist_ok=True)
    
    # Smooth data for plotting
    window = 50
    def moving_avg(data):
        return [sum(data[max(0, i-window):i+1]) / len(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    plt.figure()
    plt.plot(moving_avg(rewards_history))
    plt.title("Q-Learning: Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Moving Avg)")
    plt.savefig("results/plots/q_learning_reward.png")
    
    plt.figure()
    plt.plot(moving_avg(capture_history))
    plt.title("Q-Learning: Capture Rate")
    plt.xlabel("Episode")
    plt.ylabel("Capture Rate (Moving Avg)")
    plt.savefig("results/plots/q_learning_capture_rate.png")
    
    print("Plots saved to results/plots/")

if __name__ == "__main__":
    main()
