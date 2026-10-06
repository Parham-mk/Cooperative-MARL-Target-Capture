import os
import sys
import csv
import json
import random
from collections import defaultdict
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.evaluation_config as eval_cfg
import configs.robustness_config as rob_cfg
from env.target_capture_env import TargetCaptureEnv
from env.position import Position
from experiments.generate_behavioral_examples import load_independent_q, load_cooperative_q
from analysis.trajectory_analysis import TrajectoryRecorder

def get_q_coverage(agent, obs) -> float:
    """Returns 1.0 if state is in Q-table, 0.0 otherwise."""
    state_key = agent._get_state_key(obs)
    # q_table could be a dict (Independent) or SharedQTable (Cooperative)
    if hasattr(agent.q_table, "get_q_values"):
        # SharedQTable
        return 1.0 if state_key in agent.q_table.q_table else 0.0
    else:
        # dict
        return 1.0 if state_key in agent.q_table else 0.0

def evaluate_robustness_episode(
    env: TargetCaptureEnv,
    agent0,
    agent1,
    initial_state_kwargs: dict = None,
    seed: int = None
) -> dict:
    """Evaluates a single episode with epsilon=0 and returns metrics."""
    
    # Ensure epsilon=0
    agent0.epsilon = 0.0
    agent1.epsilon = 0.0
    
    if initial_state_kwargs is not None:
        state = env.reset(seed=seed)
        env.agent_0.position = initial_state_kwargs["agent_0_pos"]
        env.agent_1.position = initial_state_kwargs["agent_1_pos"]
        env.target.position = initial_state_kwargs["target_pos"]
        state = env.get_state()
    else:
        state = env.reset(seed=seed)
        
    done = False
    
    # Store start positions
    start_a0 = state["agent_0"]
    start_a1 = state["agent_1"]
    start_tgt = state["target"]
    
    total_reward = 0.0
    steps = 0
    
    seen_states = 0
    total_states = 0
    
    while not done:
        obs0 = {
            "agent_position": state["agent_0"], 
            "target_position": state["target"],
            "agent_0": state["agent_0"], 
            "agent_1": state["agent_1"], 
            "target": state["target"]
        }
        obs1 = {
            "agent_position": state["agent_1"], 
            "target_position": state["target"],
            "agent_0": state["agent_0"], 
            "agent_1": state["agent_1"], 
            "target": state["target"]
        }
        
        seen_states += get_q_coverage(agent0, obs0)
        seen_states += get_q_coverage(agent1, obs1)
        total_states += 2
        
        a0 = agent0.select_action(obs0)
        a1 = agent1.select_action(obs1)
        
        next_state, info = env.step({"agent_0": a0, "agent_1": a1})
        
        # Reward calculator not strictly necessary for episode sum unless we do full calc,
        # but env.step doesn't return full reward unless we use RewardCalculator.
        # Actually, wait, env.step only gives step metadata. Let's just use info['captured'] for capture rate.
        # Or we can compute reward? We don't strictly need reward for the main robustness metrics, 
        # but the prompt asks for "Mean Episode Reward".
        # Let's instantiate a RewardCalculator.
        from env.rewards import RewardCalculator
        rc = RewardCalculator()
        rew = rc.calculate(
            agents=[env.agent_0, env.agent_1],
            target=env.target,
            previous_positions=state,
            captured=info.get("captured", False)
        )["total_reward"]
        
        total_reward += rew
        steps += 1
        state = next_state
        
        if info.get("terminated", False) or info.get("truncated", False):
            done = True
            
    coverage = seen_states / total_states if total_states > 0 else 1.0
    
    return {
        "captured": int(info.get("captured", False)),
        "episode_length": steps,
        "capture_time": steps if info.get("captured", False) else None,
        "episode_reward": total_reward,
        "q_state_coverage": coverage,
        "initial_agent_0_x": start_a0.x,
        "initial_agent_0_y": start_a0.y,
        "initial_agent_1_x": start_a1.x,
        "initial_agent_1_y": start_a1.y,
        "initial_target_x": start_tgt.x,
        "initial_target_y": start_tgt.y,
    }

def generate_unseen_initial_states(num_cases: int, seed: int, grid_size: int, exclude_seeds: list) -> list:
    """Generates N valid combinations of initial states not explicitly aligned with standard seeds."""
    rng = random.Random(seed)
    # We will generate completely random valid configurations.
    # To be extremely thorough, we just randomly generate distinct positions.
    cases = []
    
    while len(cases) < num_cases:
        p1 = (rng.randint(0, grid_size-1), rng.randint(0, grid_size-1))
        p2 = (rng.randint(0, grid_size-1), rng.randint(0, grid_size-1))
        pt = (rng.randint(0, grid_size-1), rng.randint(0, grid_size-1))
        
        if p1 != p2 and p1 != pt and p2 != pt:
            cases.append({
                "agent_0_pos": Position(p1[0], p1[1]),
                "agent_1_pos": Position(p2[0], p2[1]),
                "target_pos": Position(pt[0], pt[1])
            })
    return cases

def run_evaluation(method_name: str, condition_name: str, agent0, agent1, config_kwargs: list) -> list:
    """Runs a batch of episodes based on config_kwargs list."""
    env = TargetCaptureEnv(grid_size=rob_cfg.GRID_SIZE, max_steps=rob_cfg.MAX_STEPS)
    results = []
    
    for idx, kwargs in enumerate(config_kwargs):
        res = evaluate_robustness_episode(env, agent0, agent1, **kwargs)
        res["method"] = method_name
        res["condition"] = condition_name
        res["episode"] = idx
        results.append(res)
    return results

def main():
    print("--- Starting Phase 14 Robustness Evaluation ---")
    
    os.makedirs("results/robustness/raw", exist_ok=True)
    os.makedirs("results/robustness/summaries", exist_ok=True)
    os.makedirs("results/robustness/plots", exist_ok=True)
    
    methods = [
        ("Independent Q-Learning", load_independent_q),
        ("Cooperative Q-Learning", load_cooperative_q)
    ]
    
    all_raw_results = []
    
    for idx, t_seed in enumerate(rob_cfg.TRAINING_SEEDS):
        print(f"Loading agents for training seed {t_seed}...")
        
        for method_name, loader in methods:
            config = eval_cfg.ExperimentConfig()
            a0, a1, _ = loader(config, t_seed, idx)
            
            # Condition A: In Distribution (use evaluation config's exact seed method)
            # Actually, standard evaluation uses: eval_seed = 1000 + training_seed * 100 + ep (from phase 12)
            # Phase 11 used ExperimentConfig evaluation_seed logic: 
            # evaluation_seed = self.evaluation_seed_base + seed_index * self.eval_episodes + episode
            print(f"  [{method_name}] Running Condition A: In Distribution")
            cond_a_kwargs = []
            ec = eval_cfg.ExperimentConfig()
            for ep in range(rob_cfg.NUM_EVAL_EPISODES):
                e_seed = ec.evaluation_seed(t_seed, ep)
                cond_a_kwargs.append({"seed": e_seed, "initial_state_kwargs": None})
            
            results_a = run_evaluation(method_name, "In Distribution", a0, a1, cond_a_kwargs)
            for r in results_a:
                r["training_seed"] = t_seed
            all_raw_results.extend(results_a)
            
            # Condition B: Held-Out Seeds
            print(f"  [{method_name}] Running Condition B: Unseen Seeds")
            cond_b_kwargs = []
            # We use a base robustness seed associated with the training seed to ensure separation.
            r_seed_base = 50_000_000 + t_seed * 1000
            for ep in range(rob_cfg.NUM_EVAL_EPISODES):
                cond_b_kwargs.append({"seed": r_seed_base + ep, "initial_state_kwargs": None})
            
            results_b = run_evaluation(method_name, "Unseen Seeds", a0, a1, cond_b_kwargs)
            for r in results_b:
                r["training_seed"] = t_seed
            all_raw_results.extend(results_b)
            
            # Condition C: Unseen Initial States
            print(f"  [{method_name}] Running Condition C: Unseen Initial States")
            # Generate deterministic unseen states using a seed specific to this configuration
            c_states = generate_unseen_initial_states(
                rob_cfg.NUM_INITIAL_STATE_CASES, 
                seed=999_999 + t_seed, 
                grid_size=rob_cfg.GRID_SIZE, 
                exclude_seeds=[]
            )
            cond_c_kwargs = []
            for st in c_states:
                cond_c_kwargs.append({"seed": None, "initial_state_kwargs": st})
            
            results_c = run_evaluation(method_name, "Unseen Initial States", a0, a1, cond_c_kwargs)
            for r in results_c:
                r["training_seed"] = t_seed
            all_raw_results.extend(results_c)
            
            # Condition D: Stress Cases
            print(f"  [{method_name}] Running Condition D: Stress Cases")
            cond_d_kwargs = []
            for sc in rob_cfg.STRESS_CASES:
                st = {
                    "agent_0_pos": Position(sc.agent_0[0], sc.agent_0[1]),
                    "agent_1_pos": Position(sc.agent_1[0], sc.agent_1[1]),
                    "target_pos": Position(sc.target[0], sc.target[1])
                }
                cond_d_kwargs.append({"seed": None, "initial_state_kwargs": st})
                
            results_d = run_evaluation(method_name, "Stress Cases", a0, a1, cond_d_kwargs)
            # Inject stress case names instead of episode index
            for idx, r in enumerate(results_d):
                r["training_seed"] = t_seed
                r["condition"] = f"Stress: {rob_cfg.STRESS_CASES[idx].name}"
            all_raw_results.extend(results_d)

    # Save Raw Results
    with open("results/robustness/raw/robustness_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_raw_results[0].keys())
        writer.writeheader()
        writer.writerows(all_raw_results)
        
    print("Phase 14 Generation Complete. Run analysis script next.")

if __name__ == "__main__":
    main()
