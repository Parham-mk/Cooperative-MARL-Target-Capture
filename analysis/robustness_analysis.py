import os
import csv
import json
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt

def main():
    print("--- Starting Phase 14 Robustness Analysis ---")
    
    os.makedirs("results/robustness/summaries", exist_ok=True)
    os.makedirs("results/robustness/plots", exist_ok=True)
    
    raw_results = []
    with open("results/robustness/raw/robustness_raw.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["captured"] = int(row["captured"])
            row["episode_length"] = float(row["episode_length"])
            row["episode_reward"] = float(row["episode_reward"])
            row["q_state_coverage"] = float(row["q_state_coverage"])
            if row["capture_time"]:
                row["capture_time"] = float(row["capture_time"])
            raw_results.append(row)
            
    # Aggregate summaries by method and condition
    # But wait, we first need to aggregate by seed, then across seeds, to match mean ± std across 5 independent seeds.
    # Group by (method, condition, seed)
    seed_stats = defaultdict(lambda: {"captured": 0, "length": 0.0, "coverage": 0.0, "time": 0.0, "time_count": 0, "count": 0})
    for r in raw_results:
        key = (r["method"], r["condition"], r["training_seed"])
        st = seed_stats[key]
        st["captured"] += r["captured"]
        st["length"] += r["episode_length"]
        st["coverage"] += r["q_state_coverage"]
        if r["capture_time"] is not None and r["captured"] == 1:
            st["time"] += r["capture_time"]
            st["time_count"] += 1
        st["count"] += 1
        
    # Summarize per seed
    per_seed = []
    for (method, condition, seed), st in seed_stats.items():
        c = st["count"]
        cap_rate = st["captured"] / c
        per_seed.append({
            "method": method,
            "condition": condition,
            "training_seed": seed,
            "capture_rate": cap_rate,
            "mean_length": st["length"] / c,
            "mean_coverage": st["coverage"] / c,
            "mean_time": st["time"] / st["time_count"] if st["time_count"] > 0 else float('nan')
        })
        
    # Aggregate across seeds
    final_stats = defaultdict(lambda: {"cap_rates": [], "lengths": [], "coverages": [], "times": []})
    for ps in per_seed:
        key = (ps["method"], ps["condition"])
        st = final_stats[key]
        st["cap_rates"].append(ps["capture_rate"])
        st["lengths"].append(ps["mean_length"])
        st["coverages"].append(ps["mean_coverage"])
        if not sum(1 for _ in [ps["mean_time"]] if str(ps["mean_time"]) == "nan"):
            st["times"].append(ps["mean_time"])
            
    summary_rows = []
    for (method, cond), st in final_stats.items():
        cr_mean = statistics.mean(st["cap_rates"])
        cr_std = statistics.stdev(st["cap_rates"]) if len(st["cap_rates"]) > 1 else 0.0
        
        len_mean = statistics.mean(st["lengths"])
        len_std = statistics.stdev(st["lengths"]) if len(st["lengths"]) > 1 else 0.0
        
        cov_mean = statistics.mean(st["coverages"])
        cov_std = statistics.stdev(st["coverages"]) if len(st["coverages"]) > 1 else 0.0
        
        if st["times"]:
            time_mean = statistics.mean(st["times"])
            time_std = statistics.stdev(st["times"]) if len(st["times"]) > 1 else 0.0
        else:
            time_mean = float('nan')
            time_std = 0.0
            
        summary_rows.append({
            "method": method,
            "condition": cond,
            "capture_rate_mean": cr_mean,
            "capture_rate_std": cr_std,
            "capture_time_mean": time_mean,
            "capture_time_std": time_std,
            "episode_length_mean": len_mean,
            "episode_length_std": len_std,
            "q_coverage_mean": cov_mean,
            "q_coverage_std": cov_std,
            "unseen_state_rate_mean": 1.0 - cov_mean
        })
        
    with open("results/robustness/summaries/robustness_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
        
    # Split primary conditions from stress cases
    primary_conds = ["In Distribution", "Unseen Seeds", "Unseen Initial States"]
    
    # Plot 1: Capture Rate Robustness
    plt.figure(figsize=(10, 6))
    x = range(len(primary_conds))
    width = 0.35
    
    ind_cr = [next((r["capture_rate_mean"] for r in summary_rows if r["method"] == "Independent Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    ind_std = [next((r["capture_rate_std"] for r in summary_rows if r["method"] == "Independent Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    
    coop_cr = [next((r["capture_rate_mean"] for r in summary_rows if r["method"] == "Cooperative Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    coop_std = [next((r["capture_rate_std"] for r in summary_rows if r["method"] == "Cooperative Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    
    plt.bar([i - width/2 for i in x], ind_cr, width, yerr=ind_std, label="Independent Q-Learning", capsize=5, color="blue", alpha=0.7)
    plt.bar([i + width/2 for i in x], coop_cr, width, yerr=coop_std, label="Cooperative Q-Learning", capsize=5, color="orange", alpha=0.7)
    
    plt.ylabel("Capture Rate")
    plt.title("Capture Rate Robustness across Held-Out Conditions")
    plt.xticks(x, primary_conds)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig("results/robustness/plots/capture_rate_robustness.png", dpi=150)
    plt.close()
    
    # Plot 2: Performance Drop (Delta)
    plt.figure(figsize=(10, 6))
    ind_base = ind_cr[0]
    coop_base = coop_cr[0]
    
    ind_drop = [cr - ind_base for cr in ind_cr[1:]]
    coop_drop = [cr - coop_base for cr in coop_cr[1:]]
    drop_conds = primary_conds[1:]
    
    x_drop = range(len(drop_conds))
    plt.bar([i - width/2 for i in x_drop], ind_drop, width, label="Independent Q-Learning", color="blue", alpha=0.7)
    plt.bar([i + width/2 for i in x_drop], coop_drop, width, label="Cooperative Q-Learning", color="orange", alpha=0.7)
    plt.ylabel("Absolute Drop in Capture Rate")
    plt.title("Performance Degradation vs In-Distribution Evaluation")
    plt.xticks(x_drop, drop_conds)
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.savefig("results/robustness/plots/performance_drop.png", dpi=150)
    plt.close()
    
    # Plot 3: Q-Table Coverage
    plt.figure(figsize=(10, 6))
    ind_cov = [next((r["q_coverage_mean"] for r in summary_rows if r["method"] == "Independent Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    coop_cov = [next((r["q_coverage_mean"] for r in summary_rows if r["method"] == "Cooperative Q-Learning" and r["condition"] == c), 0) for c in primary_conds]
    
    plt.bar([i - width/2 for i in x], ind_cov, width, label="Independent Q-Learning", color="blue", alpha=0.7)
    plt.bar([i + width/2 for i in x], coop_cov, width, label="Cooperative Q-Learning", color="orange", alpha=0.7)
    
    plt.ylabel("Q-Table Coverage (Visited Evaluation States present in Q-Table)")
    plt.title("State Representation Coverage under Distribution Shift")
    plt.xticks(x, primary_conds)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig("results/robustness/plots/q_table_coverage.png", dpi=150)
    plt.close()
    
    # Save a configuration record for reproducibility
    config_record = {
        "conditions": primary_conds,
        "grid_size_transfer_included": False,
        "note": "Grid-size transfer was excluded because tabular Q-learning does not provide reliable values for unseen relative states."
    }
    with open("results/robustness/summaries/robustness_config.json", "w") as f:
        json.dump(config_record, f, indent=4)
        
    print("Phase 14 Analysis Complete. Plots and Summaries saved.")

if __name__ == "__main__":
    main()
