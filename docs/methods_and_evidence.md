# Methods and evidence

## Environment and learning protocol

The physical state is the ordered positions of two hunters and one target. The default grid is 10×10, with distinct positions at reset and a 100-transition horizon. A seeded NumPy generator samples three cells without replacement. Hunter actions are selected from the same pre-step state and applied in hunter-0, hunter-1 order; the target moves afterward. An out-of-bounds or occupied-cell move leaves the entity in place, so hunters cannot overlap, pass through one another or swap occupied cells in one transition. This sequential collision resolution can favor the first hunter.

The target samples uniformly from boundary-valid actions, including STAY, using the existing seeded `RandomTargetPolicy`. Its sampled move can subsequently be blocked by an occupied cell. Capture is checked after the full transition: each hunter must have Manhattan distance exactly one from the target. Capture terminates; otherwise the horizon truncates. At the last step both flags can be true if capture occurs there. Reset adjacency alone does not terminate an episode.

Random hunters uniformly sample all five actions, including moves that may be blocked. The Heuristic closes horizontal distance first and then vertical distance; it does not anticipate target motion, collisions or teammate behavior. It can attempt to enter the occupied target cell and remain blocked.

Independent Q-learning uses separate dictionaries, each keyed by `(hunter_x, hunter_y, target_x, target_y)`. The teammate is omitted. Cooperative Q-learning uses two wrappers around one shared sparse table, keyed by `(target_x - hunter_x, target_y - hunter_y, teammate_x - hunter_x, teammate_y - hunter_y)`. Relative states retain teammate relationships but omit absolute boundary location and can alias physical configurations. Neither encoded state should be described as a demonstrated fully observed Markov state.

Both learned methods use learning rate 0.1, discount 0.95, initial epsilon 1.0, decay 0.995 after each episode and a 0.05 floor. Both actions are chosen before the environment transition. Independent updates affect separate tables; shared updates occur for hunter 0 and then hunter 1, so the latter observes any preceding shared-table update. Each update receives the same team reward. Capture removes bootstrap; truncation retains it. All checkpoint saves contain Q-values only; evaluation creates fresh wrappers with seeded action RNGs.

The team reward uses summed Manhattan distance `D_t` to the target **at time t**:

```text
r_t = w_d * (D_t - D_(t+1)) + w_c * I(capture at t+1) + step_penalty
default: w_d=1, w_c=20, step_penalty=-0.05
```

It is computed and added to episode return once per transition, then supplied to both updates. This difference shaping is not claimed to be policy-invariant discounted potential shaping.

## Main study and reproducibility

Five training seeds, ordered 0–4, each train both learned methods for 5,000 episodes and evaluate every method for 500 episodes. Environment seeds for training are `seed_index * 1,000,000 + episode`; action RNG seeds are `2*training_seed` and `2*training_seed+1`. Evaluation environment seeds are `10,000,000 + seed_index*500 + episode`; action RNG seeds start at `20,000,000 + 2*seed_index` and the next integer. Target RNGs reset per episode; hunter RNG streams continue within each evaluation batch. Greedy evaluation has epsilon zero but retains seeded random tie-breaking; unseen states have zero action values and do not enter the table.

Raw episode estimates are averaged within training seed, then the five seed estimates are averaged with sample standard deviation. Baselines use the corresponding five evaluation partitions and action seeds. Successful capture time excludes failed episodes; all-episode length includes the horizon for failures. A zero-success seed has missing capture time and is omitted from that conditional aggregate. Valid-seed counts are stored. SD is descriptive variability, not a confidence interval or significance test. The original main helper reports SD zero for a single available seed; ablation and robustness report N/A for fewer than two. The full studies have five valid seeds.

Saved [main summaries](../results/reproducibility/summaries/comparison_summary.csv), [seed estimates](../results/reproducibility/summaries/per_seed_summary.csv), [configuration](../results/reproducibility/summaries/experiment_config.json) and [curve source](../results/reproducibility/summaries/training_curve_source.csv) support the README. Raw capture counts were Random 165/2,500, Heuristic 909/2,500, Independent 2,498/2,500 and Cooperative 2,500/2,500. Cooperative all-episode length was 3.4360 steps shorter than Independent, about 16.9% under this protocol. Representation and sharing change together; this is not an isolated effect of cooperation.

Two original full-budget main runs (`results/validation/run_a` and `run_b`, ignored local outputs) produced **35 byte-identical artifacts** covering raw CSVs, training histories, checkpoints, comparison summaries, curve-source data and behavioral metrics/summaries. The [verification record](../results/reproducibility/reproduction_verification.json) defines the exact scope. Original inventories, summaries and behavioral agreement were revalidated during finalization. The [selected evidence manifest](../results/reproducibility/artifact_manifest.json) covers tracked evidence rather than the full ignored raw study. Figures/GIFs are not part of the 35-artifact byte comparison.

## Behavioral evidence

Trajectory metrics include the initial frame and every post-transition frame, including capture. Each episode's metric is averaged within seed and across seeds. Success examples are closest to the method's median successful duration; ties use training seed then evaluation seed. Failure examples use the first failure in the same order. The [selection table](../results/reproducibility/behavioral/representative_selection.csv) includes the absence of observed Cooperative failures. Every animation and path plot comes from an actual frozen rollout, rather than a scripted demonstration.

The [behavioral summary](../results/reproducibility/behavioral/behavioral_summary.csv) gives these mean ± sample SD values:

| Method | Mean hunter-target distance | Mean hunter separation | Simultaneous adjacency fraction |
|---|---:|---:|---:|
| Random | 6.5217 ± 0.0262 | 6.6673 ± 0.0430 | 0.0038 ± 0.0003 |
| Heuristic | 2.6032 ± 0.0314 | 2.6826 ± 0.1108 | 0.0501 ± 0.0025 |
| Independent Q-Learning | 3.5364 ± 0.0316 | 4.0453 ± 0.1273 | 0.0700 ± 0.0033 |
| Cooperative Q-Learning | 4.0687 ± 0.0408 | 4.4486 ± 0.1180 | 0.0780 ± 0.0027 |

The Heuristic stays closer on average despite capturing less often. Proximity alone is not a coordination metric. Nonoverlap makes the two adjacent hunters occupy distinct neighboring cells, so the “distinct-side” and simultaneous-adjacency indicators are mechanically redundant here. Immediate capture termination couples their fractions to success frequency and duration. These metrics do not establish stable roles, communication or a causal cooperation mechanism.

## Reward study and audit

Four shared-table variants use identical training budgets, seeds, hyperparameters, mechanics and evaluation schedules, changing one reward component at a time. Full reward weights are `(1,20,-0.05)`; no distance `(0,20,-0.05)`; no step penalty `(1,20,0)`; no capture reward `(1,0,-0.05)`. Tables start empty for every variant and seed. Episode returns across these different definitions cannot rank capture success. Full reward matched the original main Cooperative checkpoints and evaluation.

The predefined learning threshold is the first **complete 100-episode window** with capture rate at least 0.5. Earlier partial windows are NaN. Crossing times count episodes completed, so zero-based index 99 means 100 episodes. All five seeds attained the threshold in every variant: full reward 254.6±6.7 episodes, no distance 921.6±88.5, no step penalty 253.2±11.4, no capture reward 260.2±9.3. This is an exploratory-training threshold, not a sustained-convergence guarantee.

The preserved [Phase 13 report](phase13_reward_ablation.md) and [capture-rate audit](phase13_capture_rate_audit.md) contain full tables, failures and checks. A separate evaluator reproduced all 10,000 original ablation rows using independent geometry, movement, target sampling and reward checks. It evaluated each variant on 10,000 fresh episodes: the three shaping variants each captured 10,000/10,000, while no distance captured 9,291/10,000. An untrained shared-table control captured 171/2,500 (0.0684). Tables/checkpoint files stayed fixed and training/evaluation environment seeds were disjoint.

Perfect observed capture is plausible in this small task. With distinct positions each hunter is at least one step from the target; summed distance is at least two, and equals two precisely at capture. Thus distance shaping retains direct task structure without an explicit bonus. Without a step penalty the positive capture reward is still discounted by gamma 0.95. These are plausible explanations, not mechanisms identified experimentally. No necessity of the capture bonus, statistical significance, reward hacking or universal success is claimed.

## Scope and evidence access

This is an interpretable tabular benchmark with two hunters, one random target, one grid and a fixed budget. It evaluates efficient task completion and simple behavioral indicators. It is not a deep MARL benchmark, hyperparameter search, publication or proof of novel emergent cooperation. Same-grid robustness is documented in [the robustness report](robustness_evaluation.md); grid-size transfer is excluded. Future controlled comparisons could separate representation from sharing, and later studies could test larger grids, target strategies or partial observability. Those experiments are not implemented here.

Raw data/checkpoints are retained locally but ignored; selected evidence is tracked. Use [the reproduction guide](reproducibility.md) to regenerate inputs before running analyses. Historical manifests with ignored-file entries preserve provenance and do not imply those files ship in a clone.
