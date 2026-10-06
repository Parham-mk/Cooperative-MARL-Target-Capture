# Phase 13 completion report

Completed on 2026-10-06. The four-condition, five-seed study is complete, with no reduced scientific budget or unfinished runs. Phase 14 remains unimplemented.

The subsequent [capture-rate audit](phase13_capture_rate_audit.md) adds independent mechanics/capture checks, 40,000 fresh frozen evaluation episodes, and an untrained control. It found no leakage or success-accounting error and extends the complete test suite to 120 passing tests. The original study results below remain unchanged.

## Implementation

The runner reuses `train_cooperative_q()` and the Phase 12 `evaluate_with_recorder()` loop. Immutable `RewardConfig` instances construct the existing `RewardCalculator`; no reward formula is duplicated. The same frozen rollouts supply quantitative results and existing behavioral metrics. Every training call creates a fresh shared table and hunter wrappers, and every evaluation loads its checkpoint into new wrappers with reset action RNGs. Training never loads checkpoints.

The reward is `distance_weight * sum_i(d_i,previous - d_i,current) + capture_weight * captured + step_penalty`, with each distance measured against the target at that time. The capture bonus and time penalty occur once per environment transition. The same reward updates hunter 0 then hunter 1; episode reward accumulates once, in the same left-to-right order used by the existing helpers. Removing the capture bonus preserves capture termination and terminal bootstrap handling.

The environment, random target policy, collision/capture rules, agent-centric relative encoder, Q-learning mathematics, action selection, and termination versus truncation handling are unchanged. Omitted reward configuration preserves prior helper behavior and history format. Phase 13 requests additional complete-window statistics without changing default Phase 11 histories.

Created source/documentation files:

- [configs/ablation_config.py](../configs/ablation_config.py): immutable variants, protocol, seed schedules, isolated artifact paths
- [experiments/run_reward_ablation.py](../experiments/run_reward_ablation.py): fresh training, checkpoint loading, shared recorded evaluation, deterministic selection, metadata and manifests
- [analysis/reward_ablation_analysis.py](../analysis/reward_ablation_analysis.py): seed aggregation, conditional threshold summaries, aligned curve sources, Matplotlib plots, saved-data analysis
- [tests/test_reward_ablation.py](../tests/test_reward_ablation.py): 26 fast test cases, including parametrized variants
- this completion report

Modified files:

- [env/rewards.py](../env/rewards.py): frozen parameter container; existing calculation is unchanged
- [experiments/run_comparison.py](../experiments/run_comparison.py): optional reward, learning-parameter, path, and complete-window hooks in the cooperative trainer
- [experiments/evaluation_utils.py](../experiments/evaluation_utils.py): optional reward configuration
- [experiments/generate_behavioral_examples.py](../experiments/generate_behavioral_examples.py): optional reward configuration in recording
- [README.md](../README.md): actual Phase 13 results, protocol, reproduction, interpretation, limitations, updated phase status
- [.gitignore](../.gitignore): narrow exceptions for scientific evidence, with raw data and checkpoints still ignored

No existing Phase 11/12 results were edited.

## Exact protocol

| Variant | Distance weight | Capture weight | Step penalty |
|---|---:|---:|---:|
| full_reward | 1.0 | 20.0 | -0.05 |
| no_distance | 0.0 | 20.0 | -0.05 |
| no_step_penalty | 1.0 | 20.0 | 0.0 |
| no_capture_reward | 1.0 | 0.0 | -0.05 |

Shared-Policy Cooperative Q-Learning only; 10 × 10 grid; 100 steps; training seeds `[0, 1, 2, 3, 4]`; 5,000 training episodes and 500 frozen evaluation episodes per variant and seed. Learning rate 0.1, gamma 0.95, initial epsilon 1.0, per-episode decay 0.995, minimum epsilon 0.05.

At seed index `j`, training environment seeds are `j * 1,000,000 + episode`, and hunter action RNG seeds are `2 * training_seed` and the next integer. Evaluation environment seeds are `10,000,000 + j * 500 + episode`, and action RNG seeds are `20,000,000 + 2*j` and the next integer. All variants share these schedules and initializations. Evaluation uses epsilon zero and seeded random tie-breaking, without learning or insertion of unseen states.

The rolling window is 100 complete training episodes; threshold is capture rate ≥ 0.50. Stored episode indices are zero-based, and attainment reports `episode + 1` episodes completed. Early incomplete windows are NaN. Nonattainment is “Not reached”; mean threshold times are conditional on reached seeds.

## Results

All uncertainties below are sample standard deviations across five training-seed estimates, after within-seed aggregation. Capture times use successes only; all five seeds have valid capture-time estimates for every variant.

| Variant | Capture rate | Successful capture time | All-episode length | Variant-specific reward |
|---|---:|---:|---:|---:|
| full_reward | 1.0000 ± 0.0000 | 16.8412 ± 0.4470 | 16.8412 ± 0.4470 | 30.5079 ± 0.0755 |
| no_distance | 0.9372 ± 0.0099 | 32.4607 ± 1.1130 | 36.7068 ± 0.8946 | 16.9087 ± 0.2079 |
| no_step_penalty | 1.0000 ± 0.0000 | 16.9244 ± 0.5954 | 16.9244 ± 0.5954 | 31.3500 ± 0.0942 |
| no_capture_reward | 1.0000 ± 0.0000 | 15.6696 ± 0.4765 | 15.6696 ± 0.4765 | 10.5665 ± 0.1020 |

Each variant has 2,500 evaluation episodes. No distance has 2,343 successes and 157 failures; each other variant has 2,500 successes and no failures. Episode reward definitions differ and are not a success ranking.

| Variant | Reached seeds | Threshold episodes completed, conditional mean ± sample SD |
|---|---:|---:|
| full_reward | 5/5 | 254.6 ± 6.7 |
| no_distance | 5/5 | 921.6 ± 88.5 |
| no_step_penalty | 5/5 | 253.2 ± 11.4 |
| no_capture_reward | 5/5 | 260.2 ± 9.3 |

Distance shaping has **strong descriptive support** for faster threshold attainment and capture efficiency in this protocol: removing it increases threshold time about 3.62-fold and all-episode length by 19.8656 steps. It is not universally necessary: the ablated condition still captures in 93.72% of episodes.

The time penalty has **weak support** for an additional efficiency benefit at this weight: its removal changes mean duration by +0.0832 steps, with no observed capture-rate reduction. The explicit capture bonus has **weak support for necessity**, and the collapse hypothesis is contradicted by this run: removing it retains all successes and reduces mean duration by 1.1716 steps. Distance shaping, the time penalty, and capture termination might provide enough task structure without that bonus; this is a possible explanation, not an identified mechanism. No statistical significance or universal component necessity is claimed.

## Behavioral evidence

The existing Phase 12 metrics include the initial frame and all post-transition frames. Means and sample SDs are over five within-seed episode averages.

| Variant | Target distance | Hunter separation | Simultaneous adjacency | Distinct-side fraction |
|---|---:|---:|---:|---:|
| full_reward | 4.0687 ± 0.0408 | 4.4486 ± 0.1180 | 0.0780 ± 0.0027 | 0.0780 ± 0.0027 |
| no_distance | 4.9507 ± 0.0434 | 4.6409 ± 0.1104 | 0.0513 ± 0.0041 | 0.0513 ± 0.0041 |
| no_step_penalty | 4.1055 ± 0.0636 | 4.5128 ± 0.1121 | 0.0795 ± 0.0028 | 0.0795 ± 0.0028 |
| no_capture_reward | 4.1415 ± 0.0627 | 4.6172 ± 0.0797 | 0.0846 ± 0.0023 | 0.0846 ± 0.0023 |

No distance has greater average target distance and slower capture. No capture reward has faster capture despite slightly greater mean target distance than full reward. Collision constraints make the two adjacency indicators identical, and immediate capture termination couples them to success and duration. They are not independent proof of cooperation.

Successes are selected by proximity to median successful capture time, then training seed and evaluation seed. Failures use the first under that seed ordering. Five real trajectories are saved: one success per variant and one no-distance failure. All were replayed with exact actions, target actions, positions, reward values, and termination/truncation flags. The other conditions have no failures to select.

The selected no-distance failure is seed 0 / evaluation seed 10,000,065. It truncates at 100 steps, has mean target distance 7.4554 and mean hunter separation 10.4950, and ends with hunter distances 3 and 7. The paths show separated movement without simultaneous adjacency. It does not support a claim that the hunters remain close to the target but cannot complete capture. Selected examples do not support a reward-hacking or systematic oscillation claim; no prevalence of a subjective failure category is inferred from one example.

## Tests, commands, and artifact audit

The initial sandbox test attempts encountered Windows temporary-directory and Matplotlib strict-path-resolution permission failures. With writable temporary directories and execution outside that sandbox, a pristine `HEAD` snapshot passed **90 tests**, and the final implementation passed **116 tests**. Both have 13 existing dependency deprecation warnings. Tests cover exact variants, removed components, moving-target arithmetic, once-per-transition accounting, both learning updates, capture termination with zero bonus, all three reward hooks, frozen known/unseen states, restored epsilon, seed-level sample SD, missing capture times, complete-window thresholds and conditional counts, fresh tables/wrappers, isolated paths, default compatibility, tiny runs, identical repeated artifacts, and saved-data identity validation. They assert no required performance level.

Final suite, executed in PowerShell with temporary storage inside the repository:

```powershell
$env:TEMP = (Resolve-Path '.venv/phase13_temp').Path
$env:TMP = $env:TEMP
python -m pytest -q --basetemp .venv/pytest_phase13_final
```

Scientific and verified smoke commands executed:

```bash
python -m experiments.run_reward_ablation --grid-size 4 --max-steps 10 --train-episodes 10 --eval-episodes 5 --seeds 0 1 --output-dir results/ablations_smoke_verified
python -m experiments.run_reward_ablation --train-episodes 5000 --eval-episodes 500 --seeds 0 1 2 3 4 --output-dir results/ablations
python -m analysis.reward_ablation_analysis --output-dir results/ablations
```

The first smoke execution also completed in `results/ablations_smoke`; neither smoke directory contributes scientific results. Retraining requires a fresh directory, such as `results/ablations_reproduction`. The runner refuses to overwrite a saved study configuration. Analysis rebuilds from saved raw evaluation data, histories, and behavioral data, without retraining. Since raw data remain ignored, users of a fresh clone first reproduce the experiment in a new directory, then point the analysis command there.

The audit verified 20 checkpoint hashes, 100,000 complete training rows, 10,000 evaluation rows, all rolling windows and epsilon values, raw missing-value conventions, and all five selected trajectory replays. Frozen evaluation made 20 successful before/after deep table comparisons; the tests additionally cover empty tables and unseen states. Repeated tiny studies produced byte-identical numerical CSVs, checkpoints, and selected JSON trajectories. Independently retrained full-reward checkpoints and all 2,500 evaluation rows match saved Phase 11 results exactly. The full four-condition study was executed once; a second full independent ablation execution is not claimed.

All four required scientific plots, the zero-success diagnostic plot, and five trajectory plots were inspected visually. Labels, variant order, 0–1 capture-rate axes, uncertainty, successful-seed counts, and N/A handling are readable.

Artifacts live under [results/ablations](../results/ablations/): `raw/`, `raw/training/`, `checkpoints/<variant>/`, `summaries/`, `plots/`, `behavioral/`, and `trajectories/`. The [configuration](../results/ablations/summaries/ablation_config.json) records resolved parameters, reward definitions, executed command, seed schedules, runtime, and conventions. The [verification record](../results/ablations/summaries/verification.json), [checkpoint identities](../results/ablations/summaries/checkpoint_identities.csv), and [SHA-256 manifest](../results/ablations/summaries/artifact_manifest.json) record the checked inventory. Narrow Git exceptions expose summaries, figure sources, plots, and selected trajectories; raw data, per-episode behavioral tables, checkpoints, and smoke data remain ignored.

## Limitations and Phase 14 readiness

Five seeds, one fixed budget, one grid, one target policy, one shared tabular method, and one-component removals cannot resolve component interactions, optimize weights, or establish transfer to deep MARL or larger teams. First threshold crossing is not sustained attainment or proven convergence. Descriptive sample SD is not a confidence interval. Behavioral indicators are constrained by mechanics, and representative examples cannot establish failure-pattern prevalence.

Phase 13 is complete. Configuration routing, reward accounting, artifact isolation, frozen evaluation, and saved-data analysis are tested; scientific outputs and cautious interpretation are documented. The repository is ready for Phase 14 generalization/stress testing, which has not been implemented.
