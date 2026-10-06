# Phase 13 capture-rate and leakage audit

The perfect observed capture rates were investigated on 2026-10-06. No training/evaluation leakage, checkpoint sharing across conditions, fabricated success, or evaluation learning was found in the audited implementation and artifacts. Original study data and checkpoints remain unchanged.

## Independent checks

The audit uses [experiments/audit_reward_ablation.py](../experiments/audit_reward_ablation.py), a separate evaluator with reference calculations for movement, collision handling, random-target sampling, capture geometry, reward arithmetic, termination, and truncation. It does not trust the original evaluator's capture flags. It checks both hunters' Manhattan distances directly and verifies that all three entities occupy distinct, valid cells. Target choices are reproduced using a separate RNG and independently constructed valid-action list. A learning call raises an error, and Q-table values and membership are compared before and after each run.

Checks completed:

- The 25,000 training environment seeds and 2,500 original evaluation environment seeds have zero overlap.
- All 20 checkpoints have unique paths and distinct hashes, matching their saved identities. Source inspection confirms every training call constructs an empty table and never loads a checkpoint.
- Agents choose actions from current positions only. Their state encoder does not receive the environment seed, future target action, next state, reward, or capture flag.
- The independent evaluator exactly reproduced all 10,000 original evaluation rows, including capture outcomes, episode lengths, capture times, and rewards.
- An additional 2,000 evaluation episodes per checkpoint used fresh environment seeds 30,000,000–30,009,999 and fresh hunter RNG initializations 40,000,000 + 2 × seed index and the next integer. These environment seeds overlap neither training nor original evaluation seeds.
- Movement, target sampling, geometry, reward, and episode flags agreed for 1,099,146 transitions across original and fresh learned-policy rollouts.
- The 20 checkpoint files remained byte-identical after the audit. All evaluation table snapshots remained unchanged, including unseen-state membership.
- An untrained shared-table control was evaluated separately in 500 episodes per seed, using environment seeds 30,000,000–30,002,499 and the same action-RNG initialization convention. This control is a diagnostic for success accounting, not another trained reward variant or a paired per-seed comparison with the 2,000-episode blocks.

The [audit record](../results/ablations_audit/audit.json) and [fresh seed summaries](../results/ablations_audit/fresh_seed_summary.csv) preserve the results. Raw audit rows are stored locally under `results/ablations_audit/raw/` and remain ignored.

## Fresh evaluation results

Every trained variant uses the same five saved models and 10,000 additional episodes in total. The models were not retrained or tuned. Uncertainty is sample SD across five within-seed estimates.

| Condition | Original successes | Fresh successes | Fresh capture rate | Fresh mean episode length | Longest fresh episode |
|---|---:|---:|---:|---:|---:|
| Full reward | 2,500/2,500 | 10,000/10,000 | 1.0000 ± 0.0000 | 17.2942 ± 0.3458 | 89 |
| No distance | 2,343/2,500 | 9,291/10,000 | 0.9291 ± 0.0072 | 38.0196 ± 1.1696 | 100 |
| No step penalty | 2,500/2,500 | 10,000/10,000 | 1.0000 ± 0.0000 | 17.2688 ± 0.3490 | 98 |
| No capture reward | 2,500/2,500 | 10,000/10,000 | 1.0000 ± 0.0000 | 15.7965 ± 0.2027 | 84 |
| Untrained shared table (control) | — | 171/2,500 | 0.0684 ± 0.0065 | 96.3012 ± 0.2356 | 100 |

The control fails in 2,329 episodes, and no distance fails in 709 fresh episodes. The evaluator therefore demonstrably records failures. Only eight initial layouts in each trained variant's 10,000-episode fresh sample already have simultaneous adjacency; the high capture rates cannot be explained by trivial initial placements. The no-step-penalty condition includes a successful 98-step episode, so the result is not a constant short-duration success artifact.

## Why these reward removals can still solve the task

Let `D` be the sum of the two hunter-to-target Manhattan distances. Collision rules prevent a hunter from overlapping the target, so each distance is at least 1. Therefore `D >= 2`, and **D = 2 exactly when both hunters are adjacent**, satisfying the implemented capture rule. Removing the explicit capture bonus leaves dense reward for reductions in `D`. This reward still guides hunters toward the same geometric goal. The remaining time penalty and capture termination also remain in effect. The experiment removes a bonus, not all task feedback.

Without the step penalty, distance shaping and the +20 capture bonus remain. Gamma is still 0.95: a capture bonus received later has a smaller discounted value. The removed -0.05 penalty was also small relative to the capture bonus. Efficient capture can therefore persist without that penalty. These mechanisms make the observed success plausible; they do not prove which mechanism accounts for the detailed learned policies.

The environment is fully observable, has one small grid and a random target, and ends as soon as both hunters are adjacent. The target does not deliberately evade the hunters and may attempt a move into an occupied cell, which is blocked under the documented collision rules. The shared relative-state representation reuses experience across locations and hunters. About 94% of fresh hunter decisions in the three shaped conditions encounter relative states already represented in their tables. Familiar state keys in evaluation are expected for evaluation within the same task distribution; they are not training/evaluation trajectory leakage.

Capture rate is saturated for three conditions at the 100-step budget. Capture duration and training speed still distinguish their policies. Perfect observed capture and zero sample SD describe the finite samples; they do not establish that failure probability is zero for every future episode or a different task. The audit does not change the capture rule, target behavior, grid size, budget, or learning parameters to force lower rates.

## Verification and reproduction

Four new tests check the distance-minimum relationship, independent collision calculations, frozen empty tables, and detection of deliberately corrupted capture flags. The complete suite now passes **120 tests**, with 13 existing dependency deprecation warnings. No correction to the training or scientific evaluation code was required.

```bash
python -m experiments.audit_reward_ablation --study-dir results/ablations --output-dir results/ablations_audit_new --eval-episodes 2000
python -m pytest -q
```

The actual audit used `--output-dir results/ablations_audit`. Choose a new audit directory for repetition. Saved Phase 13 raw rows and checkpoints are required; a fresh clone must reproduce those ignored artifacts first. The audit is additional frozen evaluation of the existing study and does not implement Phase 14.
