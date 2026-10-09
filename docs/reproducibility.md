# Reproducing the project

Run every command from the repository root. Python 3.10 or newer is required; release validation uses Python 3.12. Runtime dependencies are NumPy, Matplotlib and Pillow; pytest runs the tests. `requirements.txt` intentionally retains the existing package list. Exact tested versions and verification scope are recorded in [release verification](release_verification.md); bitwise reproducibility across arbitrary package versions or Python releases is not promised.

## Installation and tests

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

If PowerShell activation is restricted, call `.\.venv\Scripts\python.exe` wherever `python` appears below. POSIX equivalent:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The experiment commands below are identical in both shells. Rendering is headless and requires no GUI. Commands create directories automatically. Pick a new destination if an output already exists; the demo, ablation and robustness runners protect existing evidence. The comparison and standalone training commands can overwrite their own outputs, so use fresh directories there too.

## Fast end-to-end demonstration

```text
python -m experiments.reproduce --output-dir results/demo_new
```

This creates `main/`, `ablations/` and `robustness/` under the destination. It composes existing runners, using a 4×4 grid, 30-step limit, two seeds, 50 training episodes and 25 evaluation episodes per seed. It trains the main checkpoints, generates quantitative and behavioral results/GIFs, trains all four ablations, and evaluates the main checkpoints on all robustness conditions. Ablation uses a complete 10-episode rolling window; stress probes use five repetitions per case and seed. **These are functional smoke budgets, not the research results.** No prior raw data, checkpoint, environment or ignored Project Context file is required.

## Baselines and individual policies

Baseline evaluation prints capture, length, conditional capture time and reward; it writes no study files:

```text
python -m experiments.baseline_evaluation --episodes 20 --grid-size 4 --max-steps 30 --seed 0
```

Independent training writes `results/individual_independent/checkpoints/independent_q/seed_0_agent0.pkl`, the corresponding agent1 file, and a training CSV. The evaluator requires those generated checkpoints, prints metrics and does not train:

```text
python -m experiments.train_q_learning --episodes 50 --grid-size 4 --max-steps 30 --seed 0 --output-dir results/individual_independent
python -m experiments.evaluate_q_learning --episodes 20 --grid-size 4 --max-steps 30 --seed 0 --agent0-path results/individual_independent/checkpoints/independent_q/seed_0_agent0.pkl --agent1-path results/individual_independent/checkpoints/independent_q/seed_0_agent1.pkl
```

Cooperative training writes one shared checkpoint and its training CSV:

```text
python -m experiments.train_cooperative_q_learning --episodes 50 --grid-size 4 --max-steps 30 --seed 0 --output-dir results/individual_cooperative
python -m experiments.evaluate_cooperative_q_learning --episodes 20 --grid-size 4 --max-steps 30 --seed 0 --model-path results/individual_cooperative/checkpoints/cooperative_q/seed_0.pkl
```

For the main budgets, use a 10×10 grid, 100-step limit, 5,000 training episodes and 500 evaluation episodes. Individual-policy commands are demonstrations, not substitutes for the matched multi-seed comparison.

## Full scientific reproduction

### Main comparison and behavioral evidence

```text
python -m experiments.run_comparison --grid-size 10 --max-steps 100 --train-episodes 5000 --eval-episodes 500 --seeds 0 1 2 3 4 --output-dir results/reproduced/main
python -m experiments.generate_behavioral_examples --grid-size 10 --max-steps 100 --eval-episodes 500 --seeds 0 1 2 3 4 --output-dir results/reproduced/main
```

The comparison trains Independent and Cooperative methods separately and writes 15 checkpoint files, 10 training histories, four raw evaluation CSVs, configuration, summaries and five plots. Behavioral generation loads those checkpoints and repeats frozen evaluation with the same seeds and tie streams. It saves metrics, selected trajectories, trajectory figures and GIFs. Keep grid size, horizon, evaluation count and seed order identical between these commands. Both generate 500 episodes per method and seed, 2,500 per method.

### Reward ablation and independent audit

```text
python -m experiments.run_reward_ablation --grid-size 10 --max-steps 100 --train-episodes 5000 --eval-episodes 500 --seeds 0 1 2 3 4 --output-dir results/reproduced/ablations
python -m experiments.audit_reward_ablation --study-dir results/reproduced/ablations --output-dir results/reproduced/ablations_audit --eval-episodes 2000
```

Ablation starts 20 fresh shared Q-tables and writes `raw/`, `raw/training/`, `checkpoints/`, `summaries/`, `behavioral/`, `trajectories/` and `plots/`. It needs no main checkpoint. Audit requires the generated ablation raw data and checkpoints; it independently checks original rows and runs 2,000 fresh episodes per checkpoint plus the untrained control. The audit is optional for an ordinary smoke demonstration, but reproduces the preserved capture-rate investigation.

### Robustness

```text
python -m experiments.run_robustness_evaluation --comparison-dir results/reproduced/main --output-dir results/reproduced/robustness --eval-episodes 500 --initial-state-cases 500 --stress-repeats 100
```

**Prerequisite:** the completed main comparison above, including saved configuration, raw data, histories, plots and all original full-reward checkpoints. Behavioral generation is not a prerequisite. The runner reads the source grid, horizon, training budget and seed order from its saved configuration; it never retrains. Standard evaluation exactly repeats the main learned-policy rows. Each learned method gets 2,500 episodes per primary condition and 500 per spatial probe (five probes). The total is 20,000 rollouts across both methods.

The held-out catalog is checked against the complete source training and main evaluation reset catalogs. The historical Phase 13 audit environment range 30,000,000–30,009,999 is excluded by default. If you have conducted other evaluations, add each inclusive range with `--prior-evaluation-range FIRST LAST`. Unrecorded experiments cannot be excluded automatically. Optional `--verify-against results/reproduced/main_repeat` verifies checkpoint hashes against a second completed source run. The actual preserved study used `results/validation/run_a` and `run_b`; those ignored local paths are provenance, not fresh-clone prerequisites.

### Rebuild analysis from generated raw data

```text
python -m experiments.statistical_analysis --output-dir results/reproduced/main
python -m analysis.reward_ablation_analysis --output-dir results/reproduced/ablations
python -m analysis.robustness_analysis --output-dir results/reproduced/robustness
```

These commands need the raw outputs generated above. Ablation also needs histories/behavioral metrics and saved configuration; robustness needs its raw CSV and configuration. They rebuild summaries and plots without loading checkpoints or training. Robustness analysis validates episode inventory, matched starts, outcomes and coverage counts. After deliberately regenerating artifacts, refresh a manifest with the existing helper:

```text
python -c "from pathlib import Path; from experiments.reproducibility import create_manifest; create_manifest(Path('results/reproduced/robustness'))"
```

For demo analysis, substitute `results/demo_new/main`, `results/demo_new/ablations` and `results/demo_new/robustness`.

## Deterministic repeat verification

Run the full main comparison and behavioral commands again into `results/reproduced/main_repeat`, then:

```text
python -m experiments.reproducibility --run-a results/reproduced/main --run-b results/reproduced/main_repeat --train-episodes 5000 --eval-episodes 500 --seeds 0 1 2 3 4 --publish-dir results/reproduced/evidence
```

Here “publish” only copies selected evidence to a **new local directory**; it performs no GitHub action and refuses an existing destination. The utility validates both inventories/summaries and their behavioral agreement, then compares raw data, histories, checkpoints, summaries and plot-source CSVs by SHA-256. GIFs and rendered plots are checked separately; they are not included in the 35-artifact byte-identity claim.

## What a clone contains

Selected summaries/configuration, plot-source data, figures, representative trajectories/GIFs and verification records are tracked. Large raw datasets and checkpoint collections are ignored. Therefore the saved evidence can be inspected immediately, but raw-data analyses and frozen evaluations require regeneration first. Historical manifests identify their original checked scope; ablation, audit and robustness manifests include local raw files absent from a clone. Missing ignored files are expected, not evidence that a clone contains the full raw study. See [methods and evidence](methods_and_evidence.md), [robustness report](robustness_evaluation.md), and [release verification](release_verification.md).
