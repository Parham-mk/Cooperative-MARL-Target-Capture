# Contributions

This project contributes a small, inspectable target-capture environment, tabular learning implementations, matched multi-seed experiments, behavioral diagnostics, controlled reward ablations, frozen-policy robustness tests, and reproducibility evidence. It does not claim a new reinforcement-learning algorithm or a publication.

Git history identifies the contributors as **BornaMaherani** (also recorded as Borna) and **Parham Mohammadkhani** (also recorded as Parham-mk/parham-mk). These verified names are used in citation metadata; personal-name expansions, affiliations and ORCIDs require owner confirmation. No per-person responsibility assignment is inferred from the development plan.

For a change, explain the affected behavior and run `python -m pytest -q`. For experiment changes, save the exact configuration and seeds, retain every failure, and report whether existing evidence is affected. Run new studies in fresh output directories. Raw datasets and checkpoint collections are regenerable and ignored; selected evidence stays in `results/reproducibility`, `results/ablations`, `results/ablations_audit` and `results/robustness`.

Follow [the reproduction guide](docs/reproducibility.md) for a full functional demonstration. Changes to capture, collision, target movement, reward accounting, state encodings or Q-update mathematics require explicit scientific justification and regeneration of affected evidence.
