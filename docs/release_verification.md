# Release verification

Finalization validated the intended release file set in a separate directory containing tracked files and proposed deliverables, without ignored checkpoints, raw data, virtual environments, caches or Project Context prompts. The snapshot was prepared from `git ls-files` plus nonignored proposed deliverables; no Git commit, push or release was created. Final documentation/verification records were copied into the snapshot afterward and links were rechecked. All runtime source/test files match the tested snapshot by SHA-256.

## Executed clean-checkout validation

On 2026-10-06, a new virtual environment installed only `requirements.txt` and its transitive dependencies. `pip check` passed. The snapshot used Python 3.12.7, pytest 9.1.1, NumPy 2.5.3, Matplotlib 3.11.2 and Pillow 12.3.0. Python 3.10+ is the syntax requirement; other Python versions and arbitrary package versions were not exhaustively tested.

The original pre-change suite passed **128 tests**. After functional corrections it passed **140 tests**; the final clean snapshot passed **140 tests in 11.03 seconds**, without warnings. The existing development environment uses older Matplotlib and emits 13 dependency deprecation warnings. Fast correctness checks cover environment/learning contracts, actual held-out episode ranges, catalog exclusions, reproducible frozen rollouts, visit-weighted coverage, paired deltas, complete repeated robustness pipelines and corrupt-data rejection. Cosmetic README edits have no superficial tests.

The clean snapshot executed these workflows successfully:

| Check | Outcome |
|---|---|
| All source-module imports and 15 documented CLI help commands | Passed |
| Full `python -m experiments.reproduce --output-dir results/demo_verified` | Passed; newly generated inputs only |
| Baseline evaluation; individual Independent/Cooperative training and evaluation | Passed |
| Behavioral generation using the newly generated main checkpoints | Passed; real trajectories, metrics and GIFs |
| Comparison, ablation and robustness saved-data analysis CLIs | Passed; regenerated summaries and figures |
| Independent ablation audit with five fresh episodes per checkpoint | Passed; functional smoke budget only |
| Two tiny main/behavior runs and local evidence-copy verification CLI | Passed; checkpoint/raw/summary byte comparison |
| Tiny robustness with source checkpoint-replica verification | Passed; Standard rows reproduced |
| Full clean-snapshot pytest suite | 140 passed |

The demonstration produced 200 main evaluation rows, 200 ablation rows and 400 robustness rows. Its 4×4 grid, 50-episode training, two-seed budget is explicitly distinct from the full research protocol. The exact command arguments, module inventory, output counts, dependency versions and generated-image inventory are in [the machine-readable record](release_verification.json). Full-budget main and ablation studies were not needlessly retrained during finalization. The missing full robustness study was executed once, using original checkpoints, for 20,000 episodes / 410,536 transitions.

## Artifact and citation inspection

README, contribution and documentation links were checked against the intended release file set, including Markdown anchors. All tracked artifact-manifest entries were checked: 27 main evidence entries, 27 ablation entries, three audit entries and 17 robustness entries. Ignored raw/checkpoint entries in historical manifests are deliberately excluded from release-file checks and remain documented as regenerable inputs.

Selected figures and GIF frames were visually inspected for labels, method names, legends and capture geometry. The old GIF layout clipped legend text outside its 600×600 canvas. The legend now fits below the grid; five selected main GIFs were regenerated from the **unchanged saved trajectories**. Their before/after hashes are preserved in the verification record. The main selected-evidence manifest was refreshed; no numeric results, selection criteria, recorded trajectory or checkpoint changed. The historical 35-artifact reproducibility claim excludes GIF bytes and remains intact.

`CITATION.cff` passed JSON Schema validation against the [official CFF 1.2.0 schema](https://github.com/citation-file-format/citation-file-format/blob/main/schema.json), using already installed development validators; those validators are not runtime requirements. Git history and the remote were checked for authorship/repository fields. The metadata records verified given name Borna and handle BornaMaherani, and Parham Mohammadkhani/Parham-mk. Borna's preferred full citation name, optional affiliations/ORCIDs and any future DOI are owner metadata tasks, not unfinished technical work. No publication or release date was invented.

The full-budget studies remain distinct from demonstration runs. [Main reproducibility](../results/reproducibility/reproduction_verification.json), [ablation audit](../results/ablations_audit/audit.json) and [robustness verification](../results/robustness/summaries/verification.json) describe their own scientific scope. [Reproduction instructions](reproducibility.md) specify clean-checkout prerequisites.
