"""Validation, hashing, and evidence publishing for reproduced experiments."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from experiments.statistical_analysis import aggregate_rows, read_raw_results


RAW_FILES = (
    "random_results.csv", "heuristic_results.csv",
    "independent_q_results.csv", "cooperative_q_results.csv",
)
PLOTS = (
    "capture_rate_comparison.png", "episode_length_comparison.png",
    "reward_comparison.png", "training_reward_curve.png",
    "training_capture_curve.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(root: Path, output: Path = None):
    root = Path(root)
    output = output or root / "summaries" / "artifact_manifest.json"
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != output):
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    return entries


def validate_inventory(root: Path, seeds, train_episodes: int, eval_episodes: int):
    root = Path(root)
    expected_eval_rows = len(seeds) * eval_episodes
    for filename in RAW_FILES:
        rows = read_raw_results(root / "raw" / filename)
        if len(rows) != expected_eval_rows:
            raise AssertionError(f"{filename}: expected {expected_eval_rows} rows, got {len(rows)}")
        required = {
            "method", "training_seed", "evaluation_seed", "episode", "captured",
            "episode_length", "episode_reward", "capture_time",
        }
        if not rows or set(rows[0]) != required:
            raise AssertionError(f"{filename}: unexpected schema")
        for row in rows:
            for field in ("captured", "episode_length", "episode_reward"):
                if not math.isfinite(float(row[field])):
                    raise AssertionError(f"{filename}: non-finite {field}")

    independent = list((root / "checkpoints" / "independent_q").glob("*.pkl"))
    cooperative = list((root / "checkpoints" / "cooperative_q").glob("*.pkl"))
    if len(independent) != len(seeds) * 2 or len(cooperative) != len(seeds):
        raise AssertionError("checkpoint inventory does not match seed count")
    histories = list((root / "training").glob("*.csv"))
    if len(histories) != len(seeds) * 2:
        raise AssertionError("training-history inventory does not match seed count")
    for path in histories:
        if len(read_raw_results(path)) != train_episodes:
            raise AssertionError(f"{path.name}: unexpected training row count")
    for filename in PLOTS:
        path = root / "plots" / filename
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty plot: {filename}")
    return True


def validate_summary(root: Path, tolerance: float = 1e-12):
    root = Path(root)
    stored = {
        row["method"]: row
        for row in read_raw_results(root / "summaries" / "comparison_summary.csv")
    }
    for filename in RAW_FILES:
        rows = read_raw_results(root / "raw" / filename)
        calculated, _ = aggregate_rows(rows)
        actual = stored[calculated["method"]]
        for key, expected in calculated.items():
            if key == "method":
                continue
            observed = float(actual[key])
            if math.isnan(expected):
                if not math.isnan(observed):
                    raise AssertionError(f"{calculated['method']} {key}: expected NaN")
            elif not math.isclose(observed, float(expected), rel_tol=tolerance, abs_tol=tolerance):
                raise AssertionError(
                    f"{calculated['method']} {key}: {observed} != {expected}"
                )
    return True


def validate_behavioral(root: Path, seeds, eval_episodes: int):
    root = Path(root)
    metrics = read_raw_results(root / "behavioral" / "behavioral_metrics.csv")
    expected = 4 * len(seeds) * eval_episodes
    if len(metrics) != expected:
        raise AssertionError(f"expected {expected} behavioral rows, got {len(metrics)}")
    seed_summary = read_raw_results(root / "behavioral" / "behavioral_seed_summary.csv")
    summary = read_raw_results(root / "behavioral" / "behavioral_summary.csv")
    if len(seed_summary) != 4 * len(seeds) or len(summary) != 4:
        raise AssertionError("behavioral seed or method summary is incomplete")

    quantitative = {}
    for filename in RAW_FILES:
        for row in read_raw_results(root / "raw" / filename):
            quantitative[(
                row["method"], int(row["training_seed"]), int(row["evaluation_seed"])
            )] = (int(row["captured"]), int(row["episode_length"]))
    for row in metrics:
        key = (row["method"], int(row["training_seed"]), int(row["evaluation_seed"]))
        observed = (int(row["captured"]), int(row["episode_length"]))
        if quantitative.get(key) != observed:
            raise AssertionError(f"behavioral rollout differs from Phase 11 evaluation: {key}")

    selections = read_raw_results(root / "behavioral" / "representative_selection.csv")
    if len([row for row in selections if row["outcome"] == "success"]) != 4:
        raise AssertionError("one documented success selection is required per method")
    images = list((root / "gifs").glob("*.gif"))
    if len(images) < 4:
        raise AssertionError("one success GIF is required per method")
    dimensions = set()
    for path in images:
        if path.stat().st_size == 0:
            raise AssertionError(f"empty GIF: {path}")
        with Image.open(path) as image:
            dimensions.add(image.size)
            if image.n_frames < 2:
                raise AssertionError(f"GIF has fewer than two frames: {path}")
    if len(dimensions) != 1:
        raise AssertionError("GIF frame dimensions are inconsistent")
    return True


def compare_reproductions(run_a: Path, run_b: Path):
    run_a, run_b = Path(run_a), Path(run_b)
    patterns = (
        "raw/*.csv", "training/*.csv", "checkpoints/**/*.pkl",
        "summaries/comparison_summary.csv", "summaries/per_seed_summary.csv",
        "summaries/training_curve_source.csv",
        "behavioral/behavioral_metrics.csv",
        "behavioral/behavioral_seed_summary.csv",
        "behavioral/behavioral_summary.csv",
    )
    checked = []
    for pattern in patterns:
        paths_a = sorted(run_a.glob(pattern))
        paths_b = sorted(run_b.glob(pattern))
        rel_a = [p.relative_to(run_a) for p in paths_a]
        rel_b = [p.relative_to(run_b) for p in paths_b]
        if rel_a != rel_b:
            raise AssertionError(f"file sets differ for {pattern}")
        for relative in rel_a:
            hash_a = sha256(run_a / relative)
            hash_b = sha256(run_b / relative)
            if hash_a != hash_b:
                raise AssertionError(f"reproduction differs: {relative}")
            checked.append({"path": relative.as_posix(), "sha256": hash_a})
    return checked


def publish_evidence(source: Path, destination: Path):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    include = [
        "summaries/experiment_config.json",
        "summaries/comparison_summary.csv",
        "summaries/per_seed_summary.csv",
        "summaries/training_curve_source.csv",
        "behavioral/behavioral_summary.csv",
        "behavioral/behavioral_seed_summary.csv",
        "behavioral/representative_selection.csv",
    ]
    include += [f"plots/{name}" for name in PLOTS]
    include += [
        path.relative_to(source).as_posix()
        for folder in ("trajectories", "gifs")
        for path in sorted((source / folder).glob("*"))
        if path.is_file()
    ]
    include += [
        path.relative_to(source).as_posix()
        for path in sorted((source / "behavioral").glob("*_trajectory.png"))
    ]
    for relative in include:
        src = source / relative
        if src.exists():
            dst = destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    create_manifest(destination, destination / "artifact_manifest.json")


def write_reproduction_verification(destination: Path, matches, protocol: dict):
    canonical = json.dumps(matches, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "protocol": protocol,
        "byte_identical_artifact_count": len(matches),
        "comparison_fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
        "comparison_scope": [
            "raw CSVs", "training histories", "learned checkpoints",
            "comparison summaries", "training curve source data",
            "behavioral metrics and summaries",
        ],
    }
    path = Path(destination) / "reproduction_verification.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_manifest(Path(destination), Path(destination) / "artifact_manifest.json")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate two experiment reproductions")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    for root in (args.run_a, args.run_b):
        validate_inventory(root, args.seeds, args.train_episodes, args.eval_episodes)
        validate_summary(root)
        validate_behavioral(root, args.seeds, args.eval_episodes)
        create_manifest(root)
    matches = compare_reproductions(args.run_a, args.run_b)
    if args.publish_dir:
        publish_evidence(args.run_a, args.publish_dir)
        source_config = json.loads(
            (args.run_a / "summaries" / "experiment_config.json").read_text(encoding="utf-8")
        )
        write_reproduction_verification(args.publish_dir, matches, {
            "grid_size": source_config["grid_size"],
            "max_steps": source_config["max_steps"],
            "training_seeds": args.seeds,
            "training_episodes_per_learning_method_and_seed": args.train_episodes,
            "evaluation_episodes_per_method_and_seed": args.eval_episodes,
        })
    print(f"Validated {len(matches)} byte-identical reproducibility artifacts")


if __name__ == "__main__":
    main()
