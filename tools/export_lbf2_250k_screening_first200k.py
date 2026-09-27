#!/usr/bin/env python3
"""Export the first 200k test points from the LBF2 250k candidate rerun.

The 250k pipeline selects its [BEST] candidate using the last available test
point. This script instead exports the common 0/50k/100k/150k/200k points
for the paper, without mixing new and original screening runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ENV_NAME = "lbf_2p_2f_coop"
SACRED_GROUP = "lbforaging:Foraging-8x8-2p-2f-coop-v3"
MODELS = ("qwen3.7-flash", "qwen3.7-plus", "qwen3.8-flash")
METRICS = (
    "test_return_mean",
    "test_success_rate_mean",
    "test_food_collection_rate_mean",
)
NOMINAL_STEPS = (0, 50_000, 100_000, 150_000, 200_000)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_ids(value: str) -> tuple[int, int, int]:
    try:
        ids = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run IDs must be comma-separated integers") from exc
    if len(ids) != 3 or len(set(ids)) != 3:
        raise argparse.ArgumentTypeError("provide three distinct run IDs in candidate 0,1,2 order")
    return ids


def validate_run(run_dir: Path, candidate: int, t_max: int, seed: int) -> dict:
    config = read_json(run_dir / "config.json")
    run = read_json(run_dir / "run.json")
    expected = {
        "env_name": ENV_NAME,
        "knowledge_candidate_id": candidate,
        "t_max": t_max,
        "test_interval": 50_000,
        "seed": seed,
        "use_knowledge_actions": True,
        "use_difference_credit": True,
    }
    errors = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if run.get("status") != "COMPLETED":
        errors.append(f"status={run.get('status')!r}")
    if "exp3_screen" not in str(config.get("name", "")):
        errors.append(f"name={config.get('name')!r} is not a screening run")
    if errors:
        raise ValueError(f"Unexpected run {run_dir}: " + "; ".join(errors))
    return config


def find_new_runs(group_dir: Path, seed: int) -> tuple[int, int, int]:
    by_candidate: dict[int, list[int]] = {candidate: [] for candidate in range(3)}
    for config_path in group_dir.glob("*/config.json"):
        config = read_json(config_path)
        if (
            config.get("env_name") == ENV_NAME
            and config.get("t_max") == 250_000
            and config.get("test_interval") == 50_000
            and config.get("seed") == seed
            and "exp3_screen" in str(config.get("name", ""))
        ):
            candidate = config.get("knowledge_candidate_id")
            if candidate in by_candidate:
                by_candidate[candidate].append(int(config_path.parent.name))
    if any(len(ids) != 1 for ids in by_candidate.values()):
        raise ValueError(
            f"Expected one 250k run per candidate, found {by_candidate}. "
            "Pass --run-ids ID0,ID1,ID2 to select the intended runs."
        )
    return tuple(by_candidate[candidate][0] for candidate in range(3))


def read_points(run_dir: Path, required_count: int) -> dict[str, list[tuple[int, float]]]:
    metrics = read_json(run_dir / "metrics.json")
    result = {}
    for metric in METRICS:
        entry = metrics.get(metric, {})
        steps, values = entry.get("steps", []), entry.get("values", [])
        if len(steps) != len(values) or len(values) < required_count:
            raise ValueError(
                f"{run_dir}: {metric} has {len(steps)} steps and {len(values)} "
                f"values; need at least {required_count} paired points"
            )
        points = [(int(step), float(value)) for step, value in zip(steps, values)]
        for index, nominal in enumerate(NOMINAL_STEPS[:required_count]):
            if abs(points[index][0] - nominal) > 1_000:
                raise ValueError(
                    f"{run_dir}: {metric} point {index} is at {points[index][0]}, "
                    f"not approximately {nominal}"
                )
        result[metric] = points
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=repo / "results" / "sacred")
    parser.add_argument("--run-ids", type=parse_run_ids, help="New 250k run IDs, candidate 0,1,2")
    parser.add_argument("--old-run-ids", type=parse_run_ids, default=(5, 6, 7))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "selected_lbf2_candidate_screening_250k_first200k",
    )
    args = parser.parse_args()
    group_dir = args.results_root / "qmix" / SACRED_GROUP
    new_ids = args.run_ids or find_new_runs(group_dir, args.seed)

    long_rows = []
    wide_rows = []
    run_manifest = []
    for candidate, (new_id, old_id) in enumerate(zip(new_ids, args.old_run_ids)):
        new_dir, old_dir = group_dir / str(new_id), group_dir / str(old_id)
        new_config = validate_run(new_dir, candidate, 250_000, args.seed)
        old_config = validate_run(old_dir, candidate, 200_000, args.seed)
        new_points = read_points(new_dir, 5)
        old_points = read_points(old_dir, 4)
        base = {
            "candidate_id": candidate,
            "model": MODELS[candidate],
            "new_run_id": new_id,
            "old_run_id": old_id,
        }
        wide = dict(base)
        for index, nominal in enumerate(NOMINAL_STEPS):
            label = f"{nominal // 1000}k"
            row = {**base, "nominal_step": nominal}
            for metric in METRICS:
                new_step, new_value = new_points[metric][index]
                old_value = old_points[metric][index][1] if index < len(old_points[metric]) else ""
                row[f"{metric}_actual_step"] = new_step
                row[metric] = new_value
                row[f"old_{metric}"] = old_value
                wide[f"{metric}_{label}"] = new_value
            long_rows.append(row)
        wide_rows.append(wide)
        run_manifest.append(
            {
                **base,
                "new_name": new_config["name"],
                "old_name": old_config["name"],
                "new_run_dir": str(new_dir),
                "old_run_dir": str(old_dir),
                "new_test_points": {metric: len(points) for metric, points in new_points.items()},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_fields = ["candidate_id", "model", "new_run_id", "old_run_id", "nominal_step"]
    for metric in METRICS:
        long_fields.extend((f"{metric}_actual_step", metric, f"old_{metric}"))
    wide_fields = ["candidate_id", "model", "new_run_id", "old_run_id"] + [
        f"{metric}_{nominal // 1000}k"
        for metric in METRICS
        for nominal in NOMINAL_STEPS
    ]
    long_path = args.output_dir / "lbf2_250k_first200k_paper_rows.csv"
    wide_path = args.output_dir / "lbf2_250k_first200k_wide.csv"
    manifest_path = args.output_dir / "lbf2_250k_first200k_manifest.json"
    write_csv(long_path, long_fields, long_rows)
    write_csv(wide_path, wide_fields, wide_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "note": "All paper values come from the new 250k rerun; old values are comparison only. The pipeline's [BEST] uses its last (~250k) point, not the fifth (~200k) point.",
                "nominal_steps": NOMINAL_STEPS,
                "runs": run_manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("[OK] New LBF2 250k runs, paper values through approximately 200k:")
    for row in long_rows:
        print(
            f"candidate{row['candidate_id']} run={row['new_run_id']} "
            f"nominal={row['nominal_step']:>6} "
            f"actual={row['test_return_mean_actual_step']:>6} "
            f"return={row['test_return_mean']:.6f} "
            f"success={row['test_success_rate_mean']:.6f} "
            f"food={row['test_food_collection_rate_mean']:.6f}"
        )
    print(f"[OK] paper rows: {long_path}")
    print(f"[OK] wide table: {wide_path}")
    print(f"[OK] run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
