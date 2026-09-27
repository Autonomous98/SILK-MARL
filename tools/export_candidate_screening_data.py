#!/usr/bin/env python3
"""Export the three-task candidate-screening metrics from Sacred runs.

The default run IDs match the validated 200k screening runs on the server:

* Spread3: 63, 64, 65
* Spread4: 96, 97, 98
* LBF2:     5,  6,  7

Only Python's standard library is required. CSV files use UTF-8 with BOM so
that they open cleanly in Excel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CANDIDATE_MODELS = {
    0: "qwen3.7-flash",
    1: "qwen3.7-plus",
    2: "qwen3.8-flash",
}

TASKS = {
    "spread3": {
        "display_name": "Simple Spread (3 agents)",
        "sacred_group": "pz-mpe-simple-spread",
        "env_name": "mpe_simple_spread_v3_3",
        "default_run_ids": (63, 64, 65),
        "metrics": (
            "test_return_mean",
            "test_success_rate_mean",
            "test_mean_landmark_dist_mean",
            "test_max_landmark_dist_mean",
        ),
        "task_metric": "test_mean_landmark_dist_mean",
    },
    "spread4": {
        "display_name": "Simple Spread (4 agents)",
        "sacred_group": "pz-mpe-simple-spread",
        "env_name": "mpe_simple_spread_v3_4",
        "default_run_ids": (96, 97, 98),
        "metrics": (
            "test_return_mean",
            "test_success_rate_mean",
            "test_mean_landmark_dist_mean",
            "test_max_landmark_dist_mean",
        ),
        "task_metric": "test_mean_landmark_dist_mean",
    },
    "lbf2": {
        "display_name": "LBF 8x8 (2 agents, 2 foods)",
        "sacred_group": "lbforaging:Foraging-8x8-2p-2f-coop-v3",
        "env_name": "lbf_2p_2f_coop",
        "default_run_ids": (5, 6, 7),
        "metrics": (
            "test_return_mean",
            "test_success_rate_mean",
            "test_food_collection_rate_mean",
        ),
        "task_metric": "test_food_collection_rate_mean",
    },
}

NOMINAL_STEPS = (0, 50_000, 100_000, 150_000, 200_000)
COMMON_POINT_COUNT = 4


def parse_run_ids(value: str) -> tuple[int, int, int]:
    try:
        run_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run IDs must be comma-separated integers") from exc
    if len(run_ids) != 3:
        raise argparse.ArgumentTypeError("exactly three run IDs are required")
    return run_ids  # type: ignore[return-value]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_points(metrics: dict[str, Any], metric_name: str) -> list[tuple[int, float]]:
    entry = metrics.get(metric_name, {})
    steps = entry.get("steps", [])
    values = entry.get("values", [])
    if len(steps) != len(values):
        raise ValueError(
            f"{metric_name}: step/value length mismatch ({len(steps)} != {len(values)})"
        )
    return [(int(step), float(value)) for step, value in zip(steps, values)]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo / "results" / "sacred",
        help="Sacred results root (default: <repo>/results/sacred)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "selected_candidate_screening_summary",
        help="Directory for exported CSV/JSON files",
    )
    parser.add_argument("--spread3-run-ids", type=parse_run_ids, default=(63, 64, 65))
    parser.add_argument("--spread4-run-ids", type=parse_run_ids, default=(96, 97, 98))
    parser.add_argument("--lbf2-run-ids", type=parse_run_ids, default=(5, 6, 7))
    args = parser.parse_args()

    selected_ids = {
        "spread3": args.spread3_run_ids,
        "spread4": args.spread4_run_ids,
        "lbf2": args.lbf2_run_ids,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for task_name, task in TASKS.items():
        run_ids = selected_ids[task_name]
        for candidate_id, run_id in enumerate(run_ids):
            run_dir = (
                args.results_root
                / "qmix"
                / str(task["sacred_group"])
                / str(run_id)
            )
            config = load_json(run_dir / "config.json")
            run = load_json(run_dir / "run.json")
            metrics = load_json(run_dir / "metrics.json")

            errors = []
            if run.get("status") != "COMPLETED":
                errors.append(f"status={run.get('status')}")
            if config.get("env_name") != task["env_name"]:
                errors.append(f"env_name={config.get('env_name')}")
            if int(config.get("knowledge_candidate_id", -999)) != candidate_id:
                errors.append(f"candidate={config.get('knowledge_candidate_id')}")
            if int(config.get("t_max", -1)) != 200_000:
                errors.append(f"t_max={config.get('t_max')}")
            if errors:
                raise ValueError(f"Unexpected run {run_dir}: " + ", ".join(errors))

            points_by_metric = {
                metric_name: metric_points(metrics, metric_name)
                for metric_name in task["metrics"]
            }
            missing_metrics = [name for name, points in points_by_metric.items() if not points]
            if missing_metrics:
                raise ValueError(f"Run {run_dir} has no data for: {missing_metrics}")

            min_points = min(len(points) for points in points_by_metric.values())
            if min_points < COMMON_POINT_COUNT:
                raise ValueError(
                    f"Run {run_dir} has only {min_points} points; four common points are required"
                )

            model = CANDIDATE_MODELS[candidate_id]
            base = {
                "task": task_name,
                "task_display_name": task["display_name"],
                "run_id": run_id,
                "candidate_id": candidate_id,
                "candidate_model": model,
                "env_name": config.get("env_name"),
            }

            for metric_name, points in points_by_metric.items():
                wide = {**base, "metric": metric_name}
                for point_index, (actual_step, value) in enumerate(points):
                    nominal_step = (
                        NOMINAL_STEPS[point_index]
                        if point_index < len(NOMINAL_STEPS)
                        else point_index * 50_000
                    )
                    row = {
                        **base,
                        "metric": metric_name,
                        "point_index": point_index,
                        "nominal_step": nominal_step,
                        "actual_step": actual_step,
                        "value": value,
                    }
                    all_rows.append(row)
                    if point_index < COMMON_POINT_COUNT:
                        common_rows.append(row)

                    label = f"{nominal_step // 1000}k"
                    wide[f"actual_step_{label}"] = actual_step
                    wide[f"value_{label}"] = value
                wide_rows.append(wide)

            final_index = COMMON_POINT_COUNT - 1
            return_step, return_value = points_by_metric["test_return_mean"][final_index]
            success_step, success_value = points_by_metric[
                "test_success_rate_mean"
            ][final_index]
            task_metric_name = str(task["task_metric"])
            task_metric_step, task_metric_value = points_by_metric[task_metric_name][final_index]
            final_rows.append(
                {
                    **base,
                    "nominal_step": NOMINAL_STEPS[final_index],
                    "return_actual_step": return_step,
                    "test_return_mean": return_value,
                    "success_actual_step": success_step,
                    "test_success_rate_mean": success_value,
                    "task_metric": task_metric_name,
                    "task_metric_actual_step": task_metric_step,
                    "task_metric_value": task_metric_value,
                }
            )

            manifest_runs.append(
                {
                    **base,
                    "run_dir": str(run_dir),
                    "status": run.get("status"),
                    "available_points": {
                        name: len(points) for name, points in points_by_metric.items()
                    },
                    "has_approximately_200k_point": all(
                        len(points) >= 5 for points in points_by_metric.values()
                    ),
                }
            )

    long_fields = [
        "task",
        "task_display_name",
        "run_id",
        "candidate_id",
        "candidate_model",
        "env_name",
        "metric",
        "point_index",
        "nominal_step",
        "actual_step",
        "value",
    ]
    wide_fields = [
        "task",
        "task_display_name",
        "run_id",
        "candidate_id",
        "candidate_model",
        "env_name",
        "metric",
    ]
    for nominal_step in NOMINAL_STEPS:
        label = f"{nominal_step // 1000}k"
        wide_fields.extend((f"actual_step_{label}", f"value_{label}"))

    final_fields = [
        "task",
        "task_display_name",
        "run_id",
        "candidate_id",
        "candidate_model",
        "env_name",
        "nominal_step",
        "return_actual_step",
        "test_return_mean",
        "success_actual_step",
        "test_success_rate_mean",
        "task_metric",
        "task_metric_actual_step",
        "task_metric_value",
    ]

    outputs = {
        "all_points": output_dir / "candidate_screening_all_available_points.csv",
        "common_four_points": output_dir / "candidate_screening_common_4points.csv",
        "wide_table": output_dir / "candidate_screening_table_wide.csv",
        "common_150k_summary": output_dir / "candidate_screening_150k_summary.csv",
        "manifest": output_dir / "candidate_screening_manifest.json",
    }
    write_csv(outputs["all_points"], long_fields, all_rows)
    write_csv(outputs["common_four_points"], long_fields, common_rows)
    write_csv(outputs["wide_table"], wide_fields, wide_rows)
    write_csv(outputs["common_150k_summary"], final_fields, final_rows)
    outputs["manifest"].write_text(
        json.dumps(
            {
                "description": "Candidate screening data for Spread3, Spread4, and LBF2",
                "common_nominal_steps": list(NOMINAL_STEPS[:COMMON_POINT_COUNT]),
                "candidate_models": CANDIDATE_MODELS,
                "runs": manifest_runs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[OK] exported {len(manifest_runs)} candidate-screening runs")
    for label, path in outputs.items():
        print(f"[OK] {label}: {path}")


if __name__ == "__main__":
    main()
