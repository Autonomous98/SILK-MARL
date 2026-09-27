import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TASKS = {
    "spread3": {
        "label": "Simple Spread 3-agent",
        "env_name": "mpe_simple_spread_v3_3",
        "env_key": "pz-mpe-simple-spread-3",
        "time_limit": 25,
        "n_agents": 3,
        "metrics": [
            "test_return_mean",
            "test_success_rate_mean",
            "test_mean_landmark_dist_mean",
            "test_max_landmark_dist_mean",
        ],
    },
    "spread4": {
        "label": "Simple Spread 4-agent",
        "env_name": "mpe_simple_spread_v3_4",
        "env_key": "pz-mpe-simple-spread-4",
        "time_limit": 25,
        "n_agents": 4,
        "metrics": [
            "test_return_mean",
            "test_success_rate_mean",
            "test_mean_landmark_dist_mean",
            "test_max_landmark_dist_mean",
        ],
    },
    "lbf2": {
        "label": "LBF 2p2f 8x8",
        "env_name": "lbf_2p_2f_coop",
        "env_key": "lbforaging:Foraging-8x8-2p-2f-coop-v3",
        "time_limit": 50,
        "n_agents": 2,
        "metrics": [
            "test_return_mean",
            "test_success_rate_mean",
            "test_food_collection_rate_mean",
        ],
    },
}

ALGORITHMS = ("qmix", "maddpg", "mappo")


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_metrics(run_dir):
    path = Path(run_dir) / "metrics.json"
    if path.exists():
        return read_json(path)
    info_path = Path(run_dir) / "info.json"
    if not info_path.exists():
        return {}
    info = read_json(info_path)
    metrics = {}
    for key, values in info.items():
        if key.endswith("_T") or not isinstance(values, list):
            continue
        metrics[key] = {
            "steps": info.get(key + "_T") or [],
            "values": values,
        }
    return metrics


def run_status(run_dir):
    path = Path(run_dir) / "run.json"
    if not path.exists():
        return "UNKNOWN"
    try:
        return str(read_json(path).get("status", "UNKNOWN"))
    except Exception:
        return "UNKNOWN"


def find_runs_by_name(repo, name):
    root = repo / "results" / "sacred"
    if not root.exists():
        return []
    matches = []
    for config_path in root.rglob("config.json"):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if config.get("name") == name:
            matches.append(config_path.parent)
    return sorted(matches, key=lambda path: path.stat().st_mtime)


def completed_run(repo, name):
    matches = find_runs_by_name(repo, name)
    completed = [path for path in matches if run_status(path) == "COMPLETED"]
    return completed[-1] if completed else None


def latest_run(repo, name):
    matches = find_runs_by_name(repo, name)
    return matches[-1] if matches else None


def run_id(run_dir):
    return Path(run_dir).name if run_dir is not None else None


def build_name(task_name, algorithm, seed, t_max, interval):
    return (
        f"pure_{algorithm}_{task_name}_{t_max}_"
        f"interval{interval}_seed{seed}_formal"
    )


def build_command(
    task_name,
    algorithm,
    seed,
    t_max,
    interval,
    test_nepisode,
    save_model,
    save_model_interval,
    use_cuda,
):
    task = TASKS[task_name]
    name = build_name(task_name, algorithm, seed, t_max, interval)
    command = [
        sys.executable,
        "src/main.py",
        f"--config={algorithm}",
        "--env-config=gymma",
        "with",
        f"env_args.time_limit={task['time_limit']}",
        f"env_args.key={task['env_key']}",
    ]
    if task_name.startswith("spread"):
        command.append(f"env_args.N={task['n_agents']}")
    command.extend(
        [
            f"env_name={task['env_name']}",
            "use_llm=False",
            "use_knowledge_actions=False",
            "use_difference_credit=False",
            "use_credit_assignment=False",
            "use_dynamic_credit=False",
            "knowledge_candidate_id=-1",
            f"seed={seed}",
            f"t_max={t_max}",
            f"test_interval={interval}",
            f"test_nepisode={test_nepisode}",
            f"log_interval={interval}",
            f"runner_log_interval={interval}",
            f"learner_log_interval={interval}",
            f"save_model={str(save_model)}",
            f"save_model_interval={save_model_interval}",
            "use_wandb=False",
            f"use_cuda={str(use_cuda)}",
            f"name={name}",
        ]
    )
    if algorithm == "mappo":
        command.extend(
            [
                "lr=0.0003",
                "q_nstep=5",
                "use_rnn=True",
                "standardise_rewards=True",
            ]
        )
    return name, command


def execute(command, repo, dry_run):
    print("[CMD] " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env.setdefault("MKL_NUM_THREADS", "1")
    result = subprocess.run(command, cwd=repo, env=env, check=False)
    return int(result.returncode)


def metric_points(run_dir, metric_name):
    metric = load_metrics(run_dir).get(metric_name, {})
    steps = metric.get("steps") or []
    values = metric.get("values") or []
    points = []
    for index, value in enumerate(values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        step = steps[index] if index < len(steps) else index
        points.append((int(step), numeric_value))
    return points


def make_record(task_name, algorithm, seed, name, returncode, run_dir):
    record = {
        "task": task_name,
        "task_label": TASKS[task_name]["label"],
        "algorithm": algorithm,
        "seed": seed,
        "name": name,
        "returncode": returncode,
        "run_id": run_id(run_dir),
        "run_dir": str(run_dir) if run_dir else None,
        "status": run_status(run_dir) if run_dir else "NOT_FOUND",
    }
    if run_dir:
        missing_metrics = []
        for metric_name in TASKS[task_name]["metrics"]:
            points = metric_points(run_dir, metric_name)
            record[f"{metric_name}_points"] = len(points)
            record[f"final_{metric_name}"] = points[-1][1] if points else None
            record[f"final_{metric_name}_step"] = points[-1][0] if points else None
            if not points:
                missing_metrics.append(metric_name)
        record["missing_metrics"] = ",".join(missing_metrics)
    return record


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_long_rows(records):
    rows = []
    for record in records:
        run_dir = record.get("run_dir")
        if not run_dir:
            continue
        for metric_name in TASKS[record["task"]]["metrics"]:
            for step, value in metric_points(run_dir, metric_name):
                rows.append(
                    {
                        "task": record["task"],
                        "task_label": record["task_label"],
                        "algorithm": record["algorithm"],
                        "seed": record["seed"],
                        "run_id": record["run_id"],
                        "metric": metric_name,
                        "step": step,
                        "value": value,
                        "name": record["name"],
                    }
                )
    return rows


def aggregate_rows(long_rows):
    grouped = defaultdict(list)
    for row in long_rows:
        grouped[
            (
                row["task"],
                row["task_label"],
                row["algorithm"],
                row["metric"],
                row["step"],
            )
        ].append(row["value"])
    rows = []
    for key, values in sorted(grouped.items()):
        task, task_label, algorithm, metric, step = key
        rows.append(
            {
                "task": task,
                "task_label": task_label,
                "algorithm": algorithm,
                "metric": metric,
                "step": step,
                "n_seeds": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def write_progress(out_dir, args, records):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "tasks": args.tasks,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "t_max": args.t_max,
        "interval": args.interval,
        "test_nepisode": args.test_nepisode,
        "formal_runs": records,
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "formal_final_results.csv", records)
    long_rows = collect_long_rows(records)
    write_csv(out_dir / "formal_metrics_long.csv", long_rows)
    write_csv(out_dir / "formal_metrics_mean_by_step.csv", aggregate_rows(long_rows))


def validate_args(parser, args):
    invalid_tasks = [task for task in args.tasks if task not in TASKS]
    invalid_algorithms = [
        algorithm for algorithm in args.algorithms if algorithm not in ALGORITHMS
    ]
    if invalid_tasks:
        parser.error(f"unknown tasks: {','.join(invalid_tasks)}")
    if invalid_algorithms:
        parser.error(f"unknown algorithms: {','.join(invalid_algorithms)}")
    if not args.seeds:
        parser.error("at least one seed is required")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run pure QMIX, MADDPG, and MAPPO baselines for Simple Spread "
            "3-agent, Simple Spread 4-agent, and LBF 2-agent."
        )
    )
    parser.add_argument("--tasks", default="spread3,spread4,lbf2")
    parser.add_argument("--algorithms", default="qmix,maddpg,mappo")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--t-max", type=int, default=2_000_000)
    parser.add_argument("--interval", type=int, default=1_000)
    parser.add_argument("--test-nepisode", type=int, default=10)
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-model-interval", type=int, default=500_000)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="selected_pure_marl_2m_baselines_summary",
    )
    args = parser.parse_args()
    args.tasks = parse_csv(args.tasks)
    args.algorithms = parse_csv(args.algorithms)
    args.seeds = parse_csv(args.seeds, int)
    validate_args(parser, args)

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.output_dir
    combinations = [
        (task, algorithm, seed)
        for task in args.tasks
        for algorithm in args.algorithms
        for seed in args.seeds
    ]
    records = []
    failures = []

    for index, (task, algorithm, seed) in enumerate(combinations, start=1):
        name, command = build_command(
            task,
            algorithm,
            seed,
            args.t_max,
            args.interval,
            args.test_nepisode,
            args.save_model,
            args.save_model_interval,
            args.use_cuda,
        )
        print(
            f"\n[RUN {index}/{len(combinations)}] "
            f"task={task} algorithm={algorithm} seed={seed}",
            flush=True,
        )
        existing = None if args.rerun_completed else completed_run(repo, name)
        if existing is not None:
            print(f"[SKIP] completed run: {existing}", flush=True)
            record = make_record(task, algorithm, seed, name, 0, existing)
            records.append(record)
            write_progress(out_dir, args, records)
            continue

        returncode = execute(command, repo, args.dry_run)
        current = None if args.dry_run else latest_run(repo, name)
        record = make_record(task, algorithm, seed, name, returncode, current)
        records.append(record)
        write_progress(out_dir, args, records)

        metrics_missing = bool(record.get("missing_metrics"))
        if (
            returncode != 0
            or (current and run_status(current) != "COMPLETED")
            or metrics_missing
        ):
            failures.append(record)
            print(
                f"[FAIL] task={task} algorithm={algorithm} seed={seed} "
                f"returncode={returncode} status={record['status']} "
                f"missing_metrics={record.get('missing_metrics', '')}",
                flush=True,
            )
            if not args.continue_on_error:
                break
        elif current:
            print(
                f"[OK] run_id={run_id(current)} status=COMPLETED",
                flush=True,
            )

    write_progress(out_dir, args, records)
    print(f"\n[SUMMARY] wrote: {out_dir}", flush=True)
    print(
        f"[SUMMARY] completed records={len(records)} "
        f"expected={len(combinations)} failures={len(failures)}",
        flush=True,
    )
    if failures or len(records) != len(combinations):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
