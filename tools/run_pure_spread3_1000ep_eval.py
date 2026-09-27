"""Evaluate nine pure-MARL Spread3 checkpoints for 1000 episodes each.

Run from the YOLO-MARL root after copying into tools/. Requires the existing
run_saved_model_15x1000_eval.py helper and the patched src/run.py.
"""

import argparse
import math
import os
import shlex
import statistics
import subprocess
from pathlib import Path

import run_saved_model_15x1000_eval as shared


METHODS = ("qmix", "maddpg", "mappo")
METRICS = (
    "test_return_mean",
    "test_success_rate_mean",
    "test_mean_landmark_dist_mean",
)


def selections():
    for method in METHODS:
        for seed in range(3):
            yield method, seed


def final_checkpoint(repo, method, seed):
    name = shared.training_name("spread3", method, seed)
    prefix = f"{name}_seed{seed}_pz-mpe-simple-spread_"
    candidates = []
    for parent in (repo / "results" / "models").glob(prefix + "*"):
        if not parent.is_dir():
            continue
        for step_dir in parent.iterdir():
            if (step_dir.is_dir() and step_dir.name.isdigit()
                    and int(step_dir.name) >= 2000000
                    and all((step_dir / file).is_file()
                            for file in shared.EXPECTED_FILES[method])):
                candidates.append((parent, int(step_dir.name)))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{name}: expected one complete 2M checkpoint, found "
            f"{len(candidates)}: {candidates}"
        )
    return candidates[0]


def prepare(repo, selected):
    shared.check_evaluation_loop(repo)
    required = (
        repo / "src" / "envs" / "task_metrics.py",
        repo / "src" / "envs" / "task_metric_wrapper.py",
        repo / "src" / "prompts" / "env_code" / "mpe" /
        "processed_obs_mpe_simple_spread_v3_3.py",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Spread3 task-metric dependency missing: {path}")

    jobs = []
    for method, seed in selected:
        run_dir, config = shared.training_run(repo, "spread3", method, seed)
        checkpoint_dir, step = final_checkpoint(repo, method, seed)
        if (config.get("use_llm") is not False
                or config.get("use_knowledge_actions") is not False
                or config.get("use_difference_credit") is not False):
            raise RuntimeError(f"{run_dir}: not a pure-MARL training configuration")
        if method == "mappo" and not config.get("use_rnn"):
            raise RuntimeError(f"{run_dir}: MAPPO RNN setting differs from training")
        jobs.append(("spread3", method, seed, run_dir, config, checkpoint_dir, step))
        print(
            f"[CHECK] {method} seed={seed} run={run_dir} "
            f"checkpoint={checkpoint_dir / str(step)}",
            flush=True,
        )
    return jobs


def metric_record(job, eval_dir, episodes, test_seed):
    _, method, seed, train_dir, _, parent, step = job
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        metrics_path = eval_dir / "info.json"
    data = shared.read_json(metrics_path)
    row = {
        "task": "spread3", "method": method, "train_seed": seed,
        "test_seed": test_seed, "episodes": episodes,
        "training_run": str(train_dir), "evaluation_run": str(eval_dir),
        "checkpoint": str(parent / str(step)), "checkpoint_step": step,
    }
    for metric in METRICS:
        item = data.get(metric, {})
        values = item.get("values", []) if isinstance(item, dict) else item
        if len(values) != 1 or not math.isfinite(float(values[0])):
            raise RuntimeError(
                f"{eval_dir}: expected one finite 1000-episode value for {metric}, "
                f"got {values}"
            )
        row[metric] = float(values[0])
    return row


def save_summary(output_dir, rows):
    shared.write_csv(output_dir / "model_evaluations.csv", rows, list(rows[0]))
    summary = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        item = {"task": "spread3", "method": method, "n_train_seeds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(item)
    shared.write_csv(output_dir / "three_seed_summary.csv", summary, list(summary[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", default="selected_pure_spread3_1000ep_eval_summary"
    )
    parser.add_argument("--only", help="Optional method:seed, e.g. mappo:1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes % 10:
        parser.error("--episodes must be a positive multiple of 10 for MAPPO")

    selected = list(selections())
    if args.only:
        selected = [job for job in selected if ":".join(map(str, job)) == args.only]
        if not selected:
            parser.error("--only does not match one of the nine jobs")

    repo = Path(__file__).resolve().parent.parent
    jobs = prepare(repo, selected)
    print(f"[READY] {len(jobs)} Spread3 evaluations, {args.episodes} episodes each", flush=True)
    output_dir = repo / args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, job in enumerate(jobs, 1):
        _, method, seed, _, _, checkpoint_dir, _ = job
        name, command = shared.command(job, args.episodes, args.test_seed)
        print(f"[EVAL {index}/{len(jobs)}] {method} train_seed={seed}", flush=True)
        print("[CMD] " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        completed = shared.evaluation_run(
            repo, name, checkpoint_dir, args.episodes, args.test_seed
        )
        if completed is None:
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            log_path = output_dir / (name + ".log")
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command, cwd=repo, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{name}: exit={result.returncode}; see {log_path}")
            completed = shared.evaluation_run(
                repo, name, checkpoint_dir, args.episodes, args.test_seed
            )
            if completed is None:
                raise RuntimeError(f"{name}: no completed Sacred evaluation; see {log_path}")
        else:
            print(f"[SKIP] already completed: {completed}", flush=True)

        row = metric_record(job, completed, args.episodes, args.test_seed)
        rows.append(row)
        save_summary(output_dir, rows)
        print(
            f"[OK] return={row['test_return_mean']:.6f} "
            f"success={row['test_success_rate_mean']:.6f}",
            flush=True,
        )
    if not args.dry_run:
        print(f"[SUMMARY] {len(rows)} completed; {output_dir}", flush=True)


if __name__ == "__main__":
    main()
