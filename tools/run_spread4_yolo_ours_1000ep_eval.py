"""Evaluate three Spread4 YOLO-MARL and three Ours models for 1000 episodes.

Run from the YOLO-MARL root after copying this script into tools/. Requires
run_saved_model_15x1000_eval.py in the same directory.
"""

import argparse
import hashlib
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

import run_saved_model_15x1000_eval as shared


ENV_NAME = "mpe_simple_spread_v3_4"
ENV_KEY = "pz-mpe-simple-spread-4"
SACRED_ENV_KEY = "pz-mpe-simple-spread"
CHECKPOINT_FILES = ("agent.th", "mixer.th", "opt.th")
METRICS = (
    "test_return_mean",
    "test_success_rate_mean",
    "test_mean_landmark_dist_mean",
)


def selections():
    for method in ("yolo", "ours"):
        for seed in range(3):
            yield method, seed


def training_name(method, seed):
    if method == "yolo":
        return f"qmix_baseline_simple_spread4_2000000_seed{seed}_formal"
    return (
        "qmix_exp1_exp2_exp3_simple_spread4_2000000_"
        f"candidate2_seed{seed}_formal"
    )


def training_run(repo, method, seed):
    name = training_name(method, seed)
    root = repo / "results" / "sacred" / "qmix" / SACRED_ENV_KEY
    matches = []
    for path in root.glob("*/config.json"):
        try:
            config = shared.read_json(path)
            status = shared.read_json(path.with_name("run.json")).get("status")
        except (OSError, ValueError):
            continue
        if (config.get("name") == name and config.get("seed") == seed
                and config.get("t_max") == 2000000
                and config.get("test_interval") == 1000
                and config.get("env_name") == ENV_NAME
                and status == "COMPLETED"):
            matches.append((path.parent, config))
    if len(matches) != 1:
        raise RuntimeError(
            f"{name}: expected one completed 1000-step Sacred run, found {len(matches)}"
        )
    run_dir, config = matches[0]
    expected_candidate = -1 if method == "yolo" else 2
    expected_modules = method == "ours"
    if (config.get("use_llm") is not True
            or config.get("use_knowledge_actions") != expected_modules
            or config.get("use_difference_credit") != expected_modules
            or config.get("knowledge_candidate_id") != expected_candidate):
        raise RuntimeError(f"{run_dir}: method/candidate flags do not match {method}")
    return run_dir, config


def final_checkpoint(repo, method, seed):
    name = training_name(method, seed)
    prefix = f"{name}_seed{seed}_{SACRED_ENV_KEY}_"
    expected_day = "2026-09-19" if method == "yolo" else "2026-09-20"
    candidates = []
    for parent in (repo / "results" / "models").glob(prefix + "*"):
        if not parent.is_dir() or not parent.name[len(prefix):].startswith(expected_day):
            continue
        for step_dir in parent.iterdir():
            if (step_dir.is_dir() and step_dir.name.isdigit()
                    and int(step_dir.name) >= 2000000
                    and all((step_dir / file).is_file() for file in CHECKPOINT_FILES)):
                candidates.append((parent, int(step_dir.name)))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{name}: expected one complete 2M checkpoint from {expected_day}, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def check_generated_code(repo):
    code_dir = repo / "src" / "prompts" / "gen_code" / ENV_NAME / "code"
    candidates = []
    for path in code_dir.glob("*_generated_code_*.py"):
        match = re.search(r"_generated_code_(\d+)\.py$", path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        raise RuntimeError(f"Spread4 generated planner/reward code missing: {code_dir}")
    selected = max(candidates)[2]
    print(
        f"[PLANNER] {selected} sha256={hashlib.sha256(selected.read_bytes()).hexdigest()}",
        flush=True,
    )
    for kind in ("knowledge_action_defs", "knowledge_actions", "difference_credit"):
        path = (repo / "src" / "prompts" / "env_code" / "mpe" /
                f"{kind}_{ENV_NAME}_candidate2.py")
        if not path.is_file():
            raise RuntimeError(f"Ours candidate2 module missing: {path}")


def prepare(repo, selected):
    shared.check_evaluation_loop(repo)
    check_generated_code(repo)
    prepared = []
    for method, seed in selected:
        run_dir, config = training_run(repo, method, seed)
        parent, step = final_checkpoint(repo, method, seed)
        prepared.append(("spread4", method, seed, run_dir, config, parent, step))
        print(f"[CHECK] {method} seed={seed} run={run_dir} checkpoint={parent / str(step)}", flush=True)
    return prepared


def evaluation_command(job, episodes, test_seed):
    _, method, seed, _, config, parent, step = job
    name = f"eval1000_spread4_{method}_trainseed{seed}_testseed{test_seed}_{episodes}ep"
    cmd = [
        sys.executable, "src/main.py", "--config=qmix", "--env-config=gymma",
        "with", "env_args.time_limit=25", f"env_args.key={ENV_KEY}",
        "env_args.N=4", f"env_name={ENV_NAME}", f"seed={test_seed}",
        f"checkpoint_path={parent}", f"load_step={step}",
        "evaluate=True", f"test_nepisode={episodes}", "test_greedy=True",
        "render=False", "use_wandb=False", "use_cuda=True", f"name={name}",
    ]
    for key in (
        "runner", "batch_size_run", "agent", "mac", "hidden_dim", "use_rnn",
        "obs_agent_id", "obs_last_action", "obs_individual_obs",
        "common_reward", "reward_scalarisation",
        "simple_spread_success_threshold",
    ):
        if key in config:
            cmd.append(f"{key}={shared.sacred_value(config[key])}")
    cmd.extend((
        "use_llm=True", f"use_knowledge_actions={method == 'ours'}",
        f"use_difference_credit={method == 'ours'}",
        f"knowledge_candidate_id={2 if method == 'ours' else -1}",
    ))
    return name, cmd


def metric_record(job, eval_dir, episodes, test_seed):
    _, method, seed, train_dir, _, parent, step = job
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        metrics_path = eval_dir / "info.json"
    data = shared.read_json(metrics_path)
    row = {
        "task": "spread4", "method": method, "train_seed": seed,
        "test_seed": test_seed, "episodes": episodes,
        "training_run": str(train_dir), "evaluation_run": str(eval_dir),
        "checkpoint": str(parent / str(step)), "checkpoint_step": step,
    }
    for metric in METRICS:
        item = data.get(metric, {})
        values = item.get("values", []) if isinstance(item, dict) else item
        if len(values) != 1 or not math.isfinite(float(values[0])):
            raise RuntimeError(f"{eval_dir}: expected one finite value for {metric}, got {values}")
        row[metric] = float(values[0])
    return row


def save_summary(output_dir, rows):
    shared.write_csv(output_dir / "model_evaluations.csv", rows, list(rows[0]))
    summary = []
    for method in ("yolo", "ours"):
        group = [row for row in rows if row["method"] == method]
        if not group:
            continue
        item = {"task": "spread4", "method": method, "n_train_seeds": len(group)}
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
        "--output-dir", default="selected_spread4_yolo_ours_1000ep_eval_summary"
    )
    parser.add_argument("--only", help="Optional method:seed, e.g. ours:1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    selected = list(selections())
    if args.only:
        selected = [job for job in selected if ":".join(map(str, job)) == args.only]
        if not selected:
            parser.error("--only does not match one of the six jobs")

    repo = Path(__file__).resolve().parent.parent
    prepared = prepare(repo, selected)
    print(f"[READY] {len(prepared)} Spread4 evaluations, {args.episodes} episodes each", flush=True)
    output_dir = repo / args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, job in enumerate(prepared, 1):
        _, method, seed, _, _, parent, _ = job
        name, cmd = evaluation_command(job, args.episodes, args.test_seed)
        print(f"[EVAL {index}/{len(prepared)}] {method} train_seed={seed}", flush=True)
        print("[CMD] " + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        found = shared.evaluation_run(repo, name, parent, args.episodes, args.test_seed)
        if found is None:
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            log_path = output_dir / (name + ".log")
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    cmd, cwd=repo, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                raise RuntimeError(f"{name}: exit={result.returncode}; see {log_path}")
            found = shared.evaluation_run(repo, name, parent, args.episodes, args.test_seed)
            if found is None:
                raise RuntimeError(f"{name}: no completed Sacred evaluation; see {log_path}")
        else:
            print(f"[SKIP] already completed: {found}", flush=True)
        row = metric_record(job, found, args.episodes, args.test_seed)
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
