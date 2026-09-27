"""Evaluate 15 saved Spread3/Spread4 models for 1000 episodes each.

Run from the repository root after copying this file into tools/.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


TASKS = {
    "spread3": (3, "mpe_simple_spread_v3_3", "pz-mpe-simple-spread-3"),
    "spread4": (4, "mpe_simple_spread_v3_4", "pz-mpe-simple-spread-4"),
}
METRICS = (
    "test_return_mean",
    "test_success_rate_mean",
    "test_mean_landmark_dist_mean",
    "test_max_landmark_dist_mean",
)
EXPECTED_FILES = {
    "qmix": ("agent.th", "mixer.th", "opt.th"),
    "maddpg": ("agent.th", "critic.th", "agent_opt.th", "critic_opt.th"),
    "mappo": ("agent.th", "critic.th", "agent_opt.th", "critic_opt.th"),
}


def jobs():
    for method in ("yolo", "ours"):
        for seed in range(3):
            yield "spread3", method, seed
    for method in ("qmix", "maddpg", "mappo"):
        for seed in range(3):
            yield "spread4", method, seed


def training_name(task, method, seed):
    if method == "yolo":
        return f"qmix_baseline_simple_spread3_2000000_seed{seed}_formal"
    if method == "ours":
        return (
            "qmix_exp1_exp2_exp3_simple_spread3_2000000_"
            f"candidate1_seed{seed}_formal"
        )
    return f"pure_{method}_{task}_2000000_interval1000_seed{seed}_formal"


def algorithm(method):
    return "qmix" if method in ("yolo", "ours") else method


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def training_run(repo, task, method, seed):
    name = training_name(task, method, seed)
    root = repo / "results" / "sacred" / algorithm(method) / "pz-mpe-simple-spread"
    matches = []
    for path in root.glob("*/config.json"):
        try:
            config = read_json(path)
            status = read_json(path.with_name("run.json")).get("status")
        except (OSError, ValueError):
            continue
        if (config.get("name") == name and config.get("seed") == seed
                and config.get("t_max") == 2000000
                and config.get("test_interval") == 1000
                and status == "COMPLETED"):
            matches.append((path.parent, config))
    if len(matches) != 1:
        raise RuntimeError(f"{name}: expected one completed 1000-step Sacred run, found {len(matches)}")
    return matches[0]


def model_checkpoint(repo, task, method, seed):
    name = training_name(task, method, seed)
    prefix = f"{name}_seed{seed}_pz-mpe-simple-spread_"
    model_root = repo / "results" / "models"
    directories = [p for p in model_root.glob(prefix + "*") if p.is_dir()]
    if method in ("yolo", "ours"):
        training_day = "2026-09-18" if method == "yolo" else "2026-09-19"
        directories = [p for p in directories if p.name[len(prefix):].startswith(training_day)]
    if len(directories) != 1:
        raise RuntimeError(f"{name}: expected one matching model directory, found {len(directories)}")
    parent = directories[0]
    steps = sorted(int(p.name) for p in parent.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps or steps[-1] < 2000000:
        raise RuntimeError(f"{parent}: no final 2M checkpoint")
    step = steps[-1]
    missing = [file for file in EXPECTED_FILES[algorithm(method)] if not (parent / str(step) / file).is_file()]
    if missing:
        raise RuntimeError(f"{parent / str(step)}: missing {missing}")
    return parent, step


def generated_planner(repo):
    code_dir = repo / "src" / "prompts" / "gen_code" / "mpe_simple_spread_v3_3" / "code"
    choices = []
    for path in code_dir.glob("*_generated_code_*.py"):
        match = re.search(r"_generated_code_(\d+)\.py$", path.name)
        if match:
            choices.append((int(match.group(1)), path.stat().st_mtime, path))
    if not choices:
        raise RuntimeError(f"Spread3 generated planner/reward code not found: {code_dir}")
    selected = max(choices)[2]
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    print(f"[PLANNER] {selected} sha256={digest}", flush=True)
    for kind in ("knowledge_action_defs", "knowledge_actions", "difference_credit"):
        path = (repo / "src" / "prompts" / "env_code" / "mpe" /
                f"{kind}_mpe_simple_spread_v3_3_candidate1.py")
        if not path.is_file():
            raise RuntimeError(f"Ours candidate1 module missing: {path}")
    return str(selected), digest


def check_evaluation_loop(repo):
    source = (repo / "src" / "run.py").read_text(encoding="utf-8")
    section = source.split("def evaluate_sequential(args, runner):", 1)[-1].split("def run_sequential", 1)[0]
    if "args.test_nepisode // runner.batch_size" not in section:
        raise RuntimeError("src/run.py lacks parallel-runner episode-count fix; MAPPO would run 10000 episodes")


def prepare(repo, selected):
    check_evaluation_loop(repo)
    if any(method in ("yolo", "ours") for _, method, _ in selected):
        generated_planner(repo)
    if any(task == "spread4" for task, _, _ in selected):
        for filename in ("task_metrics.py", "task_metric_wrapper.py"):
            if not (repo / "src" / "envs" / filename).is_file():
                raise RuntimeError(f"Missing pure-MARL task metric wrapper: {filename}")
    prepared = []
    for task, method, seed in selected:
        run_dir, config = training_run(repo, task, method, seed)
        parent, step = model_checkpoint(repo, task, method, seed)
        if config.get("use_llm") != (method in ("yolo", "ours")):
            raise RuntimeError(f"Training method mismatch: {run_dir}")
        if method == "ours" and (config.get("knowledge_candidate_id") != 1 or
                                 not config.get("use_knowledge_actions") or
                                 not config.get("use_difference_credit")):
            raise RuntimeError(f"Ours candidate/config mismatch: {run_dir}")
        if method == "mappo" and not config.get("use_rnn"):
            raise RuntimeError(f"MAPPO training architecture does not match expected RNN: {run_dir}")
        prepared.append((task, method, seed, run_dir, config, parent, step))
    return prepared


def sacred_value(value):
    if isinstance(value, bool):
        return str(value)
    return str(value)


def command(job, episodes, test_seed):
    task, method, seed, run_dir, config, parent, step = job
    n_agents, env_name, env_key = TASKS[task]
    eval_name = f"eval1000_{task}_{method}_trainseed{seed}_testseed{test_seed}_{episodes}ep"
    cmd = [sys.executable, "src/main.py", f"--config={algorithm(method)}",
           "--env-config=gymma", "with", "env_args.time_limit=25",
           f"env_args.key={env_key}", f"env_args.N={n_agents}",
           f"env_name={env_name}", f"seed={test_seed}",
           f"checkpoint_path={parent}", f"load_step={step}",
           "evaluate=True", f"test_nepisode={episodes}", "test_greedy=True",
           "render=False", "use_wandb=False", "use_cuda=True",
           f"name={eval_name}"]
    for key in ("runner", "batch_size_run", "agent", "mac", "hidden_dim",
                "use_rnn", "obs_agent_id", "obs_last_action", "obs_individual_obs",
                "common_reward", "reward_scalarisation", "simple_spread_success_threshold"):
        if key in config:
            cmd.append(f"{key}={sacred_value(config[key])}")
    if method in ("yolo", "ours"):
        cmd.extend(("use_llm=True", f"use_knowledge_actions={method == 'ours'}",
                    f"use_difference_credit={method == 'ours'}",
                    f"knowledge_candidate_id={1 if method == 'ours' else -1}"))
    else:
        cmd.extend(("use_llm=False", "use_knowledge_actions=False",
                    "use_difference_credit=False", "knowledge_candidate_id=-1"))
    if method == "mappo":
        for key in ("lr", "q_nstep", "standardise_rewards"):
            cmd.append(f"{key}={sacred_value(config[key])}")
    return eval_name, cmd


def evaluation_run(repo, name, checkpoint, episodes, test_seed):
    root = repo / "results" / "sacred"
    for path in sorted(root.rglob("config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            config = read_json(path)
            if (config.get("name") == name and config.get("checkpoint_path") == str(checkpoint)
                    and config.get("test_nepisode") == episodes and config.get("seed") == test_seed
                    and config.get("evaluate") is True
                    and read_json(path.with_name("run.json")).get("status") == "COMPLETED"):
                return path.parent
        except (OSError, ValueError):
            continue
    return None


def metric_record(job, eval_dir, episodes, test_seed):
    task, method, seed, train_dir, config, parent, step = job
    metrics_path = eval_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else read_json(eval_dir / "info.json")
    row = {"task": task, "method": method, "train_seed": seed,
           "test_seed": test_seed, "episodes": episodes,
           "training_run": str(train_dir), "evaluation_run": str(eval_dir),
           "checkpoint": str(parent / str(step)), "checkpoint_step": step}
    for metric in METRICS:
        item = metrics.get(metric, {})
        values = item.get("values", []) if isinstance(item, dict) else item
        if len(values) != 1 or not math.isfinite(float(values[0])):
            raise RuntimeError(f"{eval_dir}: expected one finite 1000-episode value for {metric}, got {values}")
        row[metric] = float(values[0])
    return row


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(out_dir, rows):
    if not rows:
        return
    write_csv(out_dir / "model_evaluations.csv", rows, list(rows[0]))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["method"]), []).append(row)
    summary = []
    for (task, method), group in sorted(grouped.items()):
        item = {"task": task, "method": method, "n_train_seeds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(item)
    write_csv(out_dir / "three_seed_summary.csv", summary, list(summary[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="selected_15_model_1000ep_eval_summary")
    parser.add_argument("--only", help="Optional task:method:seed, e.g. spread4:mappo:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes % 10:
        parser.error("--episodes must be a positive multiple of 10 for MAPPO's parallel runner")
    repo = Path(__file__).resolve().parent.parent
    selected = list(jobs())
    if args.only:
        selected = [job for job in selected if ":".join(map(str, job)) == args.only]
        if not selected:
            parser.error("--only does not match one of the 15 jobs")
    prepared = prepare(repo, selected)
    print(f"[READY] {len(prepared)} model evaluations; {args.episodes} episodes each", flush=True)
    out_dir = repo / args.output_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, job in enumerate(prepared, 1):
        task, method, seed, _, _, parent, _ = job
        name, cmd = command(job, args.episodes, args.test_seed)
        print(f"[EVAL {index}/{len(prepared)}] {task} {method} train_seed={seed}", flush=True)
        print("[CMD] " + subprocess.list2cmdline(cmd), flush=True)
        if args.dry_run:
            continue
        found = evaluation_run(repo, name, parent, args.episodes, args.test_seed)
        if found is None:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            log_path = out_dir / (name + ".log")
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(cmd, cwd=repo, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name}: exit={result.returncode}; see {log_path}")
            found = evaluation_run(repo, name, parent, args.episodes, args.test_seed)
            if found is None:
                raise RuntimeError(f"{name}: process exited but no matching completed Sacred run; see {log_path}")
        else:
            print(f"[SKIP] already completed: {found}", flush=True)
        row = metric_record(job, found, args.episodes, args.test_seed)
        rows.append(row)
        save_summary(out_dir, rows)
        print(f"[OK] return={row['test_return_mean']:.6f} success={row['test_success_rate_mean']:.6f}", flush=True)
    if not args.dry_run:
        print(f"[SUMMARY] {len(rows)} completed; {out_dir}", flush=True)


if __name__ == "__main__":
    main()
