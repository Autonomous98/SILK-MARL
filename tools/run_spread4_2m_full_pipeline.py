import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_MODELS = "qwen3.7-flash,qwen3.7-plus,qwen3.8-flash"
ENV_NAME = "mpe_simple_spread_v3_4"
ENV_KEY = "pz-mpe-simple-spread-4"
N_AGENTS = 4
TIME_LIMIT = 25

METRIC_ALIASES = {
    "test_return": ["test_return_mean"],
    "success_rate": ["test_success_rate_mean", "success_rate_mean", "test_success_rate"],
    "mean_landmark_dist": ["test_mean_landmark_dist_mean", "mean_landmark_dist_mean", "test_mean_landmark_dist"],
    "max_landmark_dist": ["test_max_landmark_dist_mean", "max_landmark_dist_mean", "test_max_landmark_dist"],
}

METHOD_LABELS = {
    "baseline": "YOLO-MARL Baseline",
    "exp1_exp3": "Exp1 + Exp3",
    "exp2_exp3": "Exp2 + Exp3",
    "exp1_exp2_exp3": "Exp1 + Exp2 + Exp3",
}


def run(cmd, dry_run=False, allow_fail=False):
    print("[CMD] " + " ".join(str(x) for x in cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0:
        print(f"[WARN] command failed with return code {result.returncode}")
    return result.returncode


def safe_model_name(model_name):
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def sacred_root(repo):
    return repo / "results" / "sacred" / "qmix" / "pz-mpe-simple-spread"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(run_dir):
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)
    info_path = run_dir / "info.json"
    if info_path.exists():
        info = read_json(info_path)
        metrics = {}
        for key, values in info.items():
            if key.endswith("_T") or not isinstance(values, list):
                continue
            steps = info.get(key + "_T") or []
            metrics[key] = {"steps": steps, "values": values}
        return metrics
    return {}


def find_metric_key(metrics, logical_name):
    for key in METRIC_ALIASES.get(logical_name, [logical_name]):
        if key in metrics and isinstance(metrics[key], dict) and metrics[key].get("values"):
            return key
    return None


def metric_series(run_dir, logical_name):
    metrics = load_metrics(run_dir)
    key = find_metric_key(metrics, logical_name)
    if key is None:
        return []
    raw = metrics[key]
    steps = raw.get("steps") or []
    values = raw.get("values") or []
    rows = []
    for idx, value in enumerate(values):
        step = steps[idx] if idx < len(steps) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({"step": int(step) if step is not None else idx, "value": value, "source_key": key})
    return rows


def final_metric(run_dir, metric_name="test_return_mean"):
    metrics = load_metrics(run_dir)
    aliases = [metric_name]
    if metric_name in METRIC_ALIASES:
        aliases = METRIC_ALIASES[metric_name]
    for key in aliases:
        metric = metrics.get(key)
        if isinstance(metric, dict) and metric.get("values"):
            steps = metric.get("steps") or []
            values = metric.get("values") or []
            return {
                "step": int(steps[-1]) if steps else None,
                "value": float(values[-1]),
                "source_key": key,
            }
    return None


def final_all_metrics(run_dir):
    out = {}
    for logical_name in METRIC_ALIASES:
        metric = final_metric(run_dir, logical_name)
        if metric is not None:
            out[f"final_{logical_name}"] = metric["value"]
            out[f"final_{logical_name}_step"] = metric["step"]
            out[f"{logical_name}_source_key"] = metric["source_key"]
    return out


def find_run_by_name(repo, name):
    root = sacred_root(repo)
    if not root.exists():
        return None
    matches = []
    for config_path in root.glob("*/config.json"):
        try:
            cfg = read_json(config_path)
        except Exception:
            continue
        if cfg.get("name") == name:
            matches.append(config_path.parent)
    if not matches:
        return None

    def sort_key(path):
        try:
            return int(path.name)
        except ValueError:
            return -1

    return sorted(matches, key=sort_key)[-1]


def build_training_cmd(name, seed, t_max, test_interval, use_knowledge_actions, use_difference_credit, candidate_id, use_cuda, save_model=False, save_model_interval=500000):
    cmd = [
        sys.executable,
        "src/main.py",
        "--config=qmix",
        "--env-config=gymma",
        "with",
        f"env_args.time_limit={TIME_LIMIT}",
        f"env_args.key={ENV_KEY}",
        f"env_args.N={N_AGENTS}",
        f"env_name={ENV_NAME}",
        "use_llm=True",
        f"use_knowledge_actions={str(use_knowledge_actions)}",
        f"use_difference_credit={str(use_difference_credit)}",
        f"knowledge_candidate_id={candidate_id}",
        f"seed={seed}",
        f"t_max={t_max}",
        f"test_interval={test_interval}",
        "test_nepisode=10",
        f"log_interval={test_interval}",
        f"runner_log_interval={test_interval}",
        f"learner_log_interval={test_interval}",
        f"save_model={str(save_model)}",
        f"save_model_interval={save_model_interval}",
        "use_wandb=False",
        f"use_cuda={str(use_cuda)}",
        f"name={name}",
    ]
    return cmd


def generate_candidates(args):
    if args.skip_generation:
        print("[INFO] skip candidate generation")
        return
    cmd = [
        sys.executable,
        "tools/run_exp3_multi_llm_candidates.py",
        "--env", ENV_NAME,
        "--n-agents", str(N_AGENTS),
        "--t-max", str(args.screen_t_max),
        "--test-interval", str(args.screen_interval),
        "--models", args.models,
    ]
    if args.no_fallback_on_api_error:
        cmd.append("--no-fallback-on-api-error")
    run(cmd, dry_run=args.dry_run)


def verify_candidates(args):
    if args.skip_verifier:
        print("[INFO] skip candidate verifier")
        return
    cmd = [
        sys.executable,
        "tools/verify_exp3_candidates.py",
        "--env", ENV_NAME,
        "--n-agents", str(N_AGENTS),
        "--candidates", "0,1,2",
        "--verifier-model", args.verifier_model,
        "--candidate-models", args.models,
        "--semantic-repair-attempts", str(args.semantic_repair_attempts),
    ]
    if args.run_llm_verifier:
        cmd.append("--run-llm")
    if args.allow_unverified_candidates:
        cmd.append("--allow-unverified")
    run(cmd, dry_run=args.dry_run, allow_fail=args.allow_unverified_candidates)


def screening_name(candidate_id, model_name, t_max, seed):
    return f"qmix_exp3_screen_{safe_model_name(model_name)}_candidate{candidate_id}_{ENV_NAME}_{t_max}_seed{seed}"


def run_screening(args, repo, models):
    records = []
    if args.skip_screening:
        print("[INFO] skip small-step candidate screening")
        return records
    for candidate_id, model_name in enumerate(models):
        name = screening_name(candidate_id, model_name, args.screen_t_max, args.screen_seed)
        cmd = build_training_cmd(
            name=name,
            seed=args.screen_seed,
            t_max=args.screen_t_max,
            test_interval=args.screen_interval,
            use_knowledge_actions=True,
            use_difference_credit=True,
            candidate_id=candidate_id,
            use_cuda=args.use_cuda,
            save_model=False,
            save_model_interval=args.save_model_interval,
        )
        rc = run(cmd, dry_run=args.dry_run, allow_fail=True)
        record = {
            "candidate_id": candidate_id,
            "model": model_name,
            "name": name,
            "returncode": rc,
            "final_test_return_mean": None,
            "final_step": None,
            "run_id": None,
        }
        if not args.dry_run:
            run_dir = find_run_by_name(repo, name)
            if run_dir is not None:
                metric = final_metric(run_dir, "test_return")
                record["run_id"] = run_dir.name
                if metric is not None:
                    record["final_test_return_mean"] = metric["value"]
                    record["final_step"] = metric["step"]
                    print(f"[SCREEN] candidate{candidate_id} model={model_name} run_id={run_dir.name} final={metric['value']}")
                else:
                    print(f"[SCREEN] candidate{candidate_id} model={model_name} has no final test_return")
        records.append(record)
    return records


def choose_best_candidate(records, explicit_candidate=None):
    if explicit_candidate is not None:
        print(f"[INFO] use explicit best candidate: {explicit_candidate}")
        return explicit_candidate
    scored = [r for r in records if r.get("final_test_return_mean") is not None and r.get("returncode") == 0]
    if not scored:
        raise SystemExit("No successful screening result found. Re-run without --skip-screening or pass --best-candidate.")
    best = max(scored, key=lambda r: r["final_test_return_mean"])
    print(f"[BEST] candidate{best['candidate_id']} model={best['model']} final={best['final_test_return_mean']}")
    return int(best["candidate_id"])


def formal_experiments(best_candidate):
    return [
        {"tag": "baseline", "method": "baseline", "use_knowledge_actions": False, "use_difference_credit": False, "candidate_id": -1},
        {"tag": "exp1_exp3", "method": "knowledge_action_exp3", "use_knowledge_actions": True, "use_difference_credit": False, "candidate_id": best_candidate},
        {"tag": "exp2_exp3", "method": "difference_credit_exp3", "use_knowledge_actions": False, "use_difference_credit": True, "candidate_id": best_candidate},
        {"tag": "exp1_exp2_exp3", "method": "knowledge_action_difference_credit_exp3", "use_knowledge_actions": True, "use_difference_credit": True, "candidate_id": best_candidate},
    ]


def formal_name(exp, seed, t_max):
    if exp["tag"] == "baseline":
        return f"qmix_baseline_simple_spread4_{t_max}_seed{seed}_formal"
    return f"qmix_{exp['tag']}_simple_spread4_{t_max}_candidate{exp['candidate_id']}_seed{seed}_formal"


def run_formal(args, repo, best_candidate):
    records = []
    for exp in formal_experiments(best_candidate):
        for seed in args.seeds:
            name = formal_name(exp, seed, args.formal_t_max)
            cmd = build_training_cmd(
                name=name,
                seed=seed,
                t_max=args.formal_t_max,
                test_interval=args.formal_interval,
                use_knowledge_actions=exp["use_knowledge_actions"],
                use_difference_credit=exp["use_difference_credit"],
                candidate_id=exp["candidate_id"],
                use_cuda=args.use_cuda,
                save_model=args.save_model,
                save_model_interval=args.save_model_interval,
            )
            rc = run(cmd, dry_run=args.dry_run, allow_fail=args.continue_on_error)
            record = {
                "tag": exp["tag"],
                "method": exp["method"],
                "seed": seed,
                "name": name,
                "candidate_id": exp["candidate_id"],
                "returncode": rc,
                "run_id": None,
            }
            if not args.dry_run:
                run_dir = find_run_by_name(repo, name)
                if run_dir is not None:
                    record["run_id"] = run_dir.name
                    record.update(final_all_metrics(run_dir))
                    if "final_test_return" in record:
                        print(f"[FORMAL] {exp['tag']} seed={seed} run_id={run_dir.name} final_return={record['final_test_return']}")
                    else:
                        print(f"[FORMAL] {exp['tag']} seed={seed} run_id={run_dir.name} has no test_return")
            records.append(record)
    return records


def collect_long_rows(repo, formal_records):
    rows = []
    for record in formal_records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        run_dir = sacred_root(repo) / str(run_id)
        for metric in METRIC_ALIASES:
            for point in metric_series(run_dir, metric):
                rows.append({
                    "tag": record["tag"],
                    "method": record["method"],
                    "method_label": METHOD_LABELS.get(record["tag"], record["tag"]),
                    "seed": record["seed"],
                    "candidate_id": record["candidate_id"],
                    "run_id": run_id,
                    "metric": metric,
                    "source_key": point["source_key"],
                    "step": point["step"],
                    "value": point["value"],
                    "name": record["name"],
                })
    return rows


def aggregate_by_step(long_rows):
    grouped = defaultdict(list)
    for row in long_rows:
        grouped[(row["tag"], row["method"], row["method_label"], row["metric"], row["step"])].append(row["value"])
    out = []
    for (tag, method, method_label, metric, step), values in sorted(grouped.items(), key=lambda item: (item[0][3], item[0][0], item[0][4])):
        out.append({
            "tag": tag,
            "method": method,
            "method_label": method_label,
            "metric": metric,
            "step": step,
            "n_seeds": len(values),
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        })
    return out


def write_csv(path, rows, fieldnames):
    rows = list(rows)
    fieldnames = list(dict.fromkeys(fieldnames))
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def final_table_rows(formal_records):
    rows = []
    grouped = defaultdict(list)
    for record in formal_records:
        for metric in METRIC_ALIASES:
            key = f"final_{metric}"
            if key in record and record[key] is not None:
                grouped[(record["tag"], record["method"], metric)].append((record["seed"], record[key]))
    for (tag, method, metric), seed_values in sorted(grouped.items()):
        values_by_seed = {seed: value for seed, value in seed_values}
        values = [value for _, value in sorted(seed_values)]
        row = {
            "tag": tag,
            "method": method,
            "method_label": METHOD_LABELS.get(tag, tag),
            "metric": metric,
            "seed0": values_by_seed.get(0),
            "seed1": values_by_seed.get(1),
            "seed2": values_by_seed.get(2),
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
        rows.append(row)
    return rows


def fmt(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def write_markdown_table(path, rows):
    lines = ["| Experiment | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Std | n |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {row['metric']} | {fmt(row.get('seed0'))} | {fmt(row.get('seed1'))} | {fmt(row.get('seed2'))} | {fmt(row.get('mean'))} | {fmt(row.get('std'))} | {row.get('n_seeds', '')} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_curves(out_dir, mean_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] skip plots, missing matplotlib: {exc}")
        return

    colors = {
        "baseline": "#2f73c9",
        "exp1_exp3": "#1fa463",
        "exp2_exp3": "#db7c26",
        "exp1_exp2_exp3": "#8b5cc7",
    }
    ylabels = {
        "test_return": "Test return mean",
        "success_rate": "Success rate",
        "mean_landmark_dist": "Mean landmark distance",
        "max_landmark_dist": "Max landmark distance",
    }
    titles = {
        "test_return": "4-Agent Simple Spread: 2M Test Return",
        "success_rate": "4-Agent Simple Spread: Success Rate",
        "mean_landmark_dist": "4-Agent Simple Spread: Mean Landmark Distance",
        "max_landmark_dist": "4-Agent Simple Spread: Max Landmark Distance",
    }
    for metric in METRIC_ALIASES:
        rows = [r for r in mean_rows if r["metric"] == metric]
        if not rows:
            continue
        plt.figure(figsize=(12, 6), dpi=160)
        for tag in ["baseline", "exp1_exp3", "exp2_exp3", "exp1_exp2_exp3"]:
            series = [r for r in rows if r["tag"] == tag]
            if not series:
                continue
            series.sort(key=lambda r: r["step"])
            xs = [r["step"] / 1_000_000 for r in series]
            ys = [r["mean"] for r in series]
            mins = [r["min"] for r in series]
            maxs = [r["max"] for r in series]
            label = METHOD_LABELS.get(tag, tag)
            color = colors.get(tag)
            plt.plot(xs, ys, marker="o", linewidth=2.2, markersize=3.5, label=label, color=color)
            plt.fill_between(xs, mins, maxs, alpha=0.14, color=color)
        plt.title(titles.get(metric, metric), fontsize=15, fontweight="bold")
        plt.xlabel("Environment steps (M)")
        plt.ylabel(ylabels.get(metric, metric))
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(frameon=False)
        plt.tight_layout()
        output = out_dir / f"spread4_2m_{metric}_mean_minmax_curve.png"
        plt.savefig(output)
        plt.close()
        print(f"[OK] wrote plot: {output}")


def write_summary(repo, args, models, screening_records, best_candidate, formal_records):
    out_dir = repo / "selected_spread4_2m_exp3_pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "env": ENV_NAME,
        "env_key": ENV_KEY,
        "n_agents": N_AGENTS,
        "models": models,
        "screen_t_max": args.screen_t_max,
        "formal_t_max": args.formal_t_max,
        "best_candidate": best_candidate,
        "screening": screening_records,
        "formal_runs": formal_records,
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    formal_fields = [
        "tag", "method", "seed", "candidate_id", "run_id", "returncode",
        "final_test_return", "final_success_rate", "final_mean_landmark_dist", "final_max_landmark_dist", "name",
    ]
    write_csv(out_dir / "formal_final_results.csv", formal_records, formal_fields)

    long_rows = collect_long_rows(repo, formal_records) if not args.dry_run else []
    long_fields = ["tag", "method", "method_label", "seed", "candidate_id", "run_id", "metric", "source_key", "step", "value", "name"]
    write_csv(out_dir / "formal_metrics_long.csv", long_rows, long_fields)

    mean_rows = aggregate_by_step(long_rows)
    mean_fields = ["tag", "method", "method_label", "metric", "step", "n_seeds", "mean", "min", "max", "std"]
    write_csv(out_dir / "formal_metrics_mean_by_step.csv", mean_rows, mean_fields)

    table_rows = final_table_rows(formal_records)
    table_fields = ["tag", "method", "method_label", "metric", "seed0", "seed1", "seed2", "mean", "std", "n_seeds"]
    write_csv(out_dir / "formal_final_three_line_table.csv", table_rows, table_fields)
    write_markdown_table(out_dir / "formal_final_three_line_table.md", table_rows)

    plot_metric_curves(out_dir, mean_rows)
    print(f"[OK] wrote summary to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Simple-Spread 4-agent Exp3 screening and formal 2M experiments.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Three comma-separated models for candidate generation.")
    parser.add_argument("--screen-t-max", type=int, default=200000)
    parser.add_argument("--screen-interval", type=int, default=50000)
    parser.add_argument("--screen-seed", type=int, default=0)
    parser.add_argument("--formal-t-max", type=int, default=2000000)
    parser.add_argument("--formal-interval", type=int, default=50000)
    parser.add_argument("--seeds", default="0,1,2", help="Formal seeds, comma-separated.")
    parser.add_argument("--best-candidate", type=int, default=None, help="Skip automatic selection and use this candidate id.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-verifier", action="store_true")
    parser.add_argument("--run-llm-verifier", action="store_true")
    parser.add_argument("--verifier-model", default="qwen3.8-max")
    parser.add_argument(
        "--semantic-repair-attempts",
        type=int,
        default=2,
        help="Maximum verifier-feedback repair rounds for each failed candidate.",
    )
    parser.add_argument(
        "--allow-unverified-candidates",
        action="store_true",
        help="Continue to screening even if local/LLM candidate verification does not pass.",
    )
    parser.add_argument("--skip-screening", action="store_true")
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument("--no-fallback-on-api-error", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True, help="Save checkpoints for the formal 2M runs.")
    parser.add_argument("--save-model-interval", type=int, default=500000, help="Checkpoint interval for formal runs.")
    parser.add_argument("--use-cuda", default="True", choices=["True", "False", "true", "false"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    args = parser.parse_args()
    args.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    repo = Path.cwd()
    if not (repo / "src" / "main.py").exists():
        raise SystemExit("Please run this script from the YOLO-MARL repository root.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 3:
        raise SystemExit("Experiment 3 expects exactly three models/candidates. Use --models m0,m1,m2.")

    generate_candidates(args)
    verify_candidates(args)
    screening_records = run_screening(args, repo, models)
    if args.dry_run and args.best_candidate is None:
        best_candidate = 0
        print("[DRY-RUN] no screening metrics are produced; use candidate0 only for printing formal commands")
    else:
        best_candidate = choose_best_candidate(screening_records, args.best_candidate)
    formal_records = []
    if not args.skip_formal:
        formal_records = run_formal(args, repo, best_candidate)
    if not args.dry_run:
        write_summary(repo, args, models, screening_records, best_candidate, formal_records)
    else:
        print("[DRY-RUN] summary files are not written")


if __name__ == "__main__":
    main()


