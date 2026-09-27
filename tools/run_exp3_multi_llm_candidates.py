import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, dry_run=False):
    print("[CMD] " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=True).returncode


def run_allow_fail(cmd):
    print("[CMD] " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] command failed with return code {result.returncode}; continue to next candidate")
    return result.returncode


def env_family(env_name):
    return str(env_name).split("_", 1)[0].lower()


def base_n_actions(env_name):
    return 6 if env_family(env_name) == "lbf" else 5


def infer_env_key(env_name, n_agents):
    family = env_family(env_name)
    if family == "mpe":
        if env_name.startswith("mpe_simple_spread_"):
            suffix = env_name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                return f"pz-mpe-simple-spread-{suffix}"
        return f"pz-mpe-simple-spread-{n_agents}"
    if family == "lbf":
        parts = env_name.split("_")
        # Expected prompt env names look like lbf_2p_2f_coop or lbf_3p_3f_coop.
        player_part = next((p for p in parts if p.endswith("p") and p[:-1].isdigit()), f"{n_agents}p")
        food_part = next((p for p in parts if p.endswith("f") and p[:-1].isdigit()), f"{n_agents}f")
        coop = "-coop" if "coop" in parts else ""
        return f"lbforaging:Foraging-8x8-{player_part}-{food_part}{coop}-v3"
    if family == "rware":
        return "rware:" + env_name.replace("_", "-")
    return env_name


def infer_time_limit(env_name):
    family = env_family(env_name)
    if family == "mpe":
        return 25
    if family == "lbf":
        return 50
    if family == "rware":
        return 500
    return 100


def candidate_defs(candidate_id, env_name, n_agents):
    first_id = base_n_actions(env_name)
    family = env_family(env_name)
    generic_actions = {
        0: [
            ("task_progress_action", "Choose a task-relevant movement or interaction that improves immediate objective progress."),
            ("safe_coordination_action", "Avoid local conflicts or blocking while keeping useful task progress."),
            ("completion_or_hold_action", "Complete an available interaction, or hold when the current state is already useful."),
            ("rebalance_team_action", "Move effort toward under-served entities, targets, teammates, or regions."),
        ],
        1: [
            ("safety_first_progress", "Prioritize safe movement, then continue toward the most useful task entity."),
            ("nearest_unserved_entity", "Select the nearest task entity that still needs attention and move or act toward it."),
            ("yield_or_wait", "Yield, wait, or choose a less contested task opportunity when another agent has priority."),
            ("assist_sparse_region", "Help a sparse or delayed part of the team objective."),
        ],
        2: [
            ("assignment_balanced_progress", "Use agent identity and observed task state to balance assignments across entities."),
            ("conflict_resolving_progress", "Resolve duplicated claims or congestion before continuing task progress."),
            ("stable_completion", "Keep a useful completed/near-completed state stable instead of over-moving."),
            ("global_rebalance", "Choose an alternative target or role that improves global team progress."),
        ],
    }[candidate_id]
    if family == "mpe":
        generic_actions[0] = ("move_to_nearest_uncovered_landmark", "Move toward the nearest landmark that is not already covered.")
    elif family == "lbf":
        generic_actions[0] = ("approach_or_load_food", "Move toward reachable food and load/pickup when the team is ready.")
    elif family == "rware":
        generic_actions[0] = ("serve_requested_shelf", "Move toward, pick, deliver, or return the shelf that best advances warehouse service.")
    while len(generic_actions) < max(4, n_agents):
        idx = len(generic_actions)
        generic_actions.append((f"adaptive_role_{idx}", "Choose a task-adaptive role inferred from processed_obs."))
    lines = ["KNOWLEDGE_ACTIONS = {"]
    for offset, (name, desc) in enumerate(generic_actions[:max(4, n_agents)], start=first_id):
        lines.append(f'    {offset}: {{"name": "{name}", "description": "{desc}"}},')
    lines.append("}\n")
    return "\n".join(lines)


def candidate_actions(candidate_id, env_name, n_agents):
    defs = candidate_defs(candidate_id, env_name, n_agents)
    first_id = base_n_actions(env_name)
    max_primitive = first_id - 1
    return f'''import numpy as np

{defs}
BASE_N_ACTIONS = {first_id}
N_KNOWLEDGE_ACTIONS = len(KNOWLEDGE_ACTIONS)
TOTAL_N_ACTIONS = BASE_N_ACTIONS + N_KNOWLEDGE_ACTIONS
CANDIDATE_MAPPING_MODE = {candidate_id}


def map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08):
    action_id = int(action_id)
    if 0 <= action_id < BASE_N_ACTIONS:
        return action_id
    # Generic safe fallback for non-LLM static candidates. Real Experiment 3 should prefer LLM-generated code.
    if BASE_N_ACTIONS >= 6 and action_id in KNOWLEDGE_ACTIONS and "load" in KNOWLEDGE_ACTIONS[action_id].get("name", ""):
        return min(5, {max_primitive})
    return 0
'''


def candidate_credit(candidate_id):
    return '''import numpy as np


def difference_credit_fn(prev_processed_obs, primitive_actions):
    if isinstance(prev_processed_obs, dict):
        if "agent_infos" in prev_processed_obs and isinstance(prev_processed_obs["agent_infos"], list):
            agent_ids = list(range(len(prev_processed_obs["agent_infos"])))
        else:
            agent_ids = [k for k in prev_processed_obs.keys() if str(k).startswith("agent_")]
            if not agent_ids:
                agent_ids = list(prev_processed_obs.keys())
    elif isinstance(prev_processed_obs, tuple) and len(prev_processed_obs) >= 2 and isinstance(prev_processed_obs[1], dict):
        agent_ids = list(prev_processed_obs[1].keys())
    else:
        try:
            agent_ids = list(range(len(prev_processed_obs)))
        except Exception:
            agent_ids = []
    return {agent_id: 0.0 for agent_id in agent_ids}
'''


def write_static_candidates(env_name, n_agents):
    family = env_family(env_name)
    env_dir = Path("src/prompts/env_code") / family
    env_dir.mkdir(parents=True, exist_ok=True)
    labels = ["static_progress", "static_safety", "static_assignment"]
    for candidate_id, label in enumerate(labels):
        (env_dir / f"knowledge_action_defs_{env_name}_candidate{candidate_id}.py").write_text(
            candidate_defs(candidate_id, env_name, n_agents), encoding="utf-8"
        )
        (env_dir / f"knowledge_actions_{env_name}_candidate{candidate_id}.py").write_text(
            candidate_actions(candidate_id, env_name, n_agents), encoding="utf-8"
        )
        (env_dir / f"difference_credit_{env_name}_candidate{candidate_id}.py").write_text(
            candidate_credit(candidate_id), encoding="utf-8"
        )
        print(f"[FALLBACK] wrote candidate{candidate_id} to {env_dir}: {label}")


def build_validation_cmd(args, candidate_id, model_name):
    safe_model = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    name = f"qmix_exp3_multi_llm_{safe_model}_candidate{candidate_id}_{args.env}_{args.t_max}_seed{args.seed}"
    key = args.env_key or infer_env_key(args.env, args.n_agents)
    time_limit = args.time_limit if args.time_limit is not None else infer_time_limit(args.env)
    cmd = [
        sys.executable,
        "src/main.py",
        "--config=qmix",
        f"--env-config={args.env_config}",
        "with",
        f"env_args.time_limit={time_limit}",
        f"env_args.key={key}",
        f"env_name={args.env}",
        "use_llm=True",
        "use_knowledge_actions=True",
        f"use_difference_credit={str(args.use_difference_credit)}",
        f"knowledge_candidate_id={candidate_id}",
        f"seed={args.seed}",
        f"t_max={args.t_max}",
        f"test_interval={args.test_interval}",
        f"test_nepisode={args.test_nepisode}",
        f"log_interval={args.test_interval}",
        f"runner_log_interval={args.test_interval}",
        f"learner_log_interval={args.test_interval}",
        "save_model=False",
        "use_wandb=False",
        f"use_cuda={str(args.use_cuda)}",
        f"name={name}",
    ]
    if env_family(args.env) == "mpe":
        cmd.insert(cmd.index(f"env_name={args.env}"), f"env_args.N={args.n_agents}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Generate and optionally validate Experiment 3 multi-LLM candidates.")
    parser.add_argument("--env", default="mpe_simple_spread_7")
    parser.add_argument("--n-agents", type=int, default=7)
    parser.add_argument("--env-config", default="gymma")
    parser.add_argument("--env-key", default=None, help="Override Gymnasium/PettingZoo env key. If omitted, infer from --env.")
    parser.add_argument("--time-limit", type=int, default=None, help="Override episode time limit. If omitted, infer from --env family.")
    parser.add_argument("--t-max", type=int, default=200000, help="Small validation length for each candidate.")
    parser.add_argument("--test-interval", type=int, default=50000)
    parser.add_argument("--test-nepisode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="qwen3.7-flash,qwen3.7-plus,qwen3.8-flash")
    parser.add_argument("--use-difference-credit", action="store_true")
    parser.add_argument("--run-validation", action="store_true", help="Run candidate validation after generation. Default only prints commands.")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--use-static-candidates", action="store_true", help="Do not call API; write three local safe fallback candidates.")
    parser.add_argument("--no-fallback-on-api-error", action="store_true", help="Disable static fallback when codeGeneration API call fails.")
    parser.add_argument("--use-cuda", default="True", choices=["True", "False", "true", "false"])
    args = parser.parse_args()

    repo = Path.cwd()
    if not (repo / "src" / "main.py").exists():
        raise SystemExit("Please run this script from the YOLO-MARL repository root.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 3:
        print(f"[WARN] You requested {len(models)} models; Experiment 3 usually uses 3 candidates.")

    if args.use_static_candidates:
        write_static_candidates(args.env, args.n_agents)
        models = ["static_progress", "static_safety", "static_assignment"]
    elif not args.skip_generation:
        hydra_models = "[" + ",".join(models) + "]"
        try:
            run([
                sys.executable,
                "src/prompts/codeGeneration.py",
                f"env={args.env}",
                "exp3_generate_candidates=True",
                f"exp3_candidate_models={hydra_models}",
            ])
        except subprocess.CalledProcessError:
            if args.no_fallback_on_api_error:
                raise
            print("[WARN] LLM API generation failed; write three static safe fallback candidates instead.")
            write_static_candidates(args.env, args.n_agents)
            models = ["static_progress", "static_safety", "static_assignment"]

    print("\n[INFO] Candidate validation commands:")
    for candidate_id, model_name in enumerate(models):
        cmd = build_validation_cmd(args, candidate_id, model_name)
        if args.run_validation:
            run_allow_fail(cmd)
        else:
            print(" ".join(cmd))

    if not args.run_validation:
        print("\n[INFO] Add --run-validation to run all commands automatically.")


if __name__ == "__main__":
    main()
