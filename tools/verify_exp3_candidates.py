import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


VERIFIER_REQUIRED_FIELDS = (
    "overall_pass",
    "action_alignment",
    "credit_correctness",
    "score_reasonableness",
)


def env_family(env_name):
    return str(env_name).split("_", 1)[0].lower()


def primitive_action_ids(env_name):
    return list(range(6)) if env_family(env_name) == "lbf" else list(range(5))


def load_python_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def synthetic_processed_obs(env_name, n_agents, n_food=None):
    family = env_family(env_name)
    if family == "lbf":
        n_food = int(n_food or n_agents)
        food_info = {}
        for i in range(n_food):
            food_info[f"food_{i}"] = (np.array([1.0 + i, 2.0 + i], dtype=np.float32), 1 + (i % 2))
        agents_info = {}
        for i in range(n_agents):
            agents_info[f"agent_{i}"] = (np.array([float(i), float(i + 1)], dtype=np.float32), 1 + (i % 2))
        return food_info, agents_info
    obs = {}
    for i in range(n_agents):
        values = []
        for j in range(n_agents):
            values.append(np.array([0.12 * (j + 1 - i), -0.08 * (j + 1)], dtype=np.float32))
        for j in range(max(0, n_agents - 1)):
            values.append(np.array([0.20 + 0.03 * j, -0.10 - 0.02 * i], dtype=np.float32))
        obs[f"agent_{i}"] = values
    return obs


def synthetic_agent_ids(processed_obs, n_agents):
    if isinstance(processed_obs, tuple) and len(processed_obs) >= 2 and isinstance(processed_obs[1], dict):
        return list(processed_obs[1].keys())
    if isinstance(processed_obs, dict):
        agent_keys = [k for k in processed_obs.keys() if str(k).startswith("agent_")]
        return agent_keys or list(processed_obs.keys())
    return [f"agent_{i}" for i in range(n_agents)]


def syntax_check(path):
    text = read_text(path)
    ast.parse(text, filename=str(path))
    compile(text, str(path), "exec")
    return text


def function_signature_status(fn, expected_names):
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    missing = [name for name in expected_names if name not in params]
    return not missing, params, missing


def code_uses_name(code, name):
    tree = ast.parse(code)
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree))


def static_counterfactual_hints(code):
    lower = code.lower()
    hints = {
        "uses_primitive_actions": code_uses_name(code, "primitive_actions"),
        "mentions_score": "score" in lower,
        "mentions_counterfactual": "counterfactual" in lower or "cf_" in lower or "no_op" in lower or "noop" in lower or "no-op" in lower,
        "mentions_noop_zero": "= 0" in code or ": 0" in code or "no_op" in lower or "noop" in lower,
    }
    hints["passes_static_counterfactual_check"] = all(hints.values())
    return hints


def task_score_hints(env_name, code):
    lower = code.lower()
    family = env_family(env_name)
    if family == "mpe":
        terms = ["landmark", "cover", "collision", "distance", "agent"]
    elif family == "lbf":
        terms = ["food", "pickup", "load", "level", "agent"]
    else:
        terms = ["goal", "task", "agent", "distance", "progress"]
    matched = [term for term in terms if term in lower]
    return {"matched_terms": matched, "score_reasonable_by_keywords": len(matched) >= 2}


def candidate_paths(repo, env_name, candidate_id):
    family = env_family(env_name)
    env_dir = repo / "src" / "prompts" / "env_code" / family
    return {
        "knowledge_action_defs": env_dir / f"knowledge_action_defs_{env_name}_candidate{candidate_id}.py",
        "knowledge_actions": env_dir / f"knowledge_actions_{env_name}_candidate{candidate_id}.py",
        "difference_credit": env_dir / f"difference_credit_{env_name}_candidate{candidate_id}.py",
    }


def parse_json_response(text):
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    broad = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if broad:
        candidates.append(broad.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def verifier_decision(verdict):
    if not isinstance(verdict, dict):
        return "unavailable"
    if "error" in verdict or "raw_response" in verdict:
        return "unavailable"
    if not all(isinstance(verdict.get(field), bool) for field in VERIFIER_REQUIRED_FIELDS):
        return "unavailable"
    return "passed" if all(verdict[field] for field in VERIFIER_REQUIRED_FIELDS) else "failed"


def repair_targets(verdict):
    targets = []
    if verdict.get("action_alignment") is False:
        targets.append("knowledge_actions")
    if verdict.get("credit_correctness") is False or verdict.get("score_reasonableness") is False:
        targets.append("difference_credit")
    return targets


def extract_python_module(response):
    if not isinstance(response, str):
        raise ValueError("LLM repair response is not text")
    match = re.search(r"<code>\s*(.*?)\s*</code>", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1)
    else:
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        code = match.group(1) if match else response
    code = code.strip()
    ast.parse(code, filename="<semantic repair proposal>")
    compile(code, "<semantic repair proposal>", "exec")
    return code + "\n"


def validate_mapping_proposal(code, knowledge_actions, env_name, n_agents, n_food=None):
    import inspect

    namespace = {}
    exec(code, namespace)
    mapping_fn = namespace.get("map_knowledge_action")
    if mapping_fn is None:
        raise ValueError("missing map_knowledge_action")
    ok, params, missing = function_signature_status(
        mapping_fn, ["processed_obs", "agent_id", "action_id", "threshold"]
    )
    if not ok:
        raise ValueError(f"map_knowledge_action missing params: {missing}; got {params}")
    embedded_defs = namespace.get("KNOWLEDGE_ACTIONS")
    if embedded_defs != knowledge_actions:
        raise ValueError("proposal must include the exact existing KNOWLEDGE_ACTIONS dictionary")

    processed_obs = synthetic_processed_obs(env_name, n_agents, n_food=n_food)
    agent_ids = synthetic_agent_ids(processed_obs, n_agents)
    primitive_ids = primitive_action_ids(env_name)
    knowledge_ids = sorted(int(key) for key in knowledge_actions)
    failures = []
    for agent_id in agent_ids:
        for action_id in primitive_ids + knowledge_ids:
            try:
                mapped = int(mapping_fn(processed_obs, agent_id, action_id))
            except Exception as exc:
                failures.append(f"{agent_id}, action {action_id} raised {exc}")
                continue
            if mapped not in primitive_ids:
                failures.append(f"{agent_id}, action {action_id} -> illegal {mapped}")
    if failures:
        raise ValueError("; ".join(failures[:5]))
    for action_id in primitive_ids:
        for agent_id in agent_ids[:1]:
            if int(mapping_fn(processed_obs, agent_id, action_id)) != action_id:
                raise ValueError(f"primitive action {action_id} is not passed through unchanged")
    return {
        "passed": True,
        "signature": list(inspect.signature(mapping_fn).parameters),
        "tested_agents": len(agent_ids),
        "tested_action_ids": primitive_ids + knowledge_ids,
    }


def validate_credit_proposal(code, env_name, n_agents, n_food=None):
    namespace = {}
    exec(code, namespace)
    credit_fn = namespace.get("difference_credit_fn")
    if credit_fn is None:
        raise ValueError("missing difference_credit_fn")
    ok, params, missing = function_signature_status(
        credit_fn, ["prev_processed_obs", "primitive_actions"]
    )
    if not ok:
        raise ValueError(f"difference_credit_fn missing params: {missing}; got {params}")
    hints = static_counterfactual_hints(code)
    if not hints["passes_static_counterfactual_check"]:
        raise ValueError(f"counterfactual static checks failed: {hints}")
    score_hints = task_score_hints(env_name, code)
    if not score_hints["score_reasonable_by_keywords"]:
        raise ValueError(f"task score keyword checks failed: {score_hints}")

    processed_obs = synthetic_processed_obs(env_name, n_agents, n_food=n_food)
    agent_ids = synthetic_agent_ids(processed_obs, n_agents)
    primitive_ids = primitive_action_ids(env_name)
    noop = primitive_ids[0]
    non_noop = primitive_ids[1] if len(primitive_ids) > 1 else noop
    primitive_actions = {
        agent_id: (non_noop if index == 0 else noop)
        for index, agent_id in enumerate(agent_ids)
    }
    credits = credit_fn(processed_obs, primitive_actions)
    if not isinstance(credits, dict):
        raise ValueError("difference_credit_fn must return a dict")
    missing_agents = [agent_id for agent_id in agent_ids if agent_id not in credits]
    if missing_agents:
        raise ValueError(f"difference_credit_fn omitted agents: {missing_agents}")
    for agent_id, value in credits.items():
        try:
            numeric = float(value)
        except Exception as exc:
            raise ValueError(f"credit for {agent_id} is not numeric: {value}") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"credit for {agent_id} is not finite: {value}")
    return {
        "passed": True,
        "tested_agents": agent_ids,
        "credit_preview": {str(key): float(value) for key, value in credits.items()},
        "counterfactual_hints": hints,
        "score_hints": score_hints,
    }


def check_candidate(repo, env_name, n_agents, candidate_id, n_food=None):
    family = env_family(env_name)
    env_dir = repo / "src" / "prompts" / "env_code" / family
    defs_path = env_dir / f"knowledge_action_defs_{env_name}_candidate{candidate_id}.py"
    actions_path = env_dir / f"knowledge_actions_{env_name}_candidate{candidate_id}.py"
    credit_path = env_dir / f"difference_credit_{env_name}_candidate{candidate_id}.py"
    result = {
        "candidate_id": candidate_id,
        "files": {
            "knowledge_action_defs": str(defs_path),
            "knowledge_actions": str(actions_path),
            "difference_credit": str(credit_path),
        },
        "checks": {},
        "issues": [],
    }

    for label, path in result["files"].items():
        if not Path(path).exists():
            result["checks"][f"{label}_exists"] = False
            result["issues"].append(f"missing file: {path}")
        else:
            result["checks"][f"{label}_exists"] = True

    if result["issues"]:
        return result

    try:
        defs_code = syntax_check(defs_path)
        actions_code = syntax_check(actions_path)
        credit_code = syntax_check(credit_path)
        result["checks"]["syntax_correct"] = True
    except Exception as exc:
        result["checks"]["syntax_correct"] = False
        result["issues"].append(f"syntax/import preparation failed: {exc}")
        return result

    try:
        defs_module = load_python_module(defs_path, f"defs_{env_name}_{candidate_id}")
        actions_module = load_python_module(actions_path, f"actions_{env_name}_{candidate_id}")
        credit_module = load_python_module(credit_path, f"credit_{env_name}_{candidate_id}")
        result["checks"]["import_correct"] = True
    except Exception as exc:
        result["checks"]["import_correct"] = False
        result["issues"].append(f"import failed: {exc}")
        return result

    knowledge_actions = getattr(defs_module, "KNOWLEDGE_ACTIONS", None)
    if not isinstance(knowledge_actions, dict) or not knowledge_actions:
        result["checks"]["knowledge_actions_defined"] = False
        result["issues"].append("KNOWLEDGE_ACTIONS is missing or empty")
        knowledge_ids = []
    else:
        result["checks"]["knowledge_actions_defined"] = True
        knowledge_ids = sorted(int(k) for k in knowledge_actions.keys())
        result["knowledge_action_ids"] = knowledge_ids

    mapping_fn = getattr(actions_module, "map_knowledge_action", None)
    credit_fn = getattr(credit_module, "difference_credit_fn", None)
    if mapping_fn is None:
        result["checks"]["map_interface_correct"] = False
        result["issues"].append("missing map_knowledge_action")
    else:
        ok, params, missing = function_signature_status(
            mapping_fn, ["processed_obs", "agent_id", "action_id", "threshold"]
        )
        result["checks"]["map_interface_correct"] = ok
        result["map_signature_params"] = params
        if missing:
            result["issues"].append(f"map_knowledge_action missing params: {missing}")

    if credit_fn is None:
        result["checks"]["credit_interface_correct"] = False
        result["issues"].append("missing difference_credit_fn")
    else:
        ok, params, missing = function_signature_status(
            credit_fn, ["prev_processed_obs", "primitive_actions"]
        )
        result["checks"]["credit_interface_correct"] = ok
        result["credit_signature_params"] = params
        if missing:
            result["issues"].append(f"difference_credit_fn missing params: {missing}")

    processed_obs = synthetic_processed_obs(env_name, n_agents, n_food=n_food)
    agent_ids = synthetic_agent_ids(processed_obs, n_agents)
    primitive_ids = primitive_action_ids(env_name)

    if mapping_fn is not None and knowledge_ids:
        legal = True
        failures = []
        for agent_id in agent_ids:
            for action_id in primitive_ids + knowledge_ids:
                try:
                    mapped = int(mapping_fn(processed_obs, agent_id, action_id))
                    if mapped not in primitive_ids:
                        legal = False
                        failures.append(f"{agent_id}, action {action_id} -> illegal {mapped}")
                except Exception as exc:
                    legal = False
                    failures.append(f"{agent_id}, action {action_id} raised {exc}")
        result["checks"]["processed_obs_input_match_for_mapping"] = len(failures) == 0
        result["checks"]["primitive_action_output_legal"] = legal
        if failures:
            result["issues"].extend(failures[:5])

    if credit_fn is not None:
        try:
            primitive_actions = {agent_id: (primitive_ids[1] if idx == 0 and len(primitive_ids) > 1 else 0) for idx, agent_id in enumerate(agent_ids)}
            credits = credit_fn(processed_obs, primitive_actions)
            numeric = isinstance(credits, dict) and all(np.isfinite(float(v)) for v in credits.values())
            result["checks"]["processed_obs_input_match_for_credit"] = True
            result["checks"]["credit_output_numeric_dict"] = numeric
            result["credit_preview"] = {str(k): float(v) for k, v in list(credits.items())[:5]} if isinstance(credits, dict) else str(credits)
            if not numeric:
                result["issues"].append("difference_credit_fn output is not a numeric dict")
        except Exception as exc:
            result["checks"]["processed_obs_input_match_for_credit"] = False
            result["checks"]["credit_output_numeric_dict"] = False
            result["issues"].append(f"difference_credit_fn call failed: {exc}")

    cf_hints = static_counterfactual_hints(credit_code)
    result["checks"].update(cf_hints)
    if not cf_hints["passes_static_counterfactual_check"]:
        result["issues"].append("static check did not find clear score/no-op counterfactual use")

    result["checks"].update(task_score_hints(env_name, credit_code))
    if not result["checks"].get("score_reasonable_by_keywords", False):
        result["issues"].append("score function keywords do not clearly match the task objective")

    result["semantic_llm_prompt"] = build_llm_verifier_prompt(env_name, n_agents, knowledge_actions, actions_code, credit_code)
    result["passed_local_checks"] = not result["issues"]
    return result


def build_llm_verifier_prompt(env_name, n_agents, knowledge_actions, actions_code, credit_code):
    family = env_family(env_name)
    if family == "mpe":
        obs_contract = "MPE processed_obs is agent_id -> [N landmark relative vectors, N-1 other-agent relative vectors]."
        legal_actions = "MPE primitive actions must be integers 0-4."
        score_goal = "Simple Spread score should focus on landmark coverage, unique coverage, and collision avoidance."
    elif family == "lbf":
        obs_contract = "LBF processed_obs is (food_info, agents_info). food_info maps food_id to None or (food_pos, food_level); agents_info maps agent_id to (agent_pos, agent_level)."
        legal_actions = "LBF primitive actions must be integers 0-5, including movement/no-op and pickup/load."
        score_goal = "LBF score should focus on feasible food approach, teammate grouping, and synchronized pickup/load."
    else:
        obs_contract = "Use the task-specific processed_obs code and avoid assuming MPE-only structures."
        legal_actions = "Primitive actions must be valid for the current environment."
        score_goal = "Score should focus on task progress and coordination."
    return f"""
You are a verifier for LLM-generated YOLO-MARL candidate functions.
Evaluate candidate code for environment {env_name} with {n_agents} agents.

Check these items:
1. Syntax and import feasibility.
2. Interface correctness: map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08) and difference_credit_fn(prev_processed_obs, primitive_actions).
3. Input format match: {obs_contract}
4. Output action legality: {legal_actions}
5. Semantic consistency: each high-level action description must match the implemented mapping logic.
6. Counterfactual correctness: credit_i must equal Score(joint action) - Score(agent_i replaced by no-op), not deleting the agent or ignoring primitive_actions.
7. Score reasonableness: {score_goal}

Return concise JSON with fields: overall_pass, action_alignment, credit_correctness, score_reasonableness, issues, suggestions.

High-level action definitions:
{knowledge_actions}

Mapping function code:
<map_code>
{actions_code}
</map_code>

Difference credit code:
<credit_code>
{credit_code}
</credit_code>
"""


def find_api_key(repo):
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")


def run_llm_text(repo, prompt, model, max_tokens):
    try:
        from openai import OpenAI
    except Exception as exc:
        return None, f"openai package not available: {exc}"
    api_key = find_api_key(repo)
    if not api_key:
        return None, "No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY."
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content, None
    except Exception as exc:
        return None, str(exc)


def run_llm_verifier(repo, prompt, model):
    text, error = run_llm_text(repo, prompt, model, max_tokens=2048)
    if error:
        return {"error": error}, ""
    parsed = parse_json_response(text)
    if parsed is None:
        return {"raw_response": text}, text
    return parsed, text


def build_semantic_repair_prompt(
    env_name, n_agents, module_name, knowledge_actions, current_code, verdict, attempt
):
    family = env_family(env_name)
    if family == "mpe":
        runtime_contract = (
            "processed_obs is agent_id -> [N landmark relative vectors, "
            "N-1 other-agent relative vectors]; legal primitive actions are integers 0-4."
        )
    elif family == "lbf":
        runtime_contract = (
            "processed_obs is (food_info, agents_info), where food_info maps food_id to "
            "None or (food_pos, food_level), and agents_info maps agent_id to "
            "(agent_pos, agent_level); legal primitive actions are integers 0-5."
        )
    else:
        runtime_contract = (
            "Follow the task-specific processed_obs structure already used by the current module "
            "and return only primitive actions legal for this environment."
        )
    if module_name == "knowledge_actions":
        requirements = f"""
Rewrite only the knowledge-action grounding module.
Keep this KNOWLEDGE_ACTIONS dictionary exactly unchanged:
{knowledge_actions}

Every knowledge action id must match its name and description.
Primitive action ids must pass through unchanged.
Return only a legal primitive action id.
Include import numpy as np, the exact KNOWLEDGE_ACTIONS dictionary, and
map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08) -> int.
"""
    else:
        requirements = """
Rewrite only the counterfactual credit module.
Implement credit_i = Score(joint action) - Score(agent_i replaced by no-op).
Do not delete an agent and do not ignore primitive_actions.
The Score must reflect the task objective and penalize harmful events with the correct sign.
Include import numpy as np and
difference_credit_fn(prev_processed_obs, primitive_actions) -> dict.
"""
    return f"""
You originally generated this YOLO-MARL candidate. An independent verifier found semantic problems.
Repair attempt {attempt} for environment {env_name} with {n_agents} agents.

Verifier feedback:
<verifier_feedback>
{json.dumps(verdict, indent=2, ensure_ascii=False)}
</verifier_feedback>

Runtime contract:
{runtime_contract}

{requirements}

Current module:
<current_code>
{current_code}
</current_code>

Return only the complete replacement Python module in <code>...</code>.
Do not return explanations, patches, or a partial function.
"""


def save_audit_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value), encoding="utf-8")


def atomic_replace_module(target_path, code, audit_dir, module_name):
    backup_path = audit_dir / f"{module_name}_before.py"
    save_audit_text(backup_path, read_text(target_path))
    temp_path = target_path.with_name(f".{target_path.name}.semantic_repair.tmp")
    temp_path.write_text(code, encoding="utf-8")
    os.replace(temp_path, target_path)
    return backup_path


def restore_module(target_path, backup_path):
    temp_path = target_path.with_name(f".{target_path.name}.semantic_repair.rollback.tmp")
    shutil.copy2(backup_path, temp_path)
    os.replace(temp_path, target_path)


def attempt_semantic_repairs(
    repo,
    env_name,
    n_agents,
    n_food,
    candidate_id,
    candidate_model,
    item,
    verdict,
    output_dir,
    verifier_model,
    max_repairs,
):
    history = []
    paths = candidate_paths(repo, env_name, candidate_id)
    for attempt in range(1, max_repairs + 1):
        if verifier_decision(verdict) != "failed":
            break
        targets = repair_targets(verdict)
        round_dir = output_dir / f"candidate{candidate_id}" / f"semantic_repair_round{attempt}"
        round_dir.mkdir(parents=True, exist_ok=True)
        save_audit_text(round_dir / "verifier_before.json", json.dumps(verdict, indent=2, ensure_ascii=False))
        round_record = {
            "attempt": attempt,
            "candidate_model": candidate_model,
            "targets": targets,
            "modules": {},
            "rollback": False,
        }
        if not targets:
            round_record["stopped_reason"] = "verifier failed but did not identify a repairable module"
            history.append(round_record)
            save_audit_text(round_dir / "round_record.json", json.dumps(round_record, indent=2, ensure_ascii=False))
            break

        defs_module = load_python_module(
            paths["knowledge_action_defs"],
            f"semantic_defs_{env_name}_{candidate_id}_{attempt}",
        )
        knowledge_actions = getattr(defs_module, "KNOWLEDGE_ACTIONS", {})
        replaced = []
        backups = {}
        for module_name in targets:
            target_path = paths[module_name]
            current_code = read_text(target_path)
            prompt = build_semantic_repair_prompt(
                env_name,
                n_agents,
                module_name,
                knowledge_actions,
                current_code,
                verdict,
                attempt,
            )
            module_dir = round_dir / module_name
            save_audit_text(module_dir / "prompt.txt", prompt)
            response, error = run_llm_text(repo, prompt, candidate_model, max_tokens=8192)
            module_record = {"replaced": False}
            if error:
                module_record.update({
                    "api_error": error,
                    "rollback": "not needed; no replacement attempted",
                })
                save_audit_text(module_dir / "api_error.txt", error)
                round_record["modules"][module_name] = module_record
                continue

            save_audit_text(module_dir / "response.txt", response)
            try:
                code = extract_python_module(response)
                save_audit_text(module_dir / "proposed_code.py", code)
                if module_name == "knowledge_actions":
                    local_validation = validate_mapping_proposal(
                        code, knowledge_actions, env_name, n_agents, n_food=n_food
                    )
                else:
                    local_validation = validate_credit_proposal(
                        code, env_name, n_agents, n_food=n_food
                    )
                save_audit_text(
                    module_dir / "local_validation.json",
                    json.dumps(local_validation, indent=2, ensure_ascii=False),
                )
                backup_path = atomic_replace_module(target_path, code, module_dir, module_name)
                backups[module_name] = backup_path
                replaced.append(module_name)
                module_record.update({"replaced": True, "local_validation": local_validation})
            except Exception as exc:
                module_record.update({
                    "validation_error": str(exc),
                    "rollback": "not needed; proposal failed before atomic replacement",
                })
                save_audit_text(module_dir / "local_validation_error.txt", str(exc))
            round_record["modules"][module_name] = module_record

        if not replaced:
            round_record["stopped_reason"] = "no repair proposal passed local validation"
            history.append(round_record)
            save_audit_text(round_dir / "round_record.json", json.dumps(round_record, indent=2, ensure_ascii=False))
            continue

        repaired_item = check_candidate(repo, env_name, n_agents, candidate_id, n_food=n_food)
        if not repaired_item.get("passed_local_checks", False):
            for module_name in reversed(replaced):
                restore_module(paths[module_name], backups[module_name])
            round_record["rollback"] = True
            round_record["rollback_reason"] = repaired_item.get("issues", [])
            save_audit_text(
                round_dir / "rollback.json",
                json.dumps({
                    "performed": True,
                    "reason": repaired_item.get("issues", []),
                    "restored_modules": replaced,
                }, indent=2, ensure_ascii=False),
            )
            item = check_candidate(repo, env_name, n_agents, candidate_id, n_food=n_food)
            history.append(round_record)
            save_audit_text(round_dir / "round_record.json", json.dumps(round_record, indent=2, ensure_ascii=False))
            continue

        verify_prompt = repaired_item.get("semantic_llm_prompt", "")
        save_audit_text(round_dir / "verifier_after_prompt.txt", verify_prompt)
        verdict, raw_response = run_llm_verifier(repo, verify_prompt, verifier_model)
        save_audit_text(
            round_dir / "verifier_after_response.txt",
            raw_response or json.dumps(verdict, indent=2, ensure_ascii=False),
        )
        save_audit_text(round_dir / "verifier_after.json", json.dumps(verdict, indent=2, ensure_ascii=False))
        round_record["verifier_after"] = verdict
        round_record["verifier_status_after"] = verifier_decision(verdict)
        history.append(round_record)
        save_audit_text(round_dir / "round_record.json", json.dumps(round_record, indent=2, ensure_ascii=False))
        item = repaired_item
        if verifier_decision(verdict) == "unavailable":
            break

    item["llm_verifier"] = verdict
    item["semantic_verifier_status"] = verifier_decision(verdict)
    item["semantic_repair_history"] = history
    return item


def write_reports(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "candidate_verification_report.json"
    md_path = out_dir / "candidate_verification_report.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Experiment 3 Candidate Verification", ""]
    for item in results:
        lines.append(f"## Candidate {item['candidate_id']}")
        lines.append(f"- Local checks passed: {item.get('passed_local_checks', False)}")
        if "semantic_verifier_status" in item:
            lines.append(f"- Semantic verifier status: {item['semantic_verifier_status']}")
        lines.append(f"- Knowledge action ids: {item.get('knowledge_action_ids', [])}")
        if item.get("issues"):
            lines.append("- Issues:")
            for issue in item["issues"]:
                lines.append(f"  - {issue}")
        else:
            lines.append("- Issues: none")
        if "llm_verifier" in item:
            lines.append("- LLM verifier:")
            lines.append("```json")
            lines.append(json.dumps(item["llm_verifier"], indent=2, ensure_ascii=False))
            lines.append("```")
        if item.get("semantic_repair_history"):
            lines.append(f"- Semantic repair rounds: {len(item['semantic_repair_history'])}")
            for repair in item["semantic_repair_history"]:
                lines.append(
                    f"  - Round {repair['attempt']}: targets={repair.get('targets', [])}, "
                    f"status={repair.get('verifier_status_after', repair.get('stopped_reason', 'not reverified'))}, "
                    f"rollback={repair.get('rollback', False)}"
                )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify Experiment 3 candidate functions for YOLO-MARL.")
    parser.add_argument("--env", required=True, help="Prompt env name, e.g. mpe_simple_spread_v3_3 or lbf_3p_3f_coop.")
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--n-food", type=int, default=None, help="Only needed for LBF if it cannot be inferred from env name.")
    parser.add_argument("--candidates", default="0,1,2")
    parser.add_argument("--run-llm", action="store_true", help="Also call a verifier LLM for semantic consistency and score reasonableness.")
    parser.add_argument("--verifier-model", default="qwen3.8-max")
    parser.add_argument(
        "--candidate-models",
        default="qwen3.7-flash,qwen3.7-plus,qwen3.8-flash",
        help="Comma-separated original generator models in candidate id order.",
    )
    parser.add_argument(
        "--semantic-repair-attempts",
        type=int,
        default=2,
        help="Maximum verifier-feedback repair rounds for each failed candidate.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Return success even when semantic verification fails or is unavailable.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo = Path.cwd()
    if not (repo / "src" / "prompts" / "env_code").exists():
        raise SystemExit("Please run this script from the YOLO-MARL repository root.")

    n_food = args.n_food
    if n_food is None:
        match = re.search(r"_(\d+)f(?:_|$)", args.env)
        n_food = int(match.group(1)) if match else args.n_agents

    output_dir = Path(args.output_dir) if args.output_dir else repo / "candidate_verification_reports" / args.env
    candidate_models = [value.strip() for value in args.candidate_models.split(",") if value.strip()]
    candidate_ids = [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
    if args.run_llm and any(candidate_id >= len(candidate_models) for candidate_id in candidate_ids):
        raise SystemExit("--candidate-models must provide the original model for every candidate id")

    results = []
    for candidate_id in candidate_ids:
        print(f"[CHECK] env={args.env} candidate={candidate_id}")
        item = check_candidate(repo, args.env, args.n_agents, candidate_id, n_food=n_food)
        if args.run_llm and item.get("passed_local_checks", False):
            verifier_prompt = item.get("semantic_llm_prompt", "")
            initial_dir = output_dir / f"candidate{candidate_id}" / "initial_verifier"
            save_audit_text(initial_dir / "prompt.txt", verifier_prompt)
            verdict, raw_response = run_llm_verifier(repo, verifier_prompt, args.verifier_model)
            save_audit_text(
                initial_dir / "response.txt",
                raw_response or json.dumps(verdict, indent=2, ensure_ascii=False),
            )
            save_audit_text(initial_dir / "verdict.json", json.dumps(verdict, indent=2, ensure_ascii=False))
            item["llm_verifier"] = verdict
            item["semantic_verifier_status"] = verifier_decision(verdict)
            if (
                item.get("passed_local_checks", False)
                and verifier_decision(verdict) == "failed"
                and args.semantic_repair_attempts > 0
            ):
                print(
                    f"[REPAIR] candidate{candidate_id} verifier failed; "
                    f"returning feedback to {candidate_models[candidate_id]}"
                )
                item = attempt_semantic_repairs(
                    repo=repo,
                    env_name=args.env,
                    n_agents=args.n_agents,
                    n_food=n_food,
                    candidate_id=candidate_id,
                    candidate_model=candidate_models[candidate_id],
                    item=item,
                    verdict=verdict,
                    output_dir=output_dir,
                    verifier_model=args.verifier_model,
                    max_repairs=max(0, args.semantic_repair_attempts),
                )
        elif args.run_llm:
            item["semantic_verifier_status"] = "not_run_local_failure"
        item.pop("semantic_llm_prompt", None)
        results.append(item)

    write_reports(results, output_dir)

    failed = [
        item
        for item in results
        if not item.get("passed_local_checks", False)
        or (args.run_llm and item.get("semantic_verifier_status") != "passed")
    ]
    if failed:
        if args.allow_unverified:
            print(
                "[WARN] continuing despite failed/unavailable candidate verification "
                "because --allow-unverified was supplied"
            )
            return
        raise SystemExit(1)


if __name__ == "__main__":
    main()

