from base import BaseCodeGen
import hydra
import os
import numpy as np
import torch
from util import clean_obs_code
from openai import OpenAI
import dashscope


class CodeGen(BaseCodeGen):
    def __init__(self, cfg):
        super(CodeGen, self).__init__(cfg)
        self.cfg = cfg
        if self.cfg.model.find('gpt') != -1 or self.cfg.model.find('o1') != -1:
            self.model_type = "gpt"
        elif self.cfg.model.find('claude') != -1:
            self.model_type = "claude"
        else:
            self.model_type = "qwen"
        self.use_llm_strategy = True
        # Experiment 3 uses candidate ids to save multiple LLM-generated modules side by side.
        # -1 keeps the original single-module behavior.
        self.active_candidate_id = int(getattr(self.cfg, "knowledge_candidate_id", -1))
        self.active_candidate_model = None

        env_family = self.cfg.env.name.split("_")[0].lower()
        self.env_family = env_family
        self.environment_task = self._build_environment_task_summary()
        self.sys_prompt = "You are an AI expert specializing in multi-agent reinforcement learning."

        self.prompt_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.prompt_dir, "env_code", env_family, f"processed_obs_{self.cfg.env.name}.py"), "r") as f:
            self.processed_global_states_format = clean_obs_code(f.read())
        if env_family == "rware":
            with open(os.path.join(self.prompt_dir, "env_code", env_family, "rware_memory.py"), "r") as f:
                self.rware_memory_format = clean_obs_code(f.read())
        with open(os.path.join(self.prompt_dir, "env_code", env_family, "task2action.py"), "r") as f:
            self.llm_action_format = clean_obs_code(f.read())

        tips_family = env_family
        if not os.path.exists(os.path.join(self.prompt_dir, "tips", tips_family)) and env_family == "rware":
            tips_family = "RWARE"
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "goal.txt"), "r") as f:
            self.goal = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "format.txt"), "r") as f:
            self.format = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "rules.txt"), "r") as f:
            self.rule = f.read().format(time_steps=self.cfg.env.limit)
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "planning_signature.txt"), "r") as f:
            planning_func_signature = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "reward_signature.txt"), "r") as f:
            reward_func_signature = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "instruction_think.txt"), "r") as f:
            self.instruction_think = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "scenario", f"assignment_{self.cfg.env.name}.txt"), "r") as f:
            self.assignment_class = f.read()
        with open(os.path.join(self.prompt_dir, "tips", tips_family, "scenario_think.txt"), "r") as f:
            self.scenerio_think = f.read()
        planning_contract_path = os.path.join(
            self.prompt_dir, "tips", tips_family, "planning_contract.txt"
        )
        self.planning_contract = ""
        if os.path.exists(planning_contract_path):
            with open(planning_contract_path, "r") as f:
                self.planning_contract = f.read()

        self.format = self.format.format(
            planning_func_signature=planning_func_signature,
            reward_func_signature=reward_func_signature,
        )

        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not dashscope_api_key:
            raise RuntimeError("Set DASHSCOPE_API_KEY before generating candidates")

        dashscope.api_key = dashscope_api_key
        self.dashscope_api_key = dashscope_api_key
        self.compat_client = OpenAI(
            api_key=dashscope_api_key,
            base_url=os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        self.client = "dashscope_compatible"

    def _completion_model(self):
        if self.active_candidate_model:
            return str(self.active_candidate_model)
        cfg_model = str(getattr(self.cfg, "model", "qwen3.7-flash"))
        if cfg_model.startswith("qwen"):
            return cfg_model
        # This local code path is wired to DashScope. Keep old behavior for non-Qwen config names.
        return "qwen3.7-flash"

    def _candidate_suffix(self):
        if int(getattr(self, "active_candidate_id", -1)) >= 0:
            return f"_candidate{int(self.active_candidate_id)}"
        return ""

    def _safe_model_name(self):
        return self._completion_model().replace("/", "_").replace("-", "_").replace(".", "_")

    def _raw_name(self, stem):
        return f"{self._safe_model_name()}_{stem}{self._candidate_suffix()}"

    def _save_prompt_response_trace(self, stage, messages, response):
        if not getattr(self.cfg, "save_raw", False):
            return
        out_dir = os.path.join(self.prompt_dir, "gen_code", self.cfg.env.name, "prompt_response")
        os.makedirs(out_dir, exist_ok=True)
        file_num = len(os.listdir(out_dir))
        filename = f"{self._raw_name(stage)}_prompt_response_{file_num}.txt"
        out_path = os.path.join(out_dir, filename)
        prompt_text = []
        for idx, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            prompt_text.append(f"--- Message {idx} | role={role} ---\n{content}")
        joined_prompt = "\n\n".join(prompt_text)
        trace = (
            f"Stage: {stage}\n"
            f"Model: {self._completion_model()}\n"
            f"Environment: {self.cfg.env.name}\n"
            f"Candidate ID: {self.active_candidate_id}\n\n"
            "Prompt sent to LLM after parameter expansion:\n"
            f"{joined_prompt}\n\n"
            "LLM response:\n"
            f"{response}\n"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(trace)
        print(f"Prompt-response trace saved to {out_path}")

    def _build_environment_task_summary(self):
        env_name = str(self.cfg.env.name)
        env_family = env_name.split("_")[0].lower()
        n_agents = getattr(self.cfg.env, "n_agents", "unknown")
        if env_family == "lbf":
            n_food = getattr(self.cfg.env, "n_food", "unknown")
            return (
                f"This is a cooperative Level-Based Foraging task named {env_name}. "
                f"It has {n_agents} agents and {n_food} food items. Agents must coordinate movement, "
                "food approach, waiting, and pickup/load timing to collect food efficiently."
            )
        if env_family == "mpe":
            n_landmarks = getattr(self.cfg.env, "n_landmarks", "unknown")
            return (
                f"This is a cooperative Multi-Agent Particle Environment task named {env_name}. "
                f"It has {n_agents} agents and {n_landmarks} landmarks. Agents must spread out, "
                "cover targets, avoid collisions, and resolve duplicated coverage."
            )
        if env_family == "rware":
            return (
                f"This is a cooperative robotic warehouse task named {env_name}. It has {n_agents} agents. "
                "Agents must navigate warehouse traffic, pick requested shelves, deliver them to goals, "
                "and return shelves while avoiding blocking each other."
            )
        return (
            f"This is a cooperative multi-agent reinforcement learning task named {env_name}. "
            f"It has {n_agents} agents. Infer the task objective, entities, valid actions, and useful "
            "coordination skills from the task assignment, processed observation code, task2action helper, "
            "and strategy tips provided in the prompt."
        )

    def _primitive_action_convention(self):
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "mpe":
            return "Primitive actions are 0 stay/no-op, 1 left, 2 right, 3 down, 4 up. Return one int in 0..4."
        if env_family == "lbf":
            return (
                "Primitive actions follow env_code/lbf/task2action.py, commonly including no-op, movement, "
                "and pickup/load. Return exactly one valid primitive action id used by this environment."
            )
        if env_family == "rware":
            return (
                "Primitive actions follow env_code/rware/task2action.py, commonly including no-op, forward, "
                "turn/side movement, and load/unload/toggle. Return exactly one valid primitive action id used by this environment."
            )
        return (
            "Primitive actions are defined by the provided task2action.py code. Do not assume MPE directions; "
            "read the helper interface and return exactly one valid primitive action id for this environment."
        )

    def _primitive_action_ids(self):
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "lbf":
            return list(range(6))
        return list(range(5))

    def _processed_obs_contract(self):
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "mpe":
            return (
                "For MPE-style processed_obs, it is usually a dict: agent_id -> sequence containing landmark "
                "relative vectors followed by other-agent relative vectors. Infer N from len(processed_obs) and parse vectors robustly."
            )
        if env_family == "lbf":
            return (
                "For LBF-style processed_obs, infer food positions, food levels, agent positions/levels, and pickup feasibility "
                "from the provided processed_obs code. Do not assume landmark vectors or MPE observation layout."
            )
        if env_family == "rware":
            return (
                "For RWARE-style processed_obs, infer shelf, goal, return, carrying, orientation, and traffic/blocking information "
                "from the provided processed_obs and memory code. Do not assume landmark vectors or MPE observation layout."
            )
        return (
            "processed_obs format is task-specific. Infer its structure from the provided processed_obs code, handle dict/list/array "
            "inputs defensively, and avoid hard-coded entity counts or MPE-specific landmark assumptions."
        )

    def _knowledge_action_guidance(self):
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "mpe":
            return "Useful high-level actions may cover targets, seek uncovered targets, hold coverage, avoid collisions, resolve duplicate claims, or rebalance sparse areas."
        if env_family == "lbf":
            return "Useful high-level actions may approach reachable food, wait for teammates near high-level food, coordinate pickup/load, avoid crowding, or reassign agents to uncovered food."
        if env_family == "rware":
            return "Useful high-level actions may move to requested shelves, pick/drop shelves, route to goals/returns, avoid aisle blocking, yield in traffic, or clear congestion."
        return "Design task-relevant high-level coordination skills from the objective and observation/action helper code; do not copy MPE landmark-specific examples unless the task actually contains landmarks."

    def _score_design_guidance(self):
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "mpe":
            return "The score can reward lower target/coverage distance, unique coverage, and safe separation, while penalizing collisions and duplicated target claims."
        if env_family == "lbf":
            return "The score can reward progress toward food, feasible teammate grouping near high-level food, successful pickup/load opportunities, and reduced wasted crowding."
        if env_family == "rware":
            return "The score can reward shelf pickup/delivery/return progress, shorter routes to useful warehouse goals, carrying useful shelves, and lower traffic blocking or deadlock risk."
        return "The score should measure task progress using entities and events visible in processed_obs; choose metrics that match the task objective instead of assuming coverage distance."

    def get_completion(self, messages):
        model = self._completion_model()
        chat_messages = []
        for message in messages:
            role = message.get("role", "user")
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat_messages.append({"role": role, "content": str(message.get("content", ""))})
        try:
            resp = self.compat_client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.2,
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            # Older Qwen text models can still work through the DashScope Generation API.
            if model in ("qwen-turbo", "qwen-plus", "qwen-max", "qwen3.7-flash"):
                prompt = "\n\n".join(message["content"] for message in chat_messages)
                resp = dashscope.Generation.call(
                    model=model,
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=4096,
                )
                if resp.status_code == 200:
                    return resp.output["text"]
                raise Exception(f"DashScope API call failed with model={model}: {resp}") from exc
            raise Exception(f"DashScope compatible API call failed with model={model}: {exc}") from exc

    def generate_strategy(self):
        strat_user_prompt = f"<environment description>{self.environment_task}</environment description>"
        strat_user_prompt += f"<game_rules>{self.rule}</game_rules>"
        strat_user_prompt += f"{self.assignment_class}"
        strat_user_prompt += f"<scenario_considerations>{self.scenerio_think}</scenario_considerations>"
        strat_user_prompt += f"<planning_contract>{self.planning_contract}</planning_contract>"
        strat_user_prompt += "Hard game rules take precedence over scenario suggestions or heuristic strategies."
        strategy = ""
        if self.use_llm_strategy:
            print("Using LLM strategy")
            messages = []
            if self.model_type == "gpt":
                messages += [{"role": "system", "content": self.sys_prompt}]
            messages += [{"role": "user", "content": strat_user_prompt + self.instruction_think}]
            strategy_response = self.get_completion(messages)
            if self.cfg.save_raw:
                self.save_code_to_file(strategy_response, self._raw_name("strategy") + ".txt", "strat")
            strategy += "<tips>" + strategy_response + "</tips>"
        return strategy, strat_user_prompt

    def generate_functions(self):
        strategy_response, strat_user_prompt = self.generate_strategy()

        # Keep the final code-generation request self-contained. Some providers and
        # compatibility fallbacks flatten or otherwise lose earlier chat turns.
        functions_prompt = (
            "<function_generation_context>\n"
            f"<environment_description>{self.environment_task}</environment_description>\n"
            f"<game_rules>{self.rule}</game_rules>\n"
            f"{self.assignment_class}\n"
            f"<scenario_considerations>{self.scenerio_think}</scenario_considerations>\n"
            f"<planning_contract>{self.planning_contract}</planning_contract>\n"
            f"<generated_strategy>{strategy_response}</generated_strategy>\n"
            "Hard game rules take precedence over scenario suggestions and generated strategy advice.\n"
            "</function_generation_context>\n"
        )
        functions_prompt += f"<objective>{self.goal}</objective>"
        functions_prompt += f"The enviroment code information is provided as followed: <processed_states code>{self.processed_global_states_format}<\\processed_tates code>"
        if self.cfg.env.name.split("_")[0] == "rware":
            functions_prompt += f"The memory function is provided as followed: <memory code>{self.rware_memory_format}<\\memory code>"
        functions_prompt += f"The mapping from llm_tasks to llm_actions function as follow: <action code>{self.llm_action_format}</action code>"
        functions_prompt += f"The function generation format are given as follows:{self.format}"
        functions_prompt += "Think step-by-step before you generate two functions based on all the information given above. First, think what kind of informations are provided in processed_states and how to use them in the functions. Second, please analysis the environment descrition and think about what is the proper strategies to use and what combination of tasks for each agent you want to assign in this situation. Please not only pay attention to how to make two functions correct but also try your best to make agents coordinate in two functions based on the instrcution."
        functions_prompt += (
            "Return exactly one <code>...</code> block and no prose outside it. "
            "Inside that same block, define both required top-level functions: "
            "planning_function(processed_state) and compute_reward(processed_state, llm_tasks). "
            "Do not split the functions across separate code blocks."
        )

        if self.model_type == "claude":
            total_messages = [
                {"role": "user", "content": strat_user_prompt},
                {"role": "assistant", "content": strategy_response},
                {"role": "user", "content": functions_prompt},
            ]
        else:
            total_messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "assistant", "content": strategy_response},
                {"role": "user", "content": strat_user_prompt + functions_prompt},
            ]

        function_response = self.get_completion(total_messages)
        save_file_name = self._raw_name("generated_code")
        if self.cfg.save_raw:
            self.save_code_to_file(function_response, save_file_name + ".txt", "raw")
        clean_code = self.extract_python_functions(function_response)
        self.save_code_to_file(clean_code, save_file_name + ".py", "code")
        self.generate_auxiliary_modules(strategy_response)
        return total_messages

    def generate_auxiliary_modules(self, strategy_response=""):
        knowledge_action_defs = self._generate_knowledge_action_definitions(strategy_response)
        self._generate_knowledge_action_module(strategy_response, knowledge_action_defs)
        self._generate_difference_credit_module(strategy_response)

    def _save_env_module(self, filename, code):
        env_family = self.cfg.env.name.split("_")[0]
        out_dir = os.path.join(self.prompt_dir, "env_code", env_family)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(code.rstrip() + "\n")
        print(f"Auxiliary module saved to {out_path}")

    def _clean_or_fallback(self, response, fallback_code, required_name):
        try:
            code = self.extract_python_functions(response)
            compile(code, f"<generated {required_name}>", "exec")
            if f"def {required_name}" not in code:
                raise ValueError(f"missing function {required_name}")
            return code
        except Exception as exc:
            print(f"[WARN] generated {required_name} invalid, using safe fallback: {exc}")
            return fallback_code

    def _clean_code_or_fallback(self, response, fallback_code, required_text):
        try:
            code = self.extract_python_functions(response)
            compile(code, f"<generated {required_text}>", "exec")
            if required_text not in code:
                raise ValueError(f"missing required text {required_text}")
            return code
        except Exception as exc:
            print(f"[WARN] generated {required_text} invalid, using safe fallback: {exc}")
            return fallback_code

    def _generate_knowledge_action_definitions(self, strategy_response):
        knowledge_action_count = max(4, int(self.cfg.env.n_agents))
        first_id = max(self._primitive_action_ids()) + 1
        last_id = first_id + knowledge_action_count - 1
        prompt = f"""
You are extending YOLO-MARL with task-adaptive high-level knowledge actions.
Do not assume this is MPE Simple Spread unless the task context says so.

Task summary:
{self.environment_task}

Primitive action convention:
{self._primitive_action_convention()}

Processed observation contract:
{self._processed_obs_contract()}

Processed observation code available to the runtime:
<processed_obs_code>
{self.processed_global_states_format}
</processed_obs_code>

Task-to-action helper code:
<task2action_code>
{self.llm_action_format}
</task2action_code>

High-level action design guidance:
{self._knowledge_action_guidance()}

Please generate exactly {knowledge_action_count} state-dependent high-level actions with action ids from {first_id} to {last_id}.
These actions must be higher-level coordination skills, not primitive directions copied from the base action space.
They must depend on current processed_obs, scale to the configured number of agents/entities, and be reusable for this task family.

Use the task assignment as context:
<task_assignment>
{self.assignment_class}
</task_assignment>

Return only Python code in <code>...</code> defining exactly this object:

    KNOWLEDGE_ACTIONS = {{
        {first_id}: {{"name": "...", "description": "..."}},
        ...,
        {last_id}: {{"name": "...", "description": "..."}},
    }}

Do not define any function in this step. The mapping function will be generated in the next step.
"""
        messages = [
            {"role": "user", "content": prompt + "\nStrategy tips:\n" + str(strategy_response)}
        ]
        response = self.get_completion(messages)
        self._save_prompt_response_trace("knowledge_action_defs", messages, response)
        if self.cfg.save_raw:
            self.save_code_to_file(response, self._raw_name("knowledge_action_defs") + ".txt", "raw")
        code = self._clean_code_or_fallback(
            response,
            self._fallback_knowledge_action_definitions_code(),
            "KNOWLEDGE_ACTIONS",
        )
        self._save_env_module(f"knowledge_action_defs_{self.cfg.env.name}{self._candidate_suffix()}.py", code)
        return code

    def _synthetic_processed_obs(self):
        n_agents = int(self.cfg.env.n_agents)
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if env_family == "lbf":
            n_food = int(getattr(self.cfg.env, "n_food", n_agents))
            food_info = {}
            for i in range(n_food):
                food_info[f"food_{i}"] = (np.array([0.2 + 0.15 * i, -0.1 * i], dtype=np.float32), 1 + (i % 2))
            agents_info = {}
            for i in range(n_agents):
                agents_info[f"agent_{i}"] = (np.array([0.1 * i, -0.05 * i], dtype=np.float32), 1 + (i % 2))
            return food_info, agents_info
        if env_family == "rware":
            directions = ["up", "down", "left", "right"]
            return {
                "agent_infos": [
                    {
                        "location": np.array([i, i + 1], dtype=np.float32),
                        "is_carrying_shelf": bool(i % 2),
                        "direction": directions[i % len(directions)],
                        "can_move_forward": True,
                        "can_place_shelf": bool((i + 1) % 2),
                    }
                    for i in range(n_agents)
                ],
                "empty_shelves_pos": [np.array([1, 2], dtype=np.float32), np.array([2, 3], dtype=np.float32)],
                "workstation location": [[4, 10], [5, 10]],
                "return location": [[1, 1], [2, 1]],
            }
        if env_family != "mpe":
            return None
        obs = {}
        for i in range(n_agents):
            values = []
            for j in range(n_agents):
                values.append(np.array([0.15 * (j + 1 - i), -0.1 * (j + 1)], dtype=np.float32))
            for j in range(n_agents - 1):
                values.append(np.array([0.2 + 0.03 * j, -0.12 - 0.02 * i], dtype=np.float32))
            obs[f"agent_{i}"] = values
        return obs

    def _synthetic_agent_ids(self, processed_obs):
        n_agents = int(self.cfg.env.n_agents)
        env_family = str(getattr(self, "env_family", self.cfg.env.name.split("_")[0])).lower()
        if processed_obs is None:
            return []
        if env_family == "lbf" and isinstance(processed_obs, tuple) and len(processed_obs) >= 2:
            agents_info = processed_obs[1]
            if isinstance(agents_info, dict):
                return list(agents_info.keys())
        if env_family == "rware":
            return list(range(n_agents))
        if isinstance(processed_obs, dict):
            return [k for k in processed_obs.keys() if str(k).startswith("agent_")]
        return [f"agent_{i}" for i in range(n_agents)]

    def _knowledge_action_ids_from_code(self, code):
        namespace = {}
        exec(code, namespace)
        knowledge_actions = namespace.get("KNOWLEDGE_ACTIONS", {})
        if isinstance(knowledge_actions, dict) and knowledge_actions:
            return sorted(int(k) for k in knowledge_actions.keys())
        first_id = max(self._primitive_action_ids()) + 1
        return list(range(first_id, first_id + max(4, int(self.cfg.env.n_agents))))

    def _validate_knowledge_action_code(self, code, expected_action_ids=None, require_explicit_ids=False):
        namespace = {}
        exec(code, namespace)
        if "map_knowledge_action" not in namespace:
            raise ValueError("missing map_knowledge_action")
        mapping_fn = namespace["map_knowledge_action"]
        if expected_action_ids is None:
            expected_action_ids = self._knowledge_action_ids_from_code(code)
        expected_action_ids = sorted(int(x) for x in expected_action_ids)
        if require_explicit_ids:
            missing_ids = []
            for action_id in expected_action_ids:
                patterns = [
                    f"== {action_id}", f"=={action_id}", f"({action_id},", f", {action_id},",
                    f"[{action_id}]", f" {action_id}:", f"{action_id}:",
                ]
                if not any(pattern in code for pattern in patterns):
                    missing_ids.append(action_id)
            if missing_ids:
                raise ValueError(f"mapping function does not explicitly cover action ids: {missing_ids}")
        primitive_action_ids = self._primitive_action_ids()
        processed_obs = self._synthetic_processed_obs()
        agent_ids = self._synthetic_agent_ids(processed_obs)
        if not agent_ids:
            return
        for agent_id in agent_ids:
            for action_id in primitive_action_ids + expected_action_ids:
                mapped = int(mapping_fn(processed_obs, agent_id, action_id))
                if mapped not in primitive_action_ids:
                    raise ValueError(f"mapped action out of primitive range: {mapped}")

    def _clean_required_function_code(self, response, required_name):
        code = self.extract_python_functions(response)
        compile(code, f"<generated {required_name}>", "exec")
        if f"def {required_name}" not in code:
            raise ValueError(f"missing function {required_name}")
        return code

    def _build_knowledge_action_repair_prompt(
        self,
        base_prompt,
        bad_response,
        bad_code,
        validation_error,
        expected_action_ids,
        attempt_index,
    ):
        return f"""
The previous map_knowledge_action module failed automatic validation. Please repair it.

Validation error from Python:
{validation_error}

Expected knowledge action ids:
{expected_action_ids}

Important processed_obs contract:
{self._processed_obs_contract()}
Use the provided processed_obs code from the original prompt to parse the task-specific observation layout.
Do not unpack observations using an MPE-only landmark/agent format unless that is exactly what the processed_obs code defines.
Convert list/tuple/numpy values robustly and handle missing or malformed fields by returning a safe primitive action.

Primitive action convention:
{self._primitive_action_convention()}
The function must always return a single valid primitive action id for this environment.

Original generation prompt:
{base_prompt}

Previous extracted code, if any:
<previous_code>
{bad_code}
</previous_code>

Previous full LLM response:
<previous_response>
{bad_response}
</previous_response>

Repair attempt {attempt_index}: return a complete Python module only, in <code>...</code>.
The module must include import numpy as np, the exact KNOWLEDGE_ACTIONS dictionary, and def map_knowledge_action(...).
"""

    def _generate_knowledge_action_module(self, strategy_response, knowledge_action_defs):
        prompt = f"""
You are extending YOLO-MARL with a task-adaptive knowledge-action grounding module.
Do not assume this is MPE Simple Spread unless the task context says so.
Generate a Python module that defines exactly this public function:

    map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08) -> int

Task summary:
{self.environment_task}

Primitive action convention:
{self._primitive_action_convention()}

Processed observation contract:
{self._processed_obs_contract()}

Processed observation code available to the runtime:
<processed_obs_code>
{self.processed_global_states_format}
</processed_obs_code>

Task-to-action helper code:
<task2action_code>
{self.llm_action_format}
</task2action_code>

The high-level knowledge actions were generated in a previous LLM step:
<knowledge_action_definitions>
{knowledge_action_defs}
</knowledge_action_definitions>

Your task is to implement the mapping logic for THIS candidate's generated high-level actions.
If action_id is a primitive action id, return it directly.
For every generated KNOWLEDGE_ACTIONS id, write a distinct branch or dispatch entry whose behavior matches that action's name and description.
Do not route all high-level actions to no-op, and do not silently ignore later action ids.
The generated module should also include the exact KNOWLEDGE_ACTIONS dictionary above, so the runtime can inspect how many knowledge actions exist.

The code must infer entity counts from processed_obs or the provided helper code and work for the configured task scale, including n_agents={self.cfg.env.n_agents}.
Return only Python code in <code>...</code>. Include import numpy as np.
"""
        messages = [{"role": "user", "content": prompt + "\nStrategy tips:\n" + str(strategy_response)}]
        response = self.get_completion(messages)
        self._save_prompt_response_trace("knowledge_actions", messages, response)
        if self.cfg.save_raw:
            self.save_code_to_file(response, self._raw_name("knowledge_actions") + ".txt", "raw")

        expected_action_ids = self._knowledge_action_ids_from_code(knowledge_action_defs)
        fallback_code = self._fallback_knowledge_actions_code(knowledge_action_defs)
        max_repairs = int(getattr(self.cfg, "exp3_mapping_repair_attempts", 2))
        last_error = None
        last_code = ""

        for attempt in range(max_repairs + 1):
            try:
                code = self._clean_required_function_code(response, "map_knowledge_action")
                last_code = code
                self._validate_knowledge_action_code(code, expected_action_ids, require_explicit_ids=True)
                if attempt > 0:
                    print(f"[INFO] repaired map_knowledge_action passed validation on attempt {attempt}")
                self._save_env_module(f"knowledge_actions_{self.cfg.env.name}{self._candidate_suffix()}.py", code)
                return
            except Exception as exc:
                last_error = exc
                print(f"[WARN] generated map_knowledge_action validation failed on attempt {attempt}: {exc}")
                if attempt >= max_repairs:
                    break
                repair_prompt = self._build_knowledge_action_repair_prompt(
                    prompt,
                    response,
                    last_code,
                    exc,
                    expected_action_ids,
                    attempt + 1,
                )
                repair_messages = [{"role": "user", "content": repair_prompt}]
                response = self.get_completion(repair_messages)
                self._save_prompt_response_trace(f"knowledge_actions_repair{attempt + 1}", repair_messages, response)
                if self.cfg.save_raw:
                    self.save_code_to_file(
                        response,
                        self._raw_name(f"knowledge_actions_repair{attempt + 1}") + ".txt",
                        "raw",
                    )

        print(f"[WARN] map_knowledge_action still invalid after {max_repairs} repair attempts; using candidate fallback: {last_error}")
        code = fallback_code
        self._validate_knowledge_action_code(code, expected_action_ids, require_explicit_ids=False)
        self._save_env_module(f"knowledge_actions_{self.cfg.env.name}{self._candidate_suffix()}.py", code)

    def _primitive_noop_action_id(self):
        return 0

    def _synthetic_primitive_actions(self):
        primitive_ids = self._primitive_action_ids()
        noop = self._primitive_noop_action_id()
        non_noop = next((action for action in primitive_ids if action != noop), noop)
        agent_ids = self._synthetic_agent_ids(self._synthetic_processed_obs())
        return {agent_id: (non_noop if idx == 0 else noop) for idx, agent_id in enumerate(agent_ids)}

    def _validate_difference_credit_code(self, code):
        namespace = {}
        exec(code, namespace)
        if "difference_credit_fn" not in namespace:
            raise ValueError("missing difference_credit_fn")
        if "primitive_actions" not in code:
            raise ValueError("difference_credit_fn must use primitive_actions")
        lower_code = code.lower()
        required_terms = ("score", "counterfactual", "no-op")
        if not any(term in lower_code for term in required_terms):
            raise ValueError("difference_credit_fn should explicitly implement score/no-op counterfactual logic")
        credit_fn = namespace["difference_credit_fn"]
        processed_obs = self._synthetic_processed_obs()
        primitive_actions = self._synthetic_primitive_actions()
        credit_dict = credit_fn(processed_obs, primitive_actions)
        if not isinstance(credit_dict, dict):
            raise ValueError("difference_credit_fn must return a dict")
        expected_agent_ids = self._synthetic_agent_ids(processed_obs)
        if expected_agent_ids and len(credit_dict) == 0:
            raise ValueError("difference_credit_fn returned an empty dict")
        for key, value in credit_dict.items():
            try:
                numeric_value = float(value)
            except Exception as exc:
                raise ValueError(f"credit for {key} is not numeric: {value}") from exc
            if not np.isfinite(numeric_value):
                raise ValueError(f"credit for {key} is not finite: {value}")

    def _build_difference_credit_repair_prompt(
        self,
        base_prompt,
        bad_response,
        bad_code,
        validation_error,
        attempt_index,
    ):
        return f"""
The previous difference_credit_fn module failed automatic validation. Please repair it.

Validation error from Python:
{validation_error}

Important requirements:
1. Define exactly this public function:
   difference_credit_fn(prev_processed_obs, primitive_actions) -> dict
2. Return a dict mapping each agent id to a finite numeric credit value.
3. Use primitive_actions in the score calculation.
4. Implement the no-op counterfactual explicitly:
   credit_i = score(current joint action) - score(agent_i replaced by no-op)
5. Do not delete the agent from the state. Only replace that agent action with no-op.
6. The task-adaptive score should match the current environment:
   - Simple Spread: landmark coverage distance, unique coverage, collision avoidance.
   - LBF: progress toward feasible food, teammate grouping, synchronized pickup/load.
7. Parse processed_obs according to this contract:
   {self._processed_obs_contract()}
8. Handle malformed or missing values defensively.

Primitive action convention:
{self._primitive_action_convention()}

Original generation prompt:
{base_prompt}

Previous extracted code, if any:
<previous_code>
{bad_code}
</previous_code>

Previous full LLM response:
<previous_response>
{bad_response}
</previous_response>

Repair attempt {attempt_index}: return a complete Python module only, in <code>...</code>.
The module must include import numpy as np and def difference_credit_fn(...).
"""

    def _generate_difference_credit_module(self, strategy_response):
        prompt = f"""
You are extending YOLO-MARL with a task-adaptive counterfactual credit module.
Do not assume this is MPE Simple Spread unless the task context says so.
Generate a Python module that defines exactly this public function:

    difference_credit_fn(prev_processed_obs, primitive_actions) -> dict

Task summary:
{self.environment_task}

Processed observation contract:
{self._processed_obs_contract()}

Processed observation code available to the runtime:
<processed_obs_code>
{self.processed_global_states_format}
</processed_obs_code>

Primitive action convention:
{self._primitive_action_convention()}

It should estimate each agent's contribution using a no-op counterfactual:
    credit_i = score(current joint action) - score(agent_i replaced by no-op)

The fixed framework is the counterfactual difference above; the task-adaptive part is the score function.
Score design guidance:
{self._score_design_guidance()}

primitive_actions may be either a dict agent_id->action or a list ordered by agent index.
Return a dict mapping each agent id to a numeric credit value.
Return only Python code in <code>...</code>. Include import numpy as np.
"""
        messages = [{"role": "user", "content": prompt + "\nStrategy tips:\n" + str(strategy_response)}]
        response = self.get_completion(messages)
        self._save_prompt_response_trace("difference_credit", messages, response)
        if self.cfg.save_raw:
            self.save_code_to_file(response, self._raw_name("difference_credit") + ".txt", "raw")

        fallback_code = self._fallback_difference_credit_code()
        max_repairs = int(getattr(self.cfg, "exp3_credit_repair_attempts", getattr(self.cfg, "exp3_mapping_repair_attempts", 2)))
        last_error = None
        last_code = ""

        for attempt in range(max_repairs + 1):
            try:
                code = self._clean_required_function_code(response, "difference_credit_fn")
                last_code = code
                self._validate_difference_credit_code(code)
                if attempt > 0:
                    print(f"[INFO] repaired difference_credit_fn passed validation on attempt {attempt}")
                self._save_env_module(f"difference_credit_{self.cfg.env.name}{self._candidate_suffix()}.py", code)
                return
            except Exception as exc:
                last_error = exc
                print(f"[WARN] generated difference_credit_fn validation failed on attempt {attempt}: {exc}")
                if attempt >= max_repairs:
                    break
                repair_prompt = self._build_difference_credit_repair_prompt(
                    prompt,
                    response,
                    last_code,
                    exc,
                    attempt + 1,
                )
                repair_messages = [{"role": "user", "content": repair_prompt}]
                response = self.get_completion(repair_messages)
                self._save_prompt_response_trace(f"difference_credit_repair{attempt + 1}", repair_messages, response)
                if self.cfg.save_raw:
                    self.save_code_to_file(
                        response,
                        self._raw_name(f"difference_credit_repair{attempt + 1}") + ".txt",
                        "raw",
                    )

        print(f"[WARN] difference_credit_fn still invalid after {max_repairs} repair attempts; using safe fallback: {last_error}")
        code = fallback_code
        self._validate_difference_credit_code(code)
        self._save_env_module(f"difference_credit_{self.cfg.env.name}{self._candidate_suffix()}.py", code)

    def _fallback_knowledge_action_definitions_code(self):
        first_id = max(self._primitive_action_ids()) + 1
        actions = [
            ("task_progress_action", "Move toward the most useful task entity according to processed_obs."),
            ("safe_coordination_action", "Resolve local conflicts or blocking while preserving task progress."),
            ("hold_or_complete_action", "Hold, pickup, drop, or complete the task when the current state is already favorable."),
            ("rebalance_team_action", "Reassign effort toward under-served entities or teammates."),
        ]
        lines = ["KNOWLEDGE_ACTIONS = {"]
        for offset, (name, desc) in enumerate(actions, start=first_id):
            lines.append(f'    {offset}: {{"name": "{name}", "description": "{desc}"}},')
        lines.append("}\n")
        return "\n".join(lines)

    def _fallback_knowledge_actions_code(self, knowledge_action_defs=None):
        if str(getattr(self, "env_family", "")).lower() == "lbf":
            return self._fallback_lbf_knowledge_actions_code(knowledge_action_defs)
        n_agents = max(4, int(self.cfg.env.n_agents))
        candidate_id = int(getattr(self, "active_candidate_id", -1))
        if knowledge_action_defs and "KNOWLEDGE_ACTIONS" in knowledge_action_defs:
            defs_code = knowledge_action_defs.rstrip()
        else:
            defs_code = self._fallback_knowledge_action_definitions_code().rstrip()
        strategy_mode = candidate_id % 3 if candidate_id >= 0 else 0
        return f'''import numpy as np

{defs_code}

BASE_N_ACTIONS = {max(self._primitive_action_ids()) + 1}
N_KNOWLEDGE_ACTIONS = len(KNOWLEDGE_ACTIONS)
TOTAL_N_ACTIONS = BASE_N_ACTIONS + N_KNOWLEDGE_ACTIONS
CANDIDATE_MAPPING_MODE = {strategy_mode}


def map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08):
    action_id = int(action_id)
    if action_id < BASE_N_ACTIONS:
        return action_id
    n_agents = _n_agents(processed_obs)
    agent_index = _agent_index(agent_id)
    nearest_other = _nearest_other_agent(processed_obs, agent_id)

    if CANDIDATE_MAPPING_MODE == 1 and nearest_other is not None and np.linalg.norm(nearest_other) < 0.16:
        return _move_away(nearest_other, threshold)

    if action_id == 5:
        return _move_towards(_assigned_landmark(processed_obs, agent_id), threshold)
    if action_id == 6:
        return _move_towards(_nearest_uncovered_landmark(processed_obs, agent_id), threshold)
    if action_id == 7:
        return 0 if nearest_other is None else _move_away(nearest_other, threshold)
    if action_id == 8:
        if CANDIDATE_MAPPING_MODE == 2:
            return _move_towards(_least_contested_landmark(processed_obs, agent_id), threshold)
        return _yield_or_cover(processed_obs, agent_id, threshold)
    if action_id == 9:
        if _on_any_landmark(processed_obs, agent_id, threshold):
            return 0
        return _move_towards(_assigned_landmark(processed_obs, agent_id), threshold)
    if action_id == 10:
        target = _farthest_uncovered_landmark(processed_obs, agent_id)
        if CANDIDATE_MAPPING_MODE == 1 and nearest_other is not None and np.linalg.norm(nearest_other) < 0.22:
            return _move_away(nearest_other, threshold)
        return _move_towards(target, threshold)
    if action_id == 11:
        if CANDIDATE_MAPPING_MODE == 0:
            return _move_towards(_nearest_uncovered_landmark(processed_obs, agent_id), threshold)
        if CANDIDATE_MAPPING_MODE == 1:
            return 0 if nearest_other is None else _move_away(nearest_other, threshold)
        return _move_towards(_landmarks(processed_obs[agent_id], n_agents)[(agent_index + 1) % n_agents], threshold)

    if action_id in KNOWLEDGE_ACTIONS:
        idx = (action_id - BASE_N_ACTIONS) % max(1, n_agents)
        return _move_towards(_landmarks(processed_obs[agent_id], n_agents)[idx], threshold)
    return 0


def _agent_index(agent_id):
    return int(str(agent_id).split("_")[-1])


def _n_agents(processed_obs):
    return len(processed_obs)


def _as_vec(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    return arr[:2]


def _landmarks(agent_obs, n_agents):
    return [_as_vec(v) for v in list(agent_obs)[:n_agents]]


def _others(agent_obs, n_agents):
    return [_as_vec(v) for v in list(agent_obs)[n_agents:]]


def _assigned_landmark(processed_obs, agent_id):
    n_agents = _n_agents(processed_obs)
    idx = _agent_index(agent_id)
    landmarks = _landmarks(processed_obs[agent_id], n_agents)
    return landmarks[idx % len(landmarks)]


def _nearest_uncovered_landmark(processed_obs, agent_id, cover_threshold=0.12):
    n_agents = _n_agents(processed_obs)
    landmarks = _landmarks(processed_obs[agent_id], n_agents)
    covered = _covered_landmark_ids(processed_obs, agent_id, cover_threshold)
    candidates = [(idx, landmark) for idx, landmark in enumerate(landmarks) if idx not in covered]
    if not candidates:
        candidates = list(enumerate(landmarks))
    return min(candidates, key=lambda item: np.linalg.norm(item[1]))[1]


def _farthest_uncovered_landmark(processed_obs, agent_id, cover_threshold=0.12):
    n_agents = _n_agents(processed_obs)
    landmarks = _landmarks(processed_obs[agent_id], n_agents)
    covered = _covered_landmark_ids(processed_obs, agent_id, cover_threshold)
    candidates = [(idx, landmark) for idx, landmark in enumerate(landmarks) if idx not in covered]
    if not candidates:
        candidates = list(enumerate(landmarks))
    return max(candidates, key=lambda item: np.linalg.norm(item[1]))[1]


def _least_contested_landmark(processed_obs, agent_id):
    n_agents = _n_agents(processed_obs)
    own_landmarks = _landmarks(processed_obs[agent_id], n_agents)
    scores = []
    for idx, landmark in enumerate(own_landmarks):
        nearby_agents = 0
        for other_id, other_obs in processed_obs.items():
            if other_id == agent_id:
                continue
            other_landmark = _landmarks(other_obs, n_agents)[idx]
            if np.linalg.norm(other_landmark) < 0.35:
                nearby_agents += 1
        scores.append((nearby_agents, np.linalg.norm(landmark), landmark))
    return min(scores, key=lambda item: (item[0], item[1]))[2]


def _covered_landmark_ids(processed_obs, agent_id, cover_threshold):
    n_agents = _n_agents(processed_obs)
    covered = set()
    for other_agent, other_obs in processed_obs.items():
        if other_agent == agent_id:
            continue
        for idx, landmark in enumerate(_landmarks(other_obs, n_agents)):
            if np.linalg.norm(landmark) < cover_threshold:
                covered.add(idx)
    return covered


def _nearest_other_agent(processed_obs, agent_id):
    n_agents = _n_agents(processed_obs)
    others = _others(processed_obs[agent_id], n_agents)
    if not others:
        return None
    return min(others, key=lambda pos: np.linalg.norm(pos))


def _on_any_landmark(processed_obs, agent_id, threshold=0.08):
    n_agents = _n_agents(processed_obs)
    return any(np.linalg.norm(pos) < threshold for pos in _landmarks(processed_obs[agent_id], n_agents))


def _yield_or_cover(processed_obs, agent_id, threshold):
    nearest_other = _nearest_other_agent(processed_obs, agent_id)
    if nearest_other is not None and np.linalg.norm(nearest_other) < 0.18:
        return _move_away(nearest_other, threshold)
    if _on_any_landmark(processed_obs, agent_id, threshold):
        return 0
    return _move_towards(_nearest_uncovered_landmark(processed_obs, agent_id), threshold)


def _move_towards(relative_pos, threshold=0.08):
    dx, dy = _as_vec(relative_pos)
    if np.linalg.norm([dx, dy]) < threshold:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def _move_away(relative_pos, threshold=0.08):
    dx, dy = _as_vec(relative_pos)
    if np.linalg.norm([dx, dy]) > max(threshold, 0.16):
        return 0
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 2
    return 3 if dy > 0 else 4
'''


    def _fallback_lbf_knowledge_actions_code(self, knowledge_action_defs=None):
        candidate_id = int(getattr(self, "active_candidate_id", -1))
        if knowledge_action_defs and "KNOWLEDGE_ACTIONS" in knowledge_action_defs:
            defs_code = knowledge_action_defs.rstrip()
        else:
            defs_code = self._fallback_knowledge_action_definitions_code().rstrip()
        template = '''import numpy as np

__KNOWLEDGE_ACTION_DEFS__

BASE_N_ACTIONS = 6
N_KNOWLEDGE_ACTIONS = len(KNOWLEDGE_ACTIONS)
TOTAL_N_ACTIONS = BASE_N_ACTIONS + N_KNOWLEDGE_ACTIONS
CANDIDATE_MAPPING_MODE = __CANDIDATE_MAPPING_MODE__


def map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08):
    action_id = int(action_id)
    if 0 <= action_id < BASE_N_ACTIONS:
        return action_id
    if action_id not in KNOWLEDGE_ACTIONS:
        return 0
    if not (isinstance(processed_obs, tuple) and len(processed_obs) >= 2):
        return 0

    food_info, agents_info = processed_obs[:2]
    if not isinstance(food_info, dict) or not isinstance(agents_info, dict):
        return 0
    agent_key = _resolve_agent_id(agents_info, agent_id)
    if agent_key is None:
        return 0
    available = _available_foods(food_info)
    if not available:
        return 0

    spec = KNOWLEDGE_ACTIONS.get(action_id, {})
    text = (str(spec.get("name", "")) + " " + str(spec.get("description", ""))).lower()
    ready_food = _ready_adjacent_food(agent_key, available, agents_info)
    pickup_skill = any(word in text for word in ("pickup", "load", "collect", "complete"))
    if pickup_skill and ready_food is not None:
        return 5

    if any(word in text for word in ("balance", "assign", "reassign", "sparse", "crowd", "conflict")):
        target = _least_crowded_food(agent_key, available, agents_info)
    elif "nearest" in text or "closest" in text or "approach" in text:
        target = _nearest_food(agent_key, available, agents_info)
    elif CANDIDATE_MAPPING_MODE % 3 == 2:
        target = _assigned_food(agent_key, available)
    elif CANDIDATE_MAPPING_MODE % 3 == 1:
        target = _least_crowded_food(agent_key, available, agents_info)
    else:
        target = _nearest_food(agent_key, available, agents_info)

    if target is None:
        return 0
    distance = _grid_distance(agents_info[agent_key][0], target[1][0])
    if distance <= 1.0:
        if pickup_skill and _team_can_load(target, agents_info):
            return 5
        return 0
    return _move_towards(agents_info[agent_key][0], target[1][0])


def _resolve_agent_id(agents_info, agent_id):
    if agent_id in agents_info:
        return agent_id
    text = str(agent_id)
    if text in agents_info:
        return text
    try:
        candidate = "agent_" + str(int(text.split("_")[-1]))
    except Exception:
        return None
    return candidate if candidate in agents_info else None


def _entity_index(entity_id):
    try:
        return int(str(entity_id).split("_")[-1])
    except Exception:
        return 0


def _available_foods(food_info):
    foods = [(food_id, value) for food_id, value in food_info.items() if value is not None]
    return sorted(foods, key=lambda item: _entity_index(item[0]))


def _as_position(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return np.zeros(2, dtype=np.float32)
    return arr[:2]


def _grid_distance(first, second):
    return float(np.sum(np.abs(_as_position(first) - _as_position(second))))


def _nearest_food(agent_id, available, agents_info):
    agent_pos = agents_info[agent_id][0]
    return min(available, key=lambda item: (_grid_distance(agent_pos, item[1][0]), _entity_index(item[0])))


def _assigned_food(agent_id, available):
    return available[_entity_index(agent_id) % len(available)]


def _least_crowded_food(agent_id, available, agents_info):
    agent_pos = agents_info[agent_id][0]
    def score(item):
        food_pos = item[1][0]
        nearby = sum(_grid_distance(info[0], food_pos) <= 1.0 for info in agents_info.values())
        return nearby, _grid_distance(agent_pos, food_pos), _entity_index(item[0])
    return min(available, key=score)


def _team_can_load(food, agents_info):
    food_pos, food_level = food[1]
    nearby_level = 0.0
    for agent_pos, agent_level in agents_info.values():
        if _grid_distance(agent_pos, food_pos) <= 1.0:
            nearby_level += float(agent_level)
    return nearby_level >= float(food_level)


def _ready_adjacent_food(agent_id, available, agents_info):
    agent_pos = agents_info[agent_id][0]
    ready = [food for food in available if _grid_distance(agent_pos, food[1][0]) <= 1.0 and _team_can_load(food, agents_info)]
    if not ready:
        return None
    return min(ready, key=lambda item: _entity_index(item[0]))


def _move_towards(agent_pos, food_pos):
    delta = _as_position(food_pos) - _as_position(agent_pos)
    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) >= abs(dy) and abs(dx) > 0.0:
        return 2 if dx > 0.0 else 1
    if abs(dy) > 0.0:
        return 4 if dy > 0.0 else 3
    return 0
'''
        return (
            template.replace("__KNOWLEDGE_ACTION_DEFS__", defs_code)
            .replace("__CANDIDATE_MAPPING_MODE__", str(candidate_id % 3 if candidate_id >= 0 else 0))
        )


    def _fallback_lbf_difference_credit_code(self):
        return '''import numpy as np


def difference_credit_fn(prev_processed_obs, primitive_actions):
    """Estimate each LBF agent contribution with an explicit no-op counterfactual."""
    if not (isinstance(prev_processed_obs, tuple) and len(prev_processed_obs) >= 2):
        return {}
    food_info, agents_info = prev_processed_obs[:2]
    if not isinstance(food_info, dict) or not isinstance(agents_info, dict):
        return {}
    agent_ids = sorted(agents_info, key=_entity_index)
    actions = _action_dict(agent_ids, primitive_actions)
    actual_score = _team_progress_score(food_info, agents_info, actions)
    credits = {}
    for agent_id in agent_ids:
        counterfactual = dict(actions)
        counterfactual[agent_id] = 0
        counterfactual_score = _team_progress_score(food_info, agents_info, counterfactual)
        credits[agent_id] = float(actual_score - counterfactual_score)
    return credits


def _action_dict(agent_ids, primitive_actions):
    if isinstance(primitive_actions, dict):
        return {agent_id: int(primitive_actions.get(agent_id, 0)) for agent_id in agent_ids}
    values = list(primitive_actions) if primitive_actions is not None else []
    return {agent_id: int(values[index]) if index < len(values) else 0 for index, agent_id in enumerate(agent_ids)}


def _entity_index(entity_id):
    try:
        return int(str(entity_id).split("_")[-1])
    except Exception:
        return 0


def _as_position(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return np.zeros(2, dtype=np.float32)
    return arr[:2]


def _grid_distance(first, second):
    return float(np.sum(np.abs(_as_position(first) - _as_position(second))))


def _next_position(position, action):
    delta = {
        1: np.array([-1.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 0.0], dtype=np.float32),
        3: np.array([0.0, -1.0], dtype=np.float32),
        4: np.array([0.0, 1.0], dtype=np.float32),
    }.get(int(action), np.zeros(2, dtype=np.float32))
    return _as_position(position) + delta


def _team_progress_score(food_info, agents_info, actions):
    available = [value for value in food_info.values() if value is not None]
    if not available:
        return 0.0
    next_positions = {
        agent_id: _next_position(info[0], actions.get(agent_id, 0))
        for agent_id, info in agents_info.items()
    }
    score = 0.0
    for food_pos, food_level in available:
        distances = {
            agent_id: _grid_distance(position, food_pos)
            for agent_id, position in next_positions.items()
        }
        score -= min(distances.values()) if distances else 0.0
        adjacent_level = sum(
            float(agents_info[agent_id][1])
            for agent_id, distance in distances.items()
            if distance <= 1.0
        )
        score += 0.1 * min(adjacent_level, float(food_level))
        loaders = [
            agent_id for agent_id, distance in distances.items()
            if actions.get(agent_id, 0) == 5 and distance <= 1.0
        ]
        loading_level = sum(float(agents_info[agent_id][1]) for agent_id in loaders)
        if loaders and loading_level >= float(food_level):
            score += 10.0 + float(food_level)
        elif loaders:
            score -= 0.5 * len(loaders)
    return float(score)
'''


    def _fallback_difference_credit_code(self):
        if str(getattr(self, "env_family", "")).lower() == "lbf":
            return self._fallback_lbf_difference_credit_code()
        return '''import numpy as np

STEP_SIZE = 0.1


def difference_credit_fn(prev_processed_obs, primitive_actions):
    agent_ids = sorted(prev_processed_obs.keys(), key=lambda x: int(x.split("_")[-1]))
    if isinstance(primitive_actions, dict):
        actions = [int(primitive_actions.get(agent_id, 0)) for agent_id in agent_ids]
    else:
        actions = [int(a) for a in primitive_actions]
    actual_score = _team_progress_score(prev_processed_obs, actions)
    credits = {}
    for idx, agent_id in enumerate(agent_ids):
        counterfactual = list(actions)
        counterfactual[idx] = 0
        cf_score = _team_progress_score(prev_processed_obs, counterfactual)
        credits[agent_id] = float(actual_score - cf_score)
    return credits


def _team_progress_score(processed_obs, actions):
    agent_ids = sorted(processed_obs.keys(), key=lambda x: int(x.split("_")[-1]))
    n_agents = len(agent_ids)
    before_landmarks = []
    after_landmarks = []
    after_agent_rel = []
    for idx, agent_id in enumerate(agent_ids):
        obs = processed_obs[agent_id]
        landmarks = [np.asarray(v, dtype=np.float32) for v in obs[:n_agents]]
        others = [np.asarray(v, dtype=np.float32) for v in obs[n_agents:]]
        delta = _action_delta(actions[idx] if idx < len(actions) else 0)
        before_landmarks.append(landmarks)
        after_landmarks.append([landmark - delta for landmark in landmarks])
        after_agent_rel.append([other - delta for other in others])
    progress = _coverage_distance(before_landmarks) - _coverage_distance(after_landmarks)
    return float(progress - _collision_penalty(after_agent_rel))


def _coverage_distance(all_landmarks):
    n_agents = len(all_landmarks)
    total = 0.0
    for landmark_idx in range(n_agents):
        total += min(np.linalg.norm(all_landmarks[agent_idx][landmark_idx]) for agent_idx in range(n_agents))
    return total


def _collision_penalty(all_other_rel, threshold=0.12):
    penalty = 0.0
    for others in all_other_rel:
        for rel in others:
            dist = np.linalg.norm(rel)
            if dist < threshold:
                penalty += threshold - dist
    return penalty / 2.0


def _action_delta(action):
    if action == 1:
        return np.array([-STEP_SIZE, 0.0], dtype=np.float32)
    if action == 2:
        return np.array([STEP_SIZE, 0.0], dtype=np.float32)
    if action == 3:
        return np.array([0.0, -STEP_SIZE], dtype=np.float32)
    if action == 4:
        return np.array([0.0, STEP_SIZE], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)
'''


    def generate_exp3_candidates(self):
        """Generate several complete auxiliary-module candidates for Experiment 3.

        Each LLM/model generates a full candidate package:
        knowledge action definitions + mapping function + difference credit function.
        The generated files are saved with suffixes candidate0, candidate1, ... so
        short validation runs can select them with knowledge_candidate_id.
        """
        models = list(getattr(self.cfg, "exp3_candidate_models", []) or [])
        if not models:
            models = ["qwen3.8-max", "kimi-k2.7-code", "deepseek-v4-pro-0813"]

        original_candidate_id = self.active_candidate_id
        original_candidate_model = self.active_candidate_model
        for candidate_id, model in enumerate(models):
            self.active_candidate_id = candidate_id
            self.active_candidate_model = str(model)
            print(f"[EXP3] generating candidate{candidate_id} with model={self.active_candidate_model}")
            strategy_response, _ = self.generate_strategy()
            self.generate_auxiliary_modules(strategy_response)

        self.active_candidate_id = original_candidate_id
        self.active_candidate_model = original_candidate_model


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    print(cfg)
    codegen = CodeGen(cfg)
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if bool(getattr(cfg, "exp3_generate_candidates", False)):
        codegen.generate_exp3_candidates()
    else:
        codegen.generate_functions()


if __name__ == "__main__":
    main()


