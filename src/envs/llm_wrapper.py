from .gymma import GymmaWrapper
from prompts.util import *
from utils.logging import get_logger
from copy import deepcopy
import numpy as np
# from .multiagentenv import MultiAgentEnv
logger = get_logger()

class LLMWrapper(GymmaWrapper):
    def __init__(self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        compute_reward_fn,
        planning_fn,
        reward_mode,
        env_name,
        use_knowledge_actions=False,
        knowledge_action_mode="normal",
        knowledge_candidate_id=-1,
        use_difference_credit=False,
        use_credit_assignment=False,
        use_dynamic_credit=False,
        credit_weight=0.0,
        credit_start_step=0,
        credit_end_step=0,
        normalize_difference_credit=True,
        credit_normalization_eps=1e-6,
        simple_spread_success_threshold=0.15,
        **kwargs,
    ):
        super().__init__(key, time_limit,
                        pretrained_wrapper,
                        seed,
                        common_reward, 
                        reward_scalarisation,
                        **kwargs)
        self.reward_mode = reward_mode
        self.set_func(planning_fn, compute_reward_fn)
        self.env_name = env_name
        self.dirname = self.env_name.split("_")[0]
        self.pre_action = [[] for _ in range(self.n_agents)]
        self.t_env = 0
        self.use_knowledge_actions = use_knowledge_actions
        self.knowledge_action_mode = str(knowledge_action_mode).lower()
        self.knowledge_candidate_id = int(knowledge_candidate_id)
        if self.knowledge_action_mode not in ("normal", "dummy", "random"):
            raise ValueError(f"Invalid knowledge_action_mode: {knowledge_action_mode}")
        self.use_difference_credit = use_difference_credit or use_credit_assignment
        self.use_dynamic_credit = use_dynamic_credit
        self.credit_weight = credit_weight
        self.credit_start_step = credit_start_step
        self.credit_end_step = credit_end_step
        self.normalize_difference_credit = bool(normalize_difference_credit)
        self.credit_normalization_eps = float(credit_normalization_eps)
        self.simple_spread_success_threshold = float(simple_spread_success_threshold)
        self.base_n_actions = super().get_total_actions()
        self.knowledge_action_fn = None
        self.knowledge_actions = {}
        self.n_knowledge_actions = 0
        self.difference_credit_fn = None

        if self.use_knowledge_actions:
            self.knowledge_action_fn = self._import_candidate_or_default(
                "knowledge_actions",
                "map_knowledge_action",
            )
            self.knowledge_actions = self._import_candidate_or_default(
                "knowledge_action_defs",
                "KNOWLEDGE_ACTIONS",
            ) or self._import_candidate_or_default(
                "knowledge_actions",
                "KNOWLEDGE_ACTIONS",
            ) or {}
            if isinstance(self.knowledge_actions, dict) and len(self.knowledge_actions) > 0:
                self.n_knowledge_actions = len(self.knowledge_actions)
            else:
                self.n_knowledge_actions = 4

        if self.use_difference_credit:
            self.difference_credit_fn = self._import_candidate_or_default(
                "difference_credit",
                "difference_credit_fn",
            )

        if self.dirname == "rware":
            self.initial_memory = {
                'workstation location': [[4,10], [5,10]],
                'empty_shelves_pos': [],
                'return location': [],
                'status of carried shelf': [False for _ in range(self.n_agents)],
                'is_carrying_shelf': [False for _ in range(self.n_agents)]
            }
            self.rware_memory = deepcopy(self.initial_memory)  # initial memory
            self.add_memory2obs =  import_function(
                f"prompts.env_code.rware.rware_memory", "rware_memory")

    def _candidate_module_name(self, module_prefix):
        suffix = f"_candidate{self.knowledge_candidate_id}" if self.knowledge_candidate_id >= 0 else ""
        return f"prompts.env_code.{self.dirname}.{module_prefix}_{self.env_name}{suffix}"

    def _import_candidate_or_default(self, module_prefix, object_name):
        module_name = self._candidate_module_name(module_prefix)
        try:
            return import_function(module_name, object_name)
        except Exception as exc:
            if self.knowledge_candidate_id < 0:
                raise
            logger.warning(
                f"Candidate module import failed ({module_name}.{object_name}): {exc}. "
                f"Falling back to default {module_prefix}_{self.env_name}."
            )
            return import_function(
                f"prompts.env_code.{self.dirname}.{module_prefix}_{self.env_name}",
                object_name,
            )

    def compute_reward(self, observations, tasks):
        raise NotImplementedError
    
    def planning_function(self, observations):
        raise NotImplementedError
    
    def set_func(self, planning_function, compute_reward):
        self.planning_function = planning_function
        self.compute_reward = compute_reward
        logger.critical(f"Planning function and compute reward function({self.reward_mode} mode) set")

    def set_t_env(self, t_env):
        self.t_env = t_env

    def get_total_actions(self):
        if self.use_knowledge_actions and self.knowledge_action_fn is not None:
            return self.base_n_actions + self.n_knowledge_actions
        return self.base_n_actions

    def get_avail_agent_actions(self, agent_id):
        avail_actions = super().get_avail_agent_actions(agent_id)
        if self.use_knowledge_actions and self.knowledge_action_fn is not None:
            avail_actions = avail_actions + [1] * self.n_knowledge_actions
        return avail_actions

    def _active_credit_weight(self):
        if not self.use_difference_credit:
            return 0.0
        if not self.use_dynamic_credit:
            return self.credit_weight
        if self.credit_start_step <= self.t_env <= self.credit_end_step:
            return self.credit_weight
        return 0.0

    def _to_action_list(self, actions):
        action_list = []
        for action in actions:
            if hasattr(action, "detach"):
                action = action.detach().cpu().view(-1)[0].item()
            else:
                action = np.asarray(action).reshape(-1)[0]
            action_list.append(int(action))
        return action_list

    def _map_ablation_knowledge_action(self, action):
        if self.knowledge_action_mode == "dummy":
            return 0
        if self.knowledge_action_mode == "random":
            fixed_mapping = list(range(self.base_n_actions))
            return fixed_mapping[(int(action) - self.base_n_actions) % len(fixed_mapping)]
        return None

    def _process_obs(self, observations):
        process_state = import_function(
            f"prompts.env_code.{self.dirname}.processed_obs_{self.env_name}",
            "process_state",
        )
        try:
            if self.dirname == "mpe":
                return process_state(observations, N=self.n_agents)
            if self.dirname == "lbf":
                return process_state(observations, p=self.n_agents, f=self._infer_n_food())
            if self.dirname == "rware":
                return process_state(observations, N=self.n_agents)
        except TypeError:
            pass
        return process_state(observations)

    def _infer_n_food(self):
        for part in str(self.env_name).split("_"):
            if part.endswith("f") and part[:-1].isdigit():
                return int(part[:-1])
        return self.n_agents

    def _normalise_credit_values(self, credit_values):
        values = np.asarray(credit_values, dtype=np.float32).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.size == 0:
            return values
        if not self.normalize_difference_credit:
            return values
        scale = float(np.max(np.abs(values)))
        if scale < self.credit_normalization_eps:
            return np.zeros_like(values)
        return values / scale

    def _add_task_metrics(self, info, processed_obs):
        if not isinstance(info, dict):
            return info
        if self.dirname == "mpe" and "simple_spread" in str(self.env_name):
            dists = self._simple_spread_landmark_dists(processed_obs)
            if dists:
                info["mean_landmark_dist"] = float(np.mean(dists))
                info["max_landmark_dist"] = float(np.max(dists))
                info["success_rate"] = float(
                    all(d <= self.simple_spread_success_threshold for d in dists)
                )
        elif self.dirname == "lbf":
            collection_rate = self._lbf_food_collection_rate(processed_obs)
            if collection_rate is not None:
                info["food_collection_rate"] = float(collection_rate)
                info["success_rate"] = float(collection_rate >= 1.0)
        return info

    def _simple_spread_landmark_dists(self, processed_obs):
        if not isinstance(processed_obs, dict) or not processed_obs:
            return []
        n_landmarks = self.n_agents
        min_dists = [float("inf") for _ in range(n_landmarks)]
        for agent_obs in processed_obs.values():
            values = list(agent_obs)
            for landmark_idx in range(min(n_landmarks, len(values))):
                vec = np.asarray(values[landmark_idx], dtype=np.float32).reshape(-1)
                if vec.size >= 2:
                    dist = float(np.linalg.norm(vec[:2]))
                    min_dists[landmark_idx] = min(min_dists[landmark_idx], dist)
        return [d for d in min_dists if np.isfinite(d)]

    def _lbf_food_collection_rate(self, processed_obs):
        if not (isinstance(processed_obs, tuple) and len(processed_obs) >= 1):
            return None
        food_info = processed_obs[0]
        if not isinstance(food_info, dict) or not food_info:
            return None
        total_food = len(food_info)
        collected_food = sum(1 for food in food_info.values() if food is None)
        return collected_food / max(1, total_food)

    def _map_extended_actions(self, actions, processed_obs):
        action_list = self._to_action_list(actions)
        primitive_actions = []
        knowledge_count = 0

        for agent_idx, action in enumerate(action_list):
            if (
                self.use_knowledge_actions
                and self.knowledge_action_fn is not None
                and action >= self.base_n_actions
            ):
                ablation_action = self._map_ablation_knowledge_action(action)
                if ablation_action is not None:
                    action = ablation_action
                else:
                    try:
                        action = self.knowledge_action_fn(
                            processed_obs, f"agent_{agent_idx}", action
                        )
                    except Exception as exc:
                        logger.warning(
                            f"knowledge action mapping failed for candidate {self.knowledge_candidate_id}, "
                            f"agent_{agent_idx}, action={action}: {exc}; fallback to no-op"
                        )
                        action = 0
                knowledge_count += 1
            primitive_actions.append(int(action))

        return primitive_actions, knowledge_count
    
    def step_train(self, actions):
        # print("memory", self.rware_memory)  
        prev_obs = self._obs
        processed_obs = self._process_obs(prev_obs)
        primitive_actions, knowledge_count = self._map_extended_actions(
            actions, processed_obs
        )
        obs, r, done, truncated, info = super().step(primitive_actions)

        if self.dirname == "rware":
            processed_obs, self.rware_memory = self.add_memory2obs(processed_obs, self.rware_memory, self.pre_action)

        actions_ = convert_actions(primitive_actions)
        self.pre_action = primitive_actions
        
        if self.dirname  == "lbf":
            llm_task = self.planning_function(processed_obs)
            llm_actions = lbf_task_to_actions(llm_task, processed_obs)
        elif self.dirname  == "mpe":
            llm_task = self.planning_function(processed_obs)
            llm_actions = mpe_task_to_actions(llm_task, processed_obs)
        elif self.dirname  == "rware":
            llm_task = self.planning_function(processed_obs) 
            llm_actions = rware_task_to_actions(llm_task, processed_obs)

        if  self.reward_mode == "pure":
            # pure llm reward
            reward_dict = self.compute_reward(processed_obs, actions_)
            reward = float(sum(reward_dict.values()))
        elif self.reward_mode == "mixed_constant":
            # original reward + llm constant aligned reward
            reward_dict = constant_reward_signal(
                actions_, llm_actions, llm_reward=0.001, penalty=0.001)
            reward = float(sum(reward_dict.values()))+r
        elif self.reward_mode == "mixed_normalized":
            # original reward + llm normalized code gen reward
            reward_dict = self.compute_reward(processed_obs, llm_actions, actions_)
            reward = normalized_reward(reward_dict, theta=0.01)
            reward = float(sum(reward_dict.values())) + r
        else:
            raise NotImplementedError

        credit_weight = self._active_credit_weight()
        if credit_weight > 0.0 and self.difference_credit_fn is not None:
            credit_dict = self.difference_credit_fn(processed_obs, actions_)
            raw_credit_values = list(credit_dict.values())
            credit_values = self._normalise_credit_values(raw_credit_values)
            credit_reward = float(np.mean(credit_values)) if credit_values.size > 0 else 0.0
            reward += credit_weight * credit_reward
            info["difference_credit_raw_mean"] = float(np.mean(raw_credit_values)) if raw_credit_values else 0.0
            info["difference_credit_raw_std"] = float(np.std(raw_credit_values)) if raw_credit_values else 0.0
            info["difference_credit_mean"] = credit_reward
            info["difference_credit_std"] = float(np.std(credit_values)) if credit_values.size > 0 else 0.0
            info["difference_credit_scale"] = float(np.max(np.abs(np.asarray(raw_credit_values, dtype=np.float32)))) if raw_credit_values else 0.0
        else:
            info["difference_credit_raw_mean"] = 0.0
            info["difference_credit_raw_std"] = 0.0
            info["difference_credit_mean"] = 0.0
            info["difference_credit_std"] = 0.0
            info["difference_credit_scale"] = 0.0

        info["credit_weight"] = float(credit_weight)
        info["knowledge_action_count"] = float(knowledge_count)
        info["knowledge_action_rate"] = float(knowledge_count) / max(1, self.n_agents)
        info = self._add_task_metrics(info, self._process_obs(obs))

        return obs, reward, done, truncated, info

    def step_eval(self, actions):
        processed_obs = self._process_obs(self._obs)
        primitive_actions, knowledge_count = self._map_extended_actions(
            actions, processed_obs
        )
        obs, reward, done, truncated, info = super().step(primitive_actions)
        info["knowledge_action_count"] = float(knowledge_count)
        info["knowledge_action_rate"] = float(knowledge_count) / max(1, self.n_agents)
        info["credit_weight"] = 0.0
        info["difference_credit_raw_mean"] = 0.0
        info["difference_credit_raw_std"] = 0.0
        info["difference_credit_mean"] = 0.0
        info["difference_credit_std"] = 0.0
        info["difference_credit_scale"] = 0.0
        info = self._add_task_metrics(info, self._process_obs(obs))
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        # print("New episode")
        obs, info = super().reset(seed, options)
        self.pre_action = [[] for _ in range(self.n_agents)]
        if self.dirname == "rware":
            self.rware_memory = deepcopy(self.initial_memory)  # reset memory
        return obs, info

