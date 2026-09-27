import numpy as np

from prompts.util import import_function


class TaskMetricsMixin:
    """Environment metrics shared by YOLO-MARL and pure MARL baselines."""

    def _setup_task_metrics(self, env_name, simple_spread_success_threshold=0.15):
        self.env_name = str(env_name)
        self.dirname = self.env_name.split("_")[0]
        self.simple_spread_success_threshold = float(
            simple_spread_success_threshold
        )

    def _process_obs(self, observations):
        process_state = import_function(
            f"prompts.env_code.{self.dirname}.processed_obs_{self.env_name}",
            "process_state",
        )
        try:
            if self.dirname == "mpe":
                return process_state(observations, N=self.n_agents)
            if self.dirname == "lbf":
                return process_state(
                    observations,
                    p=self.n_agents,
                    f=self._infer_n_food(),
                )
            if self.dirname == "rware":
                return process_state(observations, N=self.n_agents)
        except TypeError:
            pass
        return process_state(observations)

    def _infer_n_food(self):
        for part in self.env_name.split("_"):
            if part.endswith("f") and part[:-1].isdigit():
                return int(part[:-1])
        return self.n_agents

    def _add_task_metrics(self, info, processed_obs):
        if not isinstance(info, dict):
            return info
        if self.dirname == "mpe" and "simple_spread" in self.env_name:
            dists = self._simple_spread_landmark_dists(processed_obs)
            if dists:
                info["mean_landmark_dist"] = float(np.mean(dists))
                info["max_landmark_dist"] = float(np.max(dists))
                info["success_rate"] = float(
                    all(
                        dist <= self.simple_spread_success_threshold
                        for dist in dists
                    )
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
                vec = np.asarray(
                    values[landmark_idx], dtype=np.float32
                ).reshape(-1)
                if vec.size >= 2:
                    dist = float(np.linalg.norm(vec[:2]))
                    min_dists[landmark_idx] = min(
                        min_dists[landmark_idx], dist
                    )
        return [dist for dist in min_dists if np.isfinite(dist)]

    def _lbf_food_collection_rate(self, processed_obs):
        if not (isinstance(processed_obs, tuple) and len(processed_obs) >= 1):
            return None
        food_info = processed_obs[0]
        if not isinstance(food_info, dict) or not food_info:
            return None
        total_food = len(food_info)
        collected_food = sum(1 for food in food_info.values() if food is None)
        return collected_food / max(1, total_food)
