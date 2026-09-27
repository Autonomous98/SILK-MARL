from .gymma import GymmaWrapper
from .task_metrics import TaskMetricsMixin


class TaskMetricWrapper(TaskMetricsMixin, GymmaWrapper):
    """Gymma wrapper that records task metrics without enabling LLM features."""

    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        env_name,
        simple_spread_success_threshold=0.15,
        **kwargs,
    ):
        super().__init__(
            key,
            time_limit,
            pretrained_wrapper,
            seed,
            common_reward,
            reward_scalarisation,
            **kwargs,
        )
        self._setup_task_metrics(
            env_name,
            simple_spread_success_threshold,
        )

    def step(self, actions):
        obs, reward, done, truncated, info = super().step(actions)
        processed_obs = self._process_obs(obs)
        info = self._add_task_metrics(info, processed_obs)
        return obs, reward, done, truncated, info
