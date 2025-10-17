import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
from .reward_function import RewardFunction


class HighwayWrapper(gym.Wrapper):
    """Wrapper for Highway environment with multi-objective rewards."""

    def __init__(
        self,
        env: gym.Env,
        objectives: list[str] = None,
        reward_fn: Optional[RewardFunction] = None,
    ):
        super().__init__(env)
        self.objectives = objectives or ["speed", "safety"]
        self.n_objectives = len(self.objectives)
        self.reward_fn = reward_fn
        self.return_scalar_reward = reward_fn is not None

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        mo_reward = self._compute_multi_objective_reward(obs, reward, info)

        # Store original multi-objective reward in info
        info["mo_reward"] = mo_reward.copy()

        # Apply reward function if provided
        if self.reward_fn:
            scalar_reward = self.reward_fn(mo_reward)
            return obs, scalar_reward, terminated, truncated, info

        return obs, mo_reward, terminated, truncated, info

    def _compute_multi_objective_reward(
        self, obs: np.ndarray, reward: float, info: dict
    ) -> np.ndarray:
        rewards = []

        for obj in self.objectives:
            if obj == "speed":
                speed_reward = info.get("speed", 0.0) / 30.0
                rewards.append(speed_reward)
            elif obj == "safety":
                collision_penalty = -1.0 if info.get("crashed", False) else 0.0
                rewards.append(collision_penalty)
            elif obj == "comfort":
                comfort_reward = -info.get("jerk", 0.0)
                rewards.append(comfort_reward)

        return np.array(rewards, dtype=np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.reward_fn:
            self.reward_fn.reset()
        return self.env.reset(**kwargs)
