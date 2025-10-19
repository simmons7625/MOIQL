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

        # Compute nearest car distance from observation
        nearest_distance = self._get_nearest_car_distance(obs)
        info["nearest_distance"] = nearest_distance

        # Store original multi-objective reward in info
        info["mo_reward"] = mo_reward.copy()

        # Apply reward function if provided
        if self.reward_fn:
            scalar_reward = self.reward_fn(mo_reward, nearest_distance=nearest_distance)
            info["preference_weights"] = self.reward_fn.preference_fn.last_weights
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

    def _get_nearest_car_distance(self, obs: np.ndarray) -> float:
        """
        Compute distance to nearest car from observation.
        Highway-env observation shape is (num_vehicles, features).
        The ego vehicle is at index 0, other vehicles start from index 1.
        Features typically include [x, y, vx, vy, cos_h, sin_h].
        """
        if len(obs.shape) == 1:
            # Flattened observation, need to reshape
            # Highway-env default is 5 vehicles x 5 features = 25
            num_features = 5
            obs = obs.reshape(-1, num_features)

        if obs.shape[0] <= 1:
            # No other vehicles
            return float("inf")

        # Ego vehicle position (first row)
        ego_pos = obs[0, :2]  # [x, y]

        # Other vehicles positions (remaining rows)
        other_vehicles = obs[1:, :2]  # [x, y] for each vehicle

        # Compute Euclidean distances
        distances = np.linalg.norm(other_vehicles - ego_pos, axis=1)

        # Return minimum distance (or inf if no other vehicles)
        return float(np.min(distances)) if len(distances) > 0 else float("inf")

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.reward_fn:
            self.reward_fn.reset()
        return self.env.reset(**kwargs)
