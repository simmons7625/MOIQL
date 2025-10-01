import gymnasium as gym
import numpy as np
from typing import Tuple


class DeepSeaTreasureWrapper(gym.Wrapper):
    """Wrapper for Deep Sea Treasure environment with multi-objective rewards."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_objectives = 2

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(reward, (list, np.ndarray)):
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = np.array([reward, -1.0], dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        return self.env.reset(**kwargs)


class HighwayWrapper(gym.Wrapper):
    """Wrapper for Highway environment with multi-objective rewards."""

    def __init__(self, env: gym.Env, objectives: list[str] = None):
        super().__init__(env)
        self.objectives = objectives or ["speed", "safety"]
        self.n_objectives = len(self.objectives)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        mo_reward = self._compute_multi_objective_reward(obs, reward, info)

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
        return self.env.reset(**kwargs)


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create and wrap environment based on name."""

    if env_name.lower() == "deep_sea_treasure":
        try:
            import mo_gymnasium

            env = mo_gymnasium.make("deep-sea-treasure-v0")
        except ImportError:
            raise ImportError("mo-gymnasium required for Deep Sea Treasure")
        return DeepSeaTreasureWrapper(env)

    elif env_name.lower().startswith("highway"):
        env = gym.make(env_name, **kwargs)
        objectives = kwargs.get("objectives", ["speed", "safety"])
        return HighwayWrapper(env, objectives=objectives)

    else:
        raise ValueError(f"Unknown environment: {env_name}")
