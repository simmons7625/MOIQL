import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
from .reward_function import RewardFunction


class DeepSeaTreasureWrapper(gym.Wrapper):
    """Wrapper for Deep Sea Treasure environment with multi-objective rewards."""

    def __init__(self, env: gym.Env, reward_fn: Optional[RewardFunction] = None):
        super().__init__(env)
        self.n_objectives = 2
        self.reward_fn = reward_fn
        self.return_scalar_reward = reward_fn is not None

        # Fix observation space dtype to float32 if needed
        if hasattr(self.observation_space, "dtype"):
            import gymnasium.spaces as spaces

            if isinstance(self.observation_space, spaces.Box):
                self.observation_space = spaces.Box(
                    low=self.observation_space.low.astype(np.float32),
                    high=self.observation_space.high.astype(np.float32),
                    shape=self.observation_space.shape,
                    dtype=np.float32,
                )

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(reward, (list, np.ndarray)):
            mo_reward = np.array(reward, dtype=np.float32)
        else:
            mo_reward = np.array([reward, -1.0], dtype=np.float32)

        # Store original multi-objective reward in info
        info["mo_reward"] = mo_reward.copy()

        # Apply reward function if provided
        if self.reward_fn:
            scalar_reward = self.reward_fn(mo_reward)
            info["preference_weights"] = self.reward_fn.preference_fn.last_weights
            return obs, scalar_reward, terminated, truncated, info

        return obs, mo_reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.reward_fn:
            self.reward_fn.reset()
        return self.env.reset(**kwargs)
